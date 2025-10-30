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

/* begin lobpcg */

#define NO_SOLVER -9198

#include <time.h>

#include "HYPRE_lobpcg.h"
#include "_hypre_lobpcg.h"

/* end lobpcg */

HYPRE_Int SetStencilOffsets_1dim_3pt(HYPRE_Int  ***offsets_ptr);
HYPRE_Int SetStencilOffsets_2dim_5pt(HYPRE_Int  ***offsets_ptr);
HYPRE_Int SetStencilOffsets_3dim_7pt(HYPRE_Int  ***offsets_ptr);
HYPRE_Int SetStencilOffsets_3dim_27pt(HYPRE_Int ***offsets_ptr);
HYPRE_Int SetStencilOffsets_1dim_3pt_sym(HYPRE_Int  ***offsets_ptr);
HYPRE_Int SetStencilOffsets_2dim_5pt_sym(HYPRE_Int  ***offsets_ptr);
HYPRE_Int SetStencilOffsets_3dim_7pt_sym(HYPRE_Int  ***offsets_ptr);
HYPRE_Int SetStencilOffsets_3dim_27pt_sym(HYPRE_Int ***offsets_ptr);
HYPRE_Int SetStencilBndry(HYPRE_StructMatrix  A,
                          HYPRE_StructStencil stencil,
                          HYPRE_StructGrid    grid,
                          HYPRE_Int          *period);
HYPRE_Int SetValuesMatrix(HYPRE_StructMatrix A, HYPRE_StructGrid grid,
                          HYPRE_Real cx, HYPRE_Real cy, HYPRE_Real cz,
                          HYPRE_Real conx, HYPRE_Real cony, HYPRE_Real conz);
HYPRE_Int SetValuesCrossMatrix(HYPRE_StructMatrix A, HYPRE_StructGrid grid,
                               HYPRE_Real cx, HYPRE_Real cy, HYPRE_Real cz);
HYPRE_Int SetValuesVector(hypre_StructGrid   *grid,
                          hypre_StructVector *zvector,
                          HYPRE_Int          *period,
                          HYPRE_Int           type,
                          HYPRE_Real          value);

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
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
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

   HYPRE_Int           A_num_ghost[6] = {0, 0, 0, 0, 0, 0};
   HYPRE_Int           v_num_ghost[6] = {0, 0, 0, 0, 0, 0};

   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;
   HYPRE_StructVector  res;

   HYPRE_StructSolver  solver;
   HYPRE_StructSolver  precond;
   HYPRE_Int           num_iterations;
   HYPRE_Int           time_index;
   HYPRE_Real          final_res_norm;
   HYPRE_Real          real_res_norm;
   HYPRE_Real          rhs_norm;
   HYPRE_Real          x0_norm;
   HYPRE_Real          cf_tol;

   HYPRE_Int           num_procs, myid;
   HYPRE_Int           device_id = -1;
   HYPRE_Int           lazy_device_init = 0;

   HYPRE_Int           p = 0, q = 0, r = 0;
   HYPRE_Int           dim;
   HYPRE_Int           n_pre, n_post;
   HYPRE_Int           nblocks = 0;
   HYPRE_Int           max_levels;
   HYPRE_Int           skip;
   HYPRE_Int           sym;
   HYPRE_Int           rap;
   HYPRE_Int           matmult;
   HYPRE_Int           relax;
   HYPRE_Real          jacobi_weight;
   HYPRE_Int           usr_jacobi_weight;
   HYPRE_Int           rep, reps;
   HYPRE_Int           max_iterations;

   HYPRE_Int           rhs_type, x0_type;
   HYPRE_Real          rhs_value, x0_value;

   HYPRE_Int         **iupper;
   HYPRE_Int         **ilower;
   HYPRE_Int           istart[3];
   HYPRE_Int           periodic[3];
   HYPRE_Int         **offsets;
   HYPRE_Int           constant_coefficient = 0;
   HYPRE_Int          *stencil_entries;
   HYPRE_Int           stencil_size;
   HYPRE_Int           stencil_size_set = 0;
   HYPRE_Int           stored_stencil_size;
   HYPRE_Int           stencil_diag_entry;

   HYPRE_StructGrid    grid = NULL;
   HYPRE_StructGrid    readgrid = NULL;
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

   HYPRE_Int           prec_print_level = 0;
   HYPRE_Int           solver_print_level = 0;
   HYPRE_Int           log_level = 0;
   HYPRE_Int           print_system = 0;
   HYPRE_Int           print_vtk = 0;

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
   HYPRE_MemoryLocation   memory_location     = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy  default_exec_policy = HYPRE_EXEC_HOST;
#else
   HYPRE_MemoryLocation   memory_location     = HYPRE_MEMORY_DEVICE;
   HYPRE_ExecutionPolicy  default_exec_policy = HYPRE_EXEC_DEVICE;
#endif
   HYPRE_Int gpu_aware_mpi = 0;

#if defined (HYPRE_USING_UMPIRE)
   size_t umpire_dev_pool_size    = 4LL * 1024LL * 1024LL * 1024LL; // 4 GiB
   size_t umpire_uvm_pool_size    = 4LL * 1024LL * 1024LL * 1024LL; // 4 GiB
   size_t umpire_pinned_pool_size = 4LL * 1024LL * 1024LL * 1024LL; // 4 GiB
   size_t umpire_host_pool_size   = 4LL * 1024LL * 1024LL * 1024LL; // 4 GiB
#endif

   //HYPRE_Int device_level = -2;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );

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

   hypre_bind_device_id(device_id, myid, num_procs, comm);

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

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   dim = 3;
   stencil_size = 7;

   skip  = 0;
   sym  = 1;
   rap = 0;
   matmult = -1;
   relax = 1;
   jacobi_weight = 1.0;
   usr_jacobi_weight = 0;
   reps = 1;
   max_levels = 100;

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

   rhs_type  = 1;
   rhs_value = 1.0;
   x0_type   = 1;
   x0_value  = 0.0;

   n_pre  = 1;
   n_post = 1;

   solver_id = 1;
   solver_type = 1;
   max_iterations = 100;
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
      else if (strcmp(argv[arg_index], "-27pt") == 0)
      {
         arg_index++;
         stencil_size = 27;
         stencil_size_set = 1;
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
      else if (strcmp(argv[arg_index], "-rhszero") == 0)
      {
         arg_index++;
         rhs_type  = 1;
         rhs_value = 0.0;
      }
      else if (strcmp(argv[arg_index], "-rhsone") == 0)
      {
         arg_index++;
         rhs_type  = 1;
         rhs_value = 1.0;
      }
      else if (strcmp(argv[arg_index], "-rhsrand") == 0)
      {
         arg_index++;
         rhs_type  = 0;
         rhs_value = myid + 2747;
      }
      else if (strcmp(argv[arg_index], "-x0zero") == 0)
      {
         arg_index++;
         x0_type  = 1;
         x0_value = 0.0;
      }
      else if (strcmp(argv[arg_index], "-x0one") == 0)
      {
         arg_index++;
         x0_type  = 1;
         x0_value = 1.0;
      }
      else if (strcmp(argv[arg_index], "-x0rand") == 0)
      {
         arg_index++;
         x0_type  = 0;
      }
      else if ( strcmp(argv[arg_index], "-lvl") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-matmult") == 0 )
      {
         arg_index++;
         matmult = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         arg_index++;
         max_iterations = atoi(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-vtk") == 0 )
      {
         arg_index++;
         print_vtk = 1;
      }
      else if ( strcmp(argv[arg_index], "-pout") == 0 )
      {
         arg_index++;
         prec_print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sout") == 0 )
      {
         arg_index++;
         solver_print_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ll") == 0 )
      {
         arg_index++;
         log_level = atoi(argv[arg_index++]);
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
#if defined (HYPRE_USING_UMPIRE)
      else if ( strcmp(argv[arg_index], "-umpire_dev_pool_size") == 0 )
      {
         arg_index++;
         umpire_dev_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-umpire_uvm_pool_size") == 0 )
      {
         arg_index++;
         umpire_uvm_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-umpire_pinned_pool_size") == 0 )
      {
         arg_index++;
         umpire_pinned_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-umpire_host_pool_size") == 0 )
      {
         arg_index++;
         umpire_host_pool_size = (size_t) 1073741824 * atoi(argv[arg_index++]);
      }
#endif
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

   /* Set library log level */
   HYPRE_SetLogLevel(log_level);

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

   HYPRE_SetGpuAwareMPI(gpu_aware_mpi);

#if defined(HYPRE_USING_UMPIRE)
   /* Setup Umpire pools */
   HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE_POOL_TEST");
   HYPRE_SetUmpireUMPoolName("HYPRE_UM_POOL_TEST");
   HYPRE_SetUmpireHostPoolName("HYPRE_HOST_POOL_TEST");
   HYPRE_SetUmpirePinnedPoolName("HYPRE_PINNED_POOL_TEST");

   HYPRE_SetUmpireDevicePoolSize(umpire_dev_pool_size);
   HYPRE_SetUmpireUMPoolSize(umpire_uvm_pool_size);
   HYPRE_SetUmpireHostPoolSize(umpire_host_pool_size);
   HYPRE_SetUmpirePinnedPoolSize(umpire_pinned_pool_size);
#endif

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
      hypre_printf("  -27pt               : use 27-points stencil\n");
      hypre_printf("  -fromfile <name>    : prefix name for matrixfiles\n");
      hypre_printf("  -rhsfromfile <name> : prefix name for rhsfiles\n");
      hypre_printf("  -x0fromfile <name>  : prefix name for firstguessfiles\n");
      hypre_printf("  -rhszero            : rhs vector has zero components\n");
      hypre_printf("  -rhsone             : rhs vector has unit components\n");
      hypre_printf("  -rhsrand            : rhs vector has random components \n");
      hypre_printf("  -x0zero             : initial solution (x0) has zero components \n");
      hypre_printf("  -x0one              : initial solution (x0) has unit components \n");
      hypre_printf("  -x0rand             : initial solution (x0) has random components \n");
      hypre_printf("  -lvl <val>          : maximum number of levels (default 100);\n");
      hypre_printf("  -repeats <reps>     : number of times to repeat the run, default 1. For solver -1,0,1,3\n");
      hypre_printf("  -solver <ID>        : solver ID\n");
      hypre_printf("                       -1  - Struct Matvec\n");
      hypre_printf("                        0  - SMG (default)\n");
      hypre_printf("                        1  - PFMG\n");
      hypre_printf("                        3  - PFMG constant coeffs\n");
      hypre_printf("                        4  - PFMG constant coeffs var diag\n");
      hypre_printf("                        8  - Jacobi\n");
      hypre_printf("                        10 - CG with SMG precond\n");
      hypre_printf("                        11 - CG with PFMG precond\n");
      hypre_printf("                        13 - CG with PFMG-3 precond\n");
      hypre_printf("                        14 - CG with PFMG-4 precond\n");
      hypre_printf("                        17 - CG with 2-step Jacobi\n");
      hypre_printf("                        18 - CG with diagonal scaling\n");
      hypre_printf("                        19 - CG\n");
      hypre_printf("                        20 - Hybrid with SMG precond\n");
      hypre_printf("                        21 - Hybrid with PFMG precond\n");
      hypre_printf("                        30 - GMRES with SMG precond\n");
      hypre_printf("                        31 - GMRES with PFMG precond\n");
      hypre_printf("                        37 - GMRES with 2-step Jacobi\n");
      hypre_printf("                        38 - GMRES with diagonal scaling\n");
      hypre_printf("                        39 - GMRES\n");
      hypre_printf("                        40 - BiCGSTAB with SMG precond\n");
      hypre_printf("                        41 - BiCGSTAB with PFMG precond\n");
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
      hypre_printf("  -matmult <s>        : kernel type for structured matrix/matrix multiplication\n");
      hypre_printf("                       -1 - automatic selection (0 for CPUs, 1 for GPUs)\n");
      hypre_printf("                        0 - standard algorithm\n");
      hypre_printf("                        1 - fused algorithm\n");
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
      hypre_printf("  -tol <val>          : residual tolerance (default 1e-6)\n");
      hypre_printf("  -itr <val>          : maximal number of iterations (default 100)\n");
      hypre_printf("  -recompute <bool>   : Recompute residual in PCG?\n");
      hypre_printf("  -cf <cf>            : convergence factor for Hybrid\n");
      hypre_printf("  -print              : print out the system\n");
      hypre_printf("  -pout <val>         : print level for the preconditioner\n");
      hypre_printf("  -sout <val>         : print level for the solver\n");
      hypre_printf("  -ll <val>           : hypre's log level\n");
      hypre_printf("                        0 - (default) No messaging.\n");
      hypre_printf("                        1 - Display memory usage statistics for each MPI rank.\n");
      hypre_printf("                        2 - Display aggregate memory usage statistics over MPI ranks.\n");
#if defined (HYPRE_USING_UMPIRE)
      hypre_printf("  -umpire_dev_pool_size <val>      : device memory pool size (GiB)\n");
      hypre_printf("  -umpire_uvm_pool_size <val>      : device unified virtual memory pool size (GiB)\n");
      hypre_printf("  -umpire_pinned_pool_size <val>   : pinned memory pool size (GiB)\n");
      hypre_printf("  -umpire_host_pool_size <val>     : host memory pool size (GiB)\n");
#endif
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
      exit(0);
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
      hypre_MPI_Abort(comm, 1);
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

   /* Calculate stencil size */
   if (!stencil_size_set)
   {
      switch (dim)
      {
         case 3:
            stencil_size = 7;
            break;

         case 2:
            stencil_size = 5;
            break;

         case 1:
            stencil_size = 3;
            break;

         default:
            if (myid == 0)
            {
               hypre_printf("Unsupported dimension: %d\n", dim);
               hypre_MPI_Abort(comm, 1);
            }
      }
   }
   stored_stencil_size = (stencil_size - sym) / (1 + sym) + sym;

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
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
      if (sum == 0)
      {
         hypre_printf("  (nx, ny, nz)     = (%d, %d, %d)\n", nx, ny, nz);
         hypre_printf("  (ix, iy, iz)     = (%d, %d, %d)\n", istart[0], istart[1], istart[2]);
         hypre_printf("  (Px, Py, Pz)     = (%d, %d, %d)\n", P,  Q,  R);
         hypre_printf("  (bx, by, bz)     = (%d, %d, %d)\n", bx, by, bz);
         hypre_printf("  (px, py, pz)     = (%d, %d, %d)\n", px, py, pz);
      }
      else
      {
         hypre_printf("  the grid is read from file  \n");
      }
      {
         hypre_printf("  (cx, cy, cz)     = (%f, %f, %f)\n", cx, cy, cz);
         hypre_printf("  (conx,cony,conz) = (%f, %f, %f)\n", conx, cony, conz);
         hypre_printf("  (n_pre, n_post)  = (%d, %d)\n", n_pre, n_post);
         hypre_printf("  stsize           = %d\n", stencil_size);
         hypre_printf("  stored_stsize    = %d\n", stored_stencil_size);
         hypre_printf("  dim              = %d\n", dim);
         hypre_printf("  skip             = %d\n", skip);
         hypre_printf("  sym              = %d\n", sym);
         hypre_printf("  rap              = %d\n", rap);
         hypre_printf("  matmult          = %d\n", matmult);
         hypre_printf("  relax            = %d\n", relax);
         hypre_printf("  solver ID        = %d\n", solver_id);
         hypre_printf("  Repetitions      = %d\n", reps);
      }
      if (rhs_type == 0)
      {
         hypre_printf("  rhs value        = %20.15e\n", rhs_value);
      }
      else
      {
         hypre_printf("  rhs has random components\n");
      }
      if (x0_type == 0)
      {
         hypre_printf("  initial sol (x0) = %20.15e\n", x0_value);
      }
      else
      {
         hypre_printf("  initial sol (x0) has random components\n");
      }
      hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   hypre_MPI_Barrier(comm);

   for (rep = 0; rep < reps; ++rep)
   {
      time_index = hypre_InitializeTiming("Struct Interface");
      hypre_BeginTiming(time_index);

      /*-----------------------------------------------------------
       * Set up the stencil structure when matrix is NOT read from file
       * Set up the grid structure used when NO files are read
       *-----------------------------------------------------------*/

      switch (dim)
      {
         case 1:
            nblocks = bx;
            if (sym && stencil_size == 3)
            {
               SetStencilOffsets_1dim_3pt_sym(&offsets);
            }
            else if (!sym && stencil_size == 3)
            {
               SetStencilOffsets_1dim_3pt(&offsets);
            }
            else
            {
               hypre_printf("Stencil size not implemented!\n");
               hypre_MPI_Abort(comm, 1);
            }

            /* compute p from P and myid */
            p = myid % P;
            break;

         case 2:
            nblocks = bx * by;
            if (sym && stencil_size == 5)
            {
               SetStencilOffsets_2dim_5pt_sym(&offsets);
            }
            else if (!sym && stencil_size == 5)
            {
               SetStencilOffsets_2dim_5pt(&offsets);
            }
            else
            {
               hypre_printf("Stencil size not implemented!\n");
               hypre_MPI_Abort(comm, 1);
            }

            /* compute p,q from P,Q and myid */
            p = myid % P;
            q = (( myid - p) / P) % Q;
            break;

         case 3:
            nblocks = bx * by * bz;
            if (sym && stencil_size == 7)
            {
               SetStencilOffsets_3dim_7pt_sym(&offsets);
            }
            else if (!sym && stencil_size == 7)
            {
               SetStencilOffsets_3dim_7pt(&offsets);
            }
            else if (sym && stencil_size == 27)
            {
               SetStencilOffsets_3dim_27pt_sym(&offsets);
            }
            else if (!sym && stencil_size == 27)
            {
               SetStencilOffsets_3dim_27pt(&offsets);
            }
            else
            {
               hypre_printf("Stencil size not implemented!\n");
               hypre_MPI_Abort(comm, 1);
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

      HYPRE_StructStencilCreate(dim, stored_stencil_size, &stencil);
      for (s = 0; s < stored_stencil_size; s++)
      {
         HYPRE_StructStencilSetEntry(stencil, s, offsets[s]);
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

      if ( (read_fromfile_param    == 1) &&
           (read_x0fromfile_param  == 1) &&
           (read_rhsfromfile_param == 1))
      {
         if (myid == 0)
         {
            hypre_printf("\nReading linear system from files: matrix, rhs and x0\n");
         }

         /* ghost selection for reading the matrix and vectors */
         for (i = 0; i < dim; i++)
         {
            A_num_ghost[2 * i] = 1;
            A_num_ghost[2 * i + 1] = 1;
            v_num_ghost[2 * i] = 1;
            v_num_ghost[2 * i + 1] = 1;
         }

         HYPRE_StructMatrixRead(comm, argv[read_fromfile_index], A_num_ghost, &A);

         HYPRE_StructVectorRead(comm, argv[read_rhsfromfile_index], v_num_ghost, &b);

         HYPRE_StructVectorRead(comm, argv[read_x0fromfile_index], v_num_ghost, &x);

         readgrid     = hypre_StructMatrixGrid(A);
         readperiodic = hypre_StructGridPeriodic(readgrid);
      }

      /* beginning of sum == 0  */
      if (sum == 0)    /* no read from any file */
      {
         /*-----------------------------------------------------------
          * prepare space for the extents
          *-----------------------------------------------------------*/

         ilower = hypre_CTAlloc(HYPRE_Int*, nblocks, HYPRE_MEMORY_HOST);
         iupper = hypre_CTAlloc(HYPRE_Int*, nblocks, HYPRE_MEMORY_HOST);
         for (i = 0; i < nblocks; i++)
         {
            ilower[i] = hypre_CTAlloc(HYPRE_Int, dim, HYPRE_MEMORY_HOST);
            iupper[i] = hypre_CTAlloc(HYPRE_Int, dim, HYPRE_MEMORY_HOST);
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
               {
                  for (ix = 0; ix < bx; ix++)
                  {
                     ilower[ib][0] = istart[0] + nx * (bx * p + ix);
                     iupper[ib][0] = istart[0] + nx * (bx * p + ix + 1) - 1;
                     ilower[ib][1] = istart[1] + ny * (by * q + iy);
                     iupper[ib][1] = istart[1] + ny * (by * q + iy + 1) - 1;
                     ib++;
                  }
               }
               break;

            case 3:
               for (iz = 0; iz < bz; iz++)
               {
                  for (iy = 0; iy < by; iy++)
                  {
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
                  }
               }
               break;
         }

         HYPRE_StructGridCreate(comm, dim, &grid);
         for (ib = 0; ib < nblocks; ib++)
         {
            /* Add to the grid a new box defined by ilower[ib], iupper[ib]...*/
            HYPRE_StructGridSetExtents(grid, ilower[ib], iupper[ib]);
         }
         HYPRE_StructGridSetPeriodic(grid, periodic);
         HYPRE_StructGridAssemble(grid);

         if (print_vtk)
         {
            HYPRE_StructGridPrintVTK("struct_grid", grid);
         }

         /*-----------------------------------------------------------
          * Set up the matrix structure
          *-----------------------------------------------------------*/

         HYPRE_StructMatrixCreate(comm, grid, stencil, &A);

         if ( solver_id == 3  || solver_id == 4 ||
              solver_id == 13 || solver_id == 14 )
         {
            stencil_size    = hypre_StructStencilSize(stencil);
            stencil_entries = hypre_CTAlloc(HYPRE_Int, stencil_size,
                                            HYPRE_MEMORY_HOST);
            if (solver_id == 3 || solver_id == 13)
            {
               for (i = 0; i < stencil_size; ++i)
               {
                  stencil_entries[i] = i;
               }
               /* SetConstantEntries is where the constant_coefficient flag is set in A */
               hypre_StructMatrixSetConstantEntries(A, stencil_size, stencil_entries);
               constant_coefficient = 1;
            }
            else
            {
               stencil_diag_entry = hypre_StructStencilDiagEntry(stencil);
               hypre_assert(stencil_size >= 1);
               if (stencil_diag_entry == 0)
               {
                  stencil_entries[stencil_diag_entry] = 1;
               }
               else
               {
                  stencil_entries[stencil_diag_entry] = 0;
               }
               for (i = 0; i < stencil_size; ++i)
               {
                  if (i != stencil_diag_entry)
                  {
                     stencil_entries[i] = i;
                  }
               }
               hypre_StructMatrixSetConstantEntries(A, stencil_size, stencil_entries);
               constant_coefficient = 2;
            }
            hypre_TFree(stencil_entries, HYPRE_MEMORY_HOST);
         }
         HYPRE_StructMatrixSetSymmetric(A, sym);
         HYPRE_StructMatrixInitialize(A);

         /*-----------------------------------------------------------
          * Fill in the matrix elements
          *-----------------------------------------------------------*/

         if (stencil_size != 27)
         {
            SetValuesMatrix(A, grid, cx, cy, cz, conx, cony, conz);
         }
         else
         {
            SetValuesCrossMatrix(A, grid, cx, cy, cz);
         }

         /* Zero out stencils reaching to real boundary.
            But in constant coefficient case, no special stencils! */
         if (constant_coefficient == 0)
         {
            SetStencilBndry(A, stencil, grid, periodic);
         }

         /* Assemble matrix */
         HYPRE_StructMatrixAssemble(A);

         /*-----------------------------------------------------------
          * Set up the linear system
          *-----------------------------------------------------------*/

         HYPRE_StructVectorCreate(comm, grid, &b);
         HYPRE_StructVectorInitialize(b);

         /*-----------------------------------------------------------
          * Set vector entries
          *-----------------------------------------------------------*/

         SetValuesVector(grid, b, periodic, rhs_type, rhs_value);
         HYPRE_StructVectorAssemble(b);

         HYPRE_StructVectorCreate(comm, grid, &x);
         HYPRE_StructVectorInitialize(x);

         SetValuesVector(grid, x, periodx0, x0_type, x0_value);
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
      else if ((sum > 0 ) && (sum < 3))
      {
         /* the grid will be read from file.  */
         /* the grid will come from rhs or from x0 */
         if (read_fromfile_param == 0)
         {
            if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param == 0))
            {
               /* read right hand side, extract grid, construct matrix,
                  construct x0 */

               hypre_printf("\ninitial rhs from file prefix :%s\n",
                            argv[read_rhsfromfile_index]);

               HYPRE_StructVectorRead(comm, argv[read_rhsfromfile_index], v_num_ghost, &b);

               readgrid = hypre_StructVectorGrid(b) ;
               readperiodic = hypre_StructGridPeriodic(readgrid);

               HYPRE_StructVectorCreate(comm, readgrid, &x);
               HYPRE_StructVectorInitialize(x);

               SetValuesVector(readgrid, x, periodx0, x0_type, x0_value);
               HYPRE_StructVectorAssemble(x);

               HYPRE_StructMatrixCreate(comm,
                                        readgrid, stencil, &A);
               HYPRE_StructMatrixSetSymmetric(A, 1);
               HYPRE_StructMatrixInitialize(A);

               /*-----------------------------------------------------------
                * Fill in the matrix elements
                *-----------------------------------------------------------*/

               if (stencil_size != 27)
               {
                  SetValuesMatrix(A, readgrid, cx, cy, cz, conx, cony, conz);
               }
               else
               {
                  SetValuesCrossMatrix(A, readgrid, cx, cy, cz);
               }

               /* Zero out stencils reaching to real boundary */
               if (constant_coefficient == 0)
               {
                  SetStencilBndry(A, stencil, readgrid, readperiodic);
               }
               HYPRE_StructMatrixAssemble(A);
            }
            else if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param > 0))
            {
               /* read right hand side, extract grid, construct matrix,
                  construct x0 */

               hypre_printf("\ninitial x0 from file prefix :%s\n",
                            argv[read_x0fromfile_index]);

               HYPRE_StructVectorRead(comm, argv[read_x0fromfile_index], v_num_ghost, &x);

               readgrid = hypre_StructVectorGrid(x) ;
               readperiodic = hypre_StructGridPeriodic(readgrid);

               HYPRE_StructVectorCreate(comm, readgrid, &b);
               HYPRE_StructVectorInitialize(b);
               SetValuesVector(readgrid, b, readperiodic, rhs_type, rhs_value);

               HYPRE_StructVectorAssemble(b);

               HYPRE_StructMatrixCreate(comm,
                                        readgrid, stencil, &A);
               HYPRE_StructMatrixSetSymmetric(A, 1);
               HYPRE_StructMatrixInitialize(A);

               /*-----------------------------------------------------------
                * Fill in the matrix elements
                *-----------------------------------------------------------*/

               if (stencil_size != 27)
               {
                  SetValuesMatrix(A, readgrid, cx, cy, cz, conx, cony, conz);
               }
               else
               {
                  SetValuesCrossMatrix(A, readgrid, cx, cy, cz);
               }

               /* Zero out stencils reaching to real boundary */
               if (constant_coefficient == 0)
               {
                  SetStencilBndry(A, stencil, readgrid, readperiodic);
               }
               HYPRE_StructMatrixAssemble(A);
            }
            else if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param > 0))
            {
               /* read right hand side, extract grid, construct matrix,
                  construct x0 */

               hypre_printf("\ninitial rhs  from file prefix :%s\n",
                            argv[read_rhsfromfile_index]);
               hypre_printf("\ninitial x0  from file prefix :%s\n",
                            argv[read_x0fromfile_index]);

               HYPRE_StructVectorRead(comm, argv[read_rhsfromfile_index], v_num_ghost, &b);

               HYPRE_StructVectorRead(comm, argv[read_x0fromfile_index], v_num_ghost, &x);

               readgrid = hypre_StructVectorGrid(b);
               readperiodic = hypre_StructGridPeriodic(readgrid);

               HYPRE_StructMatrixCreate(comm,
                                        readgrid, stencil, &A);
               HYPRE_StructMatrixSetSymmetric(A, 1);
               HYPRE_StructMatrixInitialize(A);

               /*-----------------------------------------------------------
                * Fill in the matrix elements
                *-----------------------------------------------------------*/

               if (stencil_size != 27)
               {
                  SetValuesMatrix(A, readgrid, cx, cy, cz, conx, cony, conz);
               }
               else
               {
                  SetValuesCrossMatrix(A, readgrid, cx, cy, cz);
               }

               /* Zero out stencils reaching to real boundary */
               if (constant_coefficient == 0)
               {
                  SetStencilBndry(A, stencil, readgrid, readperiodic);
               }
               HYPRE_StructMatrixAssemble(A);
            }
            /* done with one case rhs=1 x0 = 1  */
         }
         /* done with the case where you do not read matrix from file */

         if (read_fromfile_param == 1)  /* still sum > 0  */
         {
            hypre_printf("\nreading matrix from file:%s\n",
                         argv[read_fromfile_index]);

            HYPRE_StructMatrixRead(comm, argv[read_fromfile_index], A_num_ghost, &A);

            readgrid     = hypre_StructMatrixGrid(A);
            readperiodic = hypre_StructGridPeriodic(readgrid);

            if ((read_rhsfromfile_param > 0) && (read_x0fromfile_param == 0))
            {
               /* read right hand side ,construct x0 */
               hypre_printf("\ninitial rhs from file prefix :%s\n",
                            argv[read_rhsfromfile_index]);

               HYPRE_StructVectorRead(comm, argv[read_rhsfromfile_index], v_num_ghost, &b);

               HYPRE_StructVectorCreate(comm, readgrid, &x);
               HYPRE_StructVectorInitialize(x);

               SetValuesVector(readgrid, x, periodx0, x0_type, x0_value);
               HYPRE_StructVectorAssemble(x);
            }
            else if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param > 0))
            {
               /* read x0, construct rhs*/
               hypre_printf("\ninitial x0 from file prefix :%s\n",
                            argv[read_x0fromfile_index]);

               HYPRE_StructVectorRead(comm, argv[read_x0fromfile_index], v_num_ghost, &x);

               HYPRE_StructVectorCreate(comm, readgrid, &b);
               HYPRE_StructVectorInitialize(b);
               SetValuesVector(readgrid, b, readperiodic, rhs_type, rhs_value);
               HYPRE_StructVectorAssemble(b);
            }
            else if ((read_rhsfromfile_param == 0) && (read_x0fromfile_param == 0))
            {
               /* construct x0 , construct b*/
               HYPRE_StructVectorCreate(comm, readgrid, &b);
               HYPRE_StructVectorInitialize(b);
               SetValuesVector(readgrid, b, readperiodic, rhs_type, rhs_value);
               HYPRE_StructVectorAssemble(b);


               HYPRE_StructVectorCreate(comm, readgrid, &x);
               HYPRE_StructVectorInitialize(x);
               SetValuesVector(readgrid, x, periodx0, x0_type, x0_value);
               HYPRE_StructVectorAssemble(x);
            }
         } /* finish the read of matrix */
      }
      /* linear system complete */

      /* RDF: Why do we need both a readgrid and a grid? */
      if (grid == NULL)
      {
         grid = readgrid;
      }

      hypre_EndTiming(time_index);
      if (reps == 1 || (solver_id != 0 && solver_id != 1 && solver_id != 3 && solver_id != 4))
      {
         hypre_PrintTiming("Struct Interface", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
      }
      else if (rep == reps - 1)
      {
         hypre_FinalizeTiming(time_index);
      }

      /* Compute vector norms */
      HYPRE_StructVectorInnerProd(b, b, &rhs_norm);
      HYPRE_StructVectorInnerProd(x, x, &x0_norm);
      rhs_norm = sqrt(rhs_norm);
      x0_norm = sqrt(x0_norm);

      /*-----------------------------------------------------------
       * Print out the system and initial guess
       *-----------------------------------------------------------*/

      if (print_system)
      {
         HYPRE_StructMatrixPrint("struct.out.A", A, 0);
         HYPRE_StructVectorPrint("struct.out.b", b, 0);
         HYPRE_StructVectorPrint("struct.out.x0", x, 0);
      }

      if (print_vtk)
      {
         HYPRE_StructGridPrintVTK("struct_grid", hypre_StructMatrixGrid(A));
         //HYPRE_StructGridPrintVTK("struct_grid", grid);
      }

      /*-----------------------------------------------------------
       * Solve the system using SMG
       *-----------------------------------------------------------*/

#if !HYPRE_MFLOPS

      if (solver_id < 0)
      {
         void  *matvec_data;

         matvec_data = hypre_StructMatvecCreate();
         hypre_StructMatvecSetup(matvec_data, A, x);

         time_index = hypre_InitializeTiming("Matvec");
         hypre_BeginTiming(time_index);

         for (i = 0; i < reps; i++)
         {
            hypre_StructMatvecCompute(matvec_data, -1.0, A, x, 1.0, b, b);
         }
         reps = 0;

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Total Matvec time", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         hypre_StructMatvecDestroy(matvec_data);
      }
      else if (solver_id == 0)
      {
         time_index = hypre_InitializeTiming("SMG Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructSMGCreate(comm, &solver);
         HYPRE_StructSMGSetMemoryUse(solver, 0);
         HYPRE_StructSMGSetMaxIter(solver, max_iterations);
         HYPRE_StructSMGSetTol(solver, tol);
         HYPRE_StructSMGSetRelChange(solver, 0);
         HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
         HYPRE_StructSMGSetNumPostRelax(solver, n_post);
         HYPRE_StructSMGSetPrintLevel(solver, solver_print_level);
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
         if (reps == 1)
         {
            hypre_PrintTiming("Setup phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if (rep == reps - 1)
         {
            hypre_FinalizeTiming(time_index);
         }

         time_index = hypre_InitializeTiming("SMG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_StructSMGSolve(solver, A, b, x);

         hypre_EndTiming(time_index);
         if (reps == 1)
         {
            hypre_PrintTiming("Solve phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if (rep == reps - 1)
         {
            hypre_PrintTiming("Interface, Setup, and Solve times:",
                              comm );
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

         HYPRE_StructPFMGCreate(comm, &solver);
         HYPRE_StructPFMGSetMaxLevels(solver, max_levels);
         HYPRE_StructPFMGSetMaxIter(solver, max_iterations);
         HYPRE_StructPFMGSetTol(solver, tol);
         HYPRE_StructPFMGSetRelChange(solver, 0);
         HYPRE_StructPFMGSetRAPType(solver, rap);
         HYPRE_StructPFMGSetMatmultType(solver, matmult);
         HYPRE_StructPFMGSetRelaxType(solver, relax);
         if (usr_jacobi_weight)
         {
            HYPRE_StructPFMGSetJacobiWeight(solver, jacobi_weight);
         }
         HYPRE_StructPFMGSetNumPreRelax(solver, n_pre);
         HYPRE_StructPFMGSetNumPostRelax(solver, n_post);
         HYPRE_StructPFMGSetSkipRelax(solver, skip);
         /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
         HYPRE_StructPFMGSetPrintLevel(solver, solver_print_level);
         HYPRE_StructPFMGSetLogging(solver, 1);

#if 0//defined(HYPRE_USING_CUDA)
         HYPRE_StructPFMGSetDeviceLevel(solver, device_level);
#endif

         HYPRE_StructPFMGSetup(solver, A, b, x);

         hypre_EndTiming(time_index);
         if (reps == 1)
         {
            hypre_PrintTiming("Setup phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if (rep == reps - 1)
         {
            hypre_FinalizeTiming(time_index);
         }

         time_index = hypre_InitializeTiming("PFMG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_StructPFMGSolve(solver, A, b, x);

         hypre_EndTiming(time_index);
         if (reps == 1)
         {
            hypre_PrintTiming("Solve phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }
         else if (rep == reps - 1)
         {
            hypre_PrintTiming("Interface, Setup, and Solve times",
                              comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();
         }

         HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
         HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_StructPFMGDestroy(solver);
      }

      /*-----------------------------------------------------------
       * Solve the system using Jacobi
       *-----------------------------------------------------------*/

      else if ( solver_id == 8 )
      {
         time_index = hypre_InitializeTiming("Jacobi Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructJacobiCreate(comm, &solver);
         HYPRE_StructJacobiSetMaxIter(solver, max_iterations);
         HYPRE_StructJacobiSetTol(solver, tol);
         HYPRE_StructJacobiSetup(solver, A, b, x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("Jacobi Solve");
         hypre_BeginTiming(time_index);

         HYPRE_StructJacobiSolve(solver, A, b, x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
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

         HYPRE_StructPCGCreate(comm, &solver);
         HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, max_iterations );
         HYPRE_PCGSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
         HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
         HYPRE_PCGSetPrintLevel( (HYPRE_Solver)solver, solver_print_level );

         if (solver_id == 10)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(comm, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
            HYPRE_StructPFMGCreate(comm, &precond);
            HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetMatmultType(precond, matmult);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0//defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                 (HYPRE_Solver) precond);
         }

         else if (solver_id == 17)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(comm, &precond);
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
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);

         HYPRE_PCGSolve( (HYPRE_Solver) solver,
                         (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
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

            HYPRE_StructPCGCreate(comm, &solver);
            HYPRE_PCGSetMaxIter( (HYPRE_Solver)solver, pcgIterations );
            HYPRE_PCGSetTol( (HYPRE_Solver)solver, pcgTol );
            HYPRE_PCGSetTwoNorm( (HYPRE_Solver)solver, 1 );
            HYPRE_PCGSetRelChange( (HYPRE_Solver)solver, 0 );
            HYPRE_PCGSetPrintLevel( (HYPRE_Solver)solver, solver_print_level );

            if (solver_id == 10)
            {
               /* use symmetric SMG as preconditioner */
               HYPRE_StructSMGCreate(comm, &precond);
               HYPRE_StructSMGSetMemoryUse(precond, 0);
               HYPRE_StructSMGSetMaxIter(precond, 1);
               HYPRE_StructSMGSetTol(precond, 0.0);
               HYPRE_StructSMGSetZeroGuess(precond);
               HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructSMGSetNumPostRelax(precond, n_post);
               HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
               HYPRE_StructPFMGCreate(comm, &precond);
               HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
               HYPRE_StructPFMGSetMaxIter(precond, 1);
               HYPRE_StructPFMGSetTol(precond, 0.0);
               HYPRE_StructPFMGSetZeroGuess(precond);
               HYPRE_StructPFMGSetRAPType(precond, rap);
               HYPRE_StructPFMGSetMatmultType(precond, matmult);
               HYPRE_StructPFMGSetRelaxType(precond, relax);
               if (usr_jacobi_weight)
               {
                  HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
               }
               HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
               HYPRE_StructPFMGSetSkipRelax(precond, skip);
               /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
               HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
               HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
               HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
               HYPRE_PCGSetPrecond( (HYPRE_Solver) solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                    (HYPRE_Solver) precond);
            }

            else if (solver_id == 17)
            {
               /* use two-step Jacobi as preconditioner */
               HYPRE_StructJacobiCreate(comm, &precond);
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
            hypre_PrintTiming("Setup phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&lobpcg_solver);
            HYPRE_LOBPCGSetMaxIter((HYPRE_Solver)lobpcg_solver, max_iterations);
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
            eigenvalues = hypre_CTAlloc( HYPRE_Real, blockSize, HYPRE_MEMORY_HOST);

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
            hypre_PrintTiming("Solve phase times", comm);
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
            else if (solver_id == 17)
            {
               HYPRE_StructJacobiDestroy(precond);
            }

            HYPRE_LOBPCGDestroy((HYPRE_Solver)lobpcg_solver);
            mv_MultiVectorDestroy( eigenvectors );
            hypre_TFree( eigenvalues, HYPRE_MEMORY_HOST);
         }
         else
         {
            time_index = hypre_InitializeTiming("LOBPCG Setup");
            hypre_BeginTiming(time_index);

            HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (HYPRE_Solver*)&solver);
            HYPRE_LOBPCGSetMaxIter( (HYPRE_Solver)solver, max_iterations );
            HYPRE_LOBPCGSetTol( (HYPRE_Solver)solver, tol );
            HYPRE_LOBPCGSetPrintLevel( (HYPRE_Solver)solver, verbosity );

            if (solver_id == 10)
            {
               /* use symmetric SMG as preconditioner */
               HYPRE_StructSMGCreate(comm, &precond);
               HYPRE_StructSMGSetMemoryUse(precond, 0);
               HYPRE_StructSMGSetMaxIter(precond, 1);
               HYPRE_StructSMGSetTol(precond, 0.0);
               HYPRE_StructSMGSetZeroGuess(precond);
               HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructSMGSetNumPostRelax(precond, n_post);
               HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
               HYPRE_StructPFMGCreate(comm, &precond);
               HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
               HYPRE_StructPFMGSetMaxIter(precond, 1);
               HYPRE_StructPFMGSetTol(precond, 0.0);
               HYPRE_StructPFMGSetZeroGuess(precond);
               HYPRE_StructPFMGSetRAPType(precond, rap);
               HYPRE_StructPFMGSetMatmultType(precond, matmult);
               HYPRE_StructPFMGSetRelaxType(precond, relax);
               if (usr_jacobi_weight)
               {
                  HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
               }
               HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
               HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
               HYPRE_StructPFMGSetSkipRelax(precond, skip);
               /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
               HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
               HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
               HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
               HYPRE_LOBPCGSetPrecond( (HYPRE_Solver) solver,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                       (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                       (HYPRE_Solver) precond);
            }

            else if (solver_id == 17)
            {
               /* use two-step Jacobi as preconditioner */
               HYPRE_StructJacobiCreate(comm, &precond);
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
            hypre_PrintTiming("Setup phase times", comm);
            hypre_FinalizeTiming(time_index);
            hypre_ClearTiming();

            eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                                 blockSize,
                                                                 x );
            eigenvalues = hypre_CTAlloc( HYPRE_Real, blockSize, HYPRE_MEMORY_HOST);

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
            hypre_PrintTiming("Solve phase times", comm);
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
            else if (solver_id == 17)
            {
               HYPRE_StructJacobiDestroy(precond);
            }

            mv_MultiVectorDestroy( eigenvectors );
            hypre_TFree( eigenvalues, HYPRE_MEMORY_HOST);
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

         HYPRE_StructHybridCreate(comm, &solver);
         HYPRE_StructHybridSetDSCGMaxIter(solver, max_iterations);
         HYPRE_StructHybridSetPCGMaxIter(solver, max_iterations);
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
         HYPRE_StructHybridSetPrintLevel(solver, solver_print_level);
         HYPRE_StructHybridSetLogging(solver, 1);
         HYPRE_StructHybridSetSolverType(solver, solver_type);
         HYPRE_StructHybridSetRecomputeResidual(solver, recompute_res);

         if (solver_id == 20)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(comm, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
            HYPRE_StructPFMGCreate(comm, &precond);
            HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetMatmultType(precond, matmult);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_StructHybridSetPrecond(solver,
                                         HYPRE_StructPFMGSolve,
                                         HYPRE_StructPFMGSetup,
                                         precond);
         }

         HYPRE_StructHybridSetup(solver, A, b, x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("Hybrid Solve");
         hypre_BeginTiming(time_index);

         HYPRE_StructHybridSolve(solver, A, b, x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
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
      }

      /*-----------------------------------------------------------
       * Solve the system using GMRES
       *-----------------------------------------------------------*/

      if ((solver_id > 29) && (solver_id < 40))
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);

         HYPRE_StructGMRESCreate(comm, &solver);
         HYPRE_GMRESSetKDim( (HYPRE_Solver) solver, 5 );
         HYPRE_GMRESSetMaxIter( (HYPRE_Solver)solver, max_iterations );
         HYPRE_GMRESSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_GMRESSetRelChange( (HYPRE_Solver)solver, 0 );
         HYPRE_GMRESSetPrintLevel( (HYPRE_Solver)solver, solver_print_level );
         HYPRE_GMRESSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 30)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(comm, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
            HYPRE_StructPFMGCreate(comm, &precond);
            HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetMatmultType(precond, matmult);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_GMRESSetPrecond( (HYPRE_Solver)solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                   (HYPRE_Solver)precond);
         }

         else if (solver_id == 37)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(comm, &precond);
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
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("GMRES Solve");
         hypre_BeginTiming(time_index);

         HYPRE_GMRESSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
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

         HYPRE_StructBiCGSTABCreate(comm, &solver);
         HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver)solver, max_iterations );
         HYPRE_BiCGSTABSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver)solver, solver_print_level );
         HYPRE_BiCGSTABSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 40)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(comm, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
            HYPRE_StructPFMGCreate(comm, &precond);
            HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetMatmultType(precond, matmult);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
            HYPRE_StructPFMGSetLogging(precond, 0);
#if 0 //defined(HYPRE_USING_CUDA)
            HYPRE_StructPFMGSetDeviceLevel(precond, device_level);
#endif
            HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver)solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSolve,
                                      (HYPRE_PtrToSolverFcn) HYPRE_StructPFMGSetup,
                                      (HYPRE_Solver)precond);
         }

         else if (solver_id == 47)
         {
            /* use two-step Jacobi as preconditioner */
            HYPRE_StructJacobiCreate(comm, &precond);
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
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("BiCGSTAB Solve");
         hypre_BeginTiming(time_index);

         HYPRE_BiCGSTABSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
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

         HYPRE_StructLGMRESCreate(comm, &solver);
         HYPRE_LGMRESSetKDim( (HYPRE_Solver) solver, 5 );
         HYPRE_LGMRESSetMaxIter( (HYPRE_Solver)solver, max_iterations );
         HYPRE_LGMRESSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_LGMRESSetPrintLevel( (HYPRE_Solver)solver, solver_print_level );
         HYPRE_LGMRESSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 50)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(comm, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
            HYPRE_StructPFMGCreate(comm, &precond);
            HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetMatmultType(precond, matmult);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
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
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("LGMRES Solve");
         hypre_BeginTiming(time_index);

         HYPRE_LGMRESSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
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

         HYPRE_StructFlexGMRESCreate(comm, &solver);
         HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, 5 );
         HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver)solver, max_iterations );
         HYPRE_FlexGMRESSetTol( (HYPRE_Solver)solver, tol );
         HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver)solver, solver_print_level );
         HYPRE_FlexGMRESSetLogging( (HYPRE_Solver)solver, 1 );

         if (solver_id == 60)
         {
            /* use symmetric SMG as preconditioner */
            HYPRE_StructSMGCreate(comm, &precond);
            HYPRE_StructSMGSetMemoryUse(precond, 0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetZeroGuess(precond);
            HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructSMGSetNumPostRelax(precond, n_post);
            HYPRE_StructSMGSetPrintLevel(precond, prec_print_level);
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
            HYPRE_StructPFMGCreate(comm, &precond);
            HYPRE_StructPFMGSetMaxLevels(precond, max_levels);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetZeroGuess(precond);
            HYPRE_StructPFMGSetRAPType(precond, rap);
            HYPRE_StructPFMGSetMatmultType(precond, matmult);
            HYPRE_StructPFMGSetRelaxType(precond, relax);
            if (usr_jacobi_weight)
            {
               HYPRE_StructPFMGSetJacobiWeight(precond, jacobi_weight);
            }
            HYPRE_StructPFMGSetNumPreRelax(precond, n_pre);
            HYPRE_StructPFMGSetNumPostRelax(precond, n_post);
            HYPRE_StructPFMGSetSkipRelax(precond, skip);
            /*HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            HYPRE_StructPFMGSetPrintLevel(precond, prec_print_level);
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
         hypre_PrintTiming("Setup phase times", comm);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();

         time_index = hypre_InitializeTiming("FlexGMRES Solve");
         hypre_BeginTiming(time_index);

         HYPRE_FlexGMRESSolve
         ( (HYPRE_Solver)solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", comm);
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

      /* Compute residual */
      HYPRE_StructVectorCreate(comm, grid, &res);
      HYPRE_StructVectorInitialize(res);
      HYPRE_StructVectorAssemble(res);
      HYPRE_StructVectorCopy(b, res);
      HYPRE_StructMatrixMatvec(-1.0, A, x, 1.0, res);
      HYPRE_StructVectorInnerProd(res, res, &real_res_norm);
      real_res_norm = sqrt(real_res_norm);
      if (rhs_norm > 0)
      {
         real_res_norm = real_res_norm / rhs_norm;
      }

      /*-----------------------------------------------------------
       * Print the solution and other info
       *-----------------------------------------------------------*/

      if (print_system)
      {
         HYPRE_StructVectorPrint("struct.out.x", x, 0);
         HYPRE_StructVectorPrint("struct.out.r", res, 0);
      }

      if (myid == 0 && rep == reps - 1 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         //         hypre_printf("RHS Norm = %20.15e\n", rhs_norm);
         //         hypre_printf("Initial LHS (x0) Norm = %20.15e\n", x0_norm);
         //         hypre_printf("Real Relative Residual Norm  = %20.15e\n", real_res_norm);
         //         hypre_printf("Final Relative Residual Norm = %20.15e\n", final_res_norm);
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
            hypre_StructMatvecCompute(matvec_data, 1.0, A, x, 1.0, b, b);
         }
         /* this counts mult-adds */
         hypre_IncFLOPCount(7 * N * imax);

         hypre_EndTiming(time_index);
         hypre_PrintTiming("Matvec time", comm);
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
      HYPRE_StructVectorDestroy(res);

      for (i = 0; i < stored_stencil_size; i++)
      {
         hypre_TFree(offsets[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(offsets, HYPRE_MEMORY_HOST);
   }

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize();
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

/* Macro for allocating and setting a 1D stencil offset */
#define ALLOC_AND_SET_OFFSET1(i, o1)                            \
   offsets[i] = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_HOST); \
   offsets[i][0] = o1

/* Macro for allocating and setting a 2D stencil offset */
#define ALLOC_AND_SET_OFFSET2(i, o1, o2)                        \
   offsets[i] = hypre_CTAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST); \
   offsets[i][0] = o1;                                          \
   offsets[i][1] = o2

/* Macro for allocating and setting a 3D stencil offset */
#define ALLOC_AND_SET_OFFSET3(i, o1, o2, o3)                    \
   offsets[i] = hypre_CTAlloc(HYPRE_Int, 3, HYPRE_MEMORY_HOST); \
   offsets[i][0] = o1;                                          \
   offsets[i][1] = o2;                                          \
   offsets[i][2] = o3

/*-------------------------------------------------------------------------
 * Set 1D offsets for 3-pt stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_1dim_3pt( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 3, HYPRE_MEMORY_HOST);

   ALLOC_AND_SET_OFFSET1(0, -1);
   ALLOC_AND_SET_OFFSET1(1,  0);
   ALLOC_AND_SET_OFFSET1(2,  1);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set 1D offsets for 3-pt symmetric stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_1dim_3pt_sym( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 2, HYPRE_MEMORY_HOST);

   ALLOC_AND_SET_OFFSET1(0, -1);
   ALLOC_AND_SET_OFFSET1(1,  0);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set 2D offsets for 5-pt stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_2dim_5pt( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 5, HYPRE_MEMORY_HOST);

   ALLOC_AND_SET_OFFSET2(0, -1,  0);
   ALLOC_AND_SET_OFFSET2(1,  0, -1);
   ALLOC_AND_SET_OFFSET2(2,  0,  0);
   ALLOC_AND_SET_OFFSET2(3,  1,  0);
   ALLOC_AND_SET_OFFSET2(4,  0,  1);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set 2D offsets for 5-pt symmetric stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_2dim_5pt_sym( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 3, HYPRE_MEMORY_HOST);

   ALLOC_AND_SET_OFFSET2(0, -1,  0);
   ALLOC_AND_SET_OFFSET2(1,  0, -1);
   ALLOC_AND_SET_OFFSET2(2,  0,  0);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set 3D offsets for 7-pt stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_3dim_7pt( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 7, HYPRE_MEMORY_HOST);

   ALLOC_AND_SET_OFFSET3(0, -1,  0,  0);
   ALLOC_AND_SET_OFFSET3(1,  0, -1,  0);
   ALLOC_AND_SET_OFFSET3(2,  0,  0, -1);
   ALLOC_AND_SET_OFFSET3(3,  0,  0,  0);
   ALLOC_AND_SET_OFFSET3(4,  1,  0,  0);
   ALLOC_AND_SET_OFFSET3(5,  0,  1,  0);
   ALLOC_AND_SET_OFFSET3(6,  0,  0,  1);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set 3D offsets for 7-pt symmetric stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_3dim_7pt_sym( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 4, HYPRE_MEMORY_HOST);

   ALLOC_AND_SET_OFFSET3(0, -1,  0,  0);
   ALLOC_AND_SET_OFFSET3(1,  0, -1,  0);
   ALLOC_AND_SET_OFFSET3(2,  0,  0, -1);
   ALLOC_AND_SET_OFFSET3(3,  0,  0,  0);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set 3D offsets for 27-pt stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_3dim_27pt( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 27, HYPRE_MEMORY_HOST);

   /* k = -1 */
   ALLOC_AND_SET_OFFSET3(0,   0,  0, -1);
   ALLOC_AND_SET_OFFSET3(1,  -1,  0, -1);
   ALLOC_AND_SET_OFFSET3(2,   1,  0, -1);
   ALLOC_AND_SET_OFFSET3(3,  -1, -1, -1);
   ALLOC_AND_SET_OFFSET3(4,   0, -1, -1);
   ALLOC_AND_SET_OFFSET3(5,   1, -1, -1);
   ALLOC_AND_SET_OFFSET3(6,  -1,  1, -1);
   ALLOC_AND_SET_OFFSET3(7,   0,  1, -1);
   ALLOC_AND_SET_OFFSET3(8,   1,  1, -1);

   /* k = 0 */
   ALLOC_AND_SET_OFFSET3(9,   0,  0,  0);
   ALLOC_AND_SET_OFFSET3(10, -1,  0,  0);
   ALLOC_AND_SET_OFFSET3(11,  1,  0,  0);
   ALLOC_AND_SET_OFFSET3(12, -1, -1,  0);
   ALLOC_AND_SET_OFFSET3(13,  0, -1,  0);
   ALLOC_AND_SET_OFFSET3(14,  1, -1,  0);
   ALLOC_AND_SET_OFFSET3(15, -1,  1,  0);
   ALLOC_AND_SET_OFFSET3(16,  0,  1,  0);
   ALLOC_AND_SET_OFFSET3(17,  1,  1,  0);

   /* k = 1 */
   ALLOC_AND_SET_OFFSET3(18,  0,  0,  1);
   ALLOC_AND_SET_OFFSET3(19, -1,  0,  1);
   ALLOC_AND_SET_OFFSET3(20,  1,  0,  1);
   ALLOC_AND_SET_OFFSET3(21, -1, -1,  1);
   ALLOC_AND_SET_OFFSET3(22,  0, -1,  1);
   ALLOC_AND_SET_OFFSET3(23,  1, -1,  1);
   ALLOC_AND_SET_OFFSET3(24, -1,  1,  1);
   ALLOC_AND_SET_OFFSET3(25,  0,  1,  1);
   ALLOC_AND_SET_OFFSET3(26,  1,  1,  1);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set 3D offsets for 27-pt symmetric stencil
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetStencilOffsets_3dim_27pt_sym( HYPRE_Int ***offsets_ptr )
{
   HYPRE_Int **offsets;

   offsets = hypre_CTAlloc(HYPRE_Int*, 14, HYPRE_MEMORY_HOST);

   /* Diagonal point */
   ALLOC_AND_SET_OFFSET3(0,   0,  0,  0);

   /* k = 0 */
   ALLOC_AND_SET_OFFSET3(1,  -1,  0,  0);
   ALLOC_AND_SET_OFFSET3(2,  -1, -1,  0);
   ALLOC_AND_SET_OFFSET3(3,   0, -1,  0);
   ALLOC_AND_SET_OFFSET3(4,   1, -1,  0);

   /* k = -1 */
   ALLOC_AND_SET_OFFSET3(5,   0,  0, -1);
   ALLOC_AND_SET_OFFSET3(6,  -1,  0, -1);
   ALLOC_AND_SET_OFFSET3(7,   1,  0, -1);
   ALLOC_AND_SET_OFFSET3(8,  -1, -1, -1);
   ALLOC_AND_SET_OFFSET3(9,   0, -1, -1);
   ALLOC_AND_SET_OFFSET3(10,  1, -1, -1);
   ALLOC_AND_SET_OFFSET3(11, -1,  1, -1);
   ALLOC_AND_SET_OFFSET3(12,  0,  1, -1);
   ALLOC_AND_SET_OFFSET3(13,  1,  1, -1);

   /* Set output pointer */
   *offsets_ptr = offsets;

   return 0;
}

/*-------------------------------------------------------------------------
 * Set values to a vector. Need to pass the initialized vector, grid,
 * period of grid, method for setting values, and the value.
 *
 * The variable "type" accepts the following parameters:
 *   0: Use random entries in [-1.0, +1.0] with seed given by "value"
 *   1: Use constant entries equal to "value"
 *
 * For periodic b.c. in all directions, rhs need to satisfy
 * compatibility condition. Achieved by setting a source and
 * sink of equal strength.
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetValuesVector( hypre_StructGrid   *grid,
                 hypre_StructVector *zvector,
                 HYPRE_Int          *period,
                 HYPRE_Int           type,
                 HYPRE_Real          value )
{
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation  memory_location = hypre_StructVectorMemoryLocation(zvector);
#endif

   HYPRE_Int             ib;
   hypre_IndexRef        ilower, iupper;
   hypre_Box            *box;
   hypre_BoxArray       *boxes;
   HYPRE_Real           *values;
   HYPRE_Real           *values_h;
   HYPRE_Int             max_volume, volume, ndim;

   ndim  = hypre_StructGridNDim(grid);
   boxes = hypre_StructGridBoxes(grid);
   hypre_SeedRand((HYPRE_Int)value);

   /* Compute max. volume among boxes, so we allocate values only once */
   max_volume = 0;
   hypre_ForBoxI(ib, boxes)
   {
      box = hypre_BoxArrayBox(boxes, ib);
      max_volume = hypre_max(max_volume, hypre_BoxVolume(box));
   }

   /* Allocate value arrays */
   values_h = hypre_CTAlloc(HYPRE_Real, max_volume, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
   values   = hypre_CTAlloc(HYPRE_Real, max_volume, memory_location);
#else
   values   = values_h;
#endif

   hypre_ForBoxI(ib, boxes)
   {
      box    = hypre_BoxArrayBox(boxes, ib);
      volume = hypre_BoxVolume(box);

      if ((ndim == 2 && period[0] != 0 && period[1] != 0) ||
          (ndim == 3 && period[0] != 0 && period[1] != 0 && period[2] != 0))
      {
         values_h[0]          =  value;
         values_h[volume - 1] = -value;
      }
      else
      {
         if (type > 0)
         {
            /* Use value */
            zypre_LoopBegin(volume, i)
            {
               values_h[i] = value;
            }
            zypre_LoopEnd()
         }
         else
         {
            /* Use random numbers */
            zypre_LoopBegin(volume, i)
            {
               values_h[i] = 2.0 * hypre_Rand() - 1.0;
            }
            zypre_LoopEnd()
         }
      }

#if defined(HYPRE_USING_GPU)
      hypre_TMemcpy(values, values_h, HYPRE_Real, volume, memory_location, HYPRE_MEMORY_HOST);
#endif

      ilower = hypre_BoxIMin(box);
      iupper = hypre_BoxIMax(box);
      HYPRE_StructVectorSetBoxValues(zvector, ilower, iupper, values);
   }

   /* Free memory */
   hypre_TFree(values_h, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
   hypre_TFree(values, memory_location);
#endif

   return 0;
}

/*-------------------------------------------------------------------------
 * Adds values to matrix based on a 7 point (3d)
 * symmetric stencil for a convection-diffusion problem.
 * It need an initialized matrix, an assembled grid, and the constants
 * that determine the 7 point (3d) convection-diffusion.
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetValuesMatrix( HYPRE_StructMatrix A,
                 HYPRE_StructGrid   grid,
                 HYPRE_Real         cx,
                 HYPRE_Real         cy,
                 HYPRE_Real         cz,
                 HYPRE_Real         conx,
                 HYPRE_Real         cony,
                 HYPRE_Real         conz )
{
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(A);
#endif

   HYPRE_Int            *stencil_indices;
   HYPRE_Int             stencil_size;
   HYPRE_Int             constant_coefficient;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   HYPRE_Real           *vvalues = NULL, *vvalues_h = NULL;
   HYPRE_Real           *cvalues = NULL, *cvalues_h = NULL;
   HYPRE_Int             i, d, s, bi = 0;
   hypre_IndexRef        ilower, iupper;
   HYPRE_Real            east, west, north, south, top, bottom, center;
   HYPRE_Int             volume, max_volume;
   HYPRE_Int             ndim, sym;

   boxes =  hypre_StructGridBoxes(grid);
   ndim  =  hypre_StructGridNDim(grid);
   sym   =  hypre_StructMatrixSymmetric(A);
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   bi = 0;

   east   = -cx;
   west   = -cx;
   north  = -cy;
   south  = -cy;
   top    = -cz;
   bottom = -cz;
   center = 0.0;
   switch (ndim)
   {
      case 3:
         center += 2.0 * cz;
         HYPRE_FALLTHROUGH;

      case 2:
         center += 2.0 * cy;
         HYPRE_FALLTHROUGH;

      default:
         center += 2.0 * cx;
         break;
   }

   stencil_size = 1 + (2 - sym) * ndim;
   stencil_indices = hypre_CTAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
   for (s = 0; s < stencil_size; s++)
   {
      stencil_indices[s] = s;
   }

   /* Compute max. volume among boxes, so we allocate values only once */
   max_volume = 0;
   hypre_ForBoxI(bi, boxes)
   {
      box = hypre_BoxArrayBox(boxes, bi);
      max_volume = hypre_max(max_volume, hypre_BoxVolume(box));
   }

   /* Allocate value arrays */
   if (constant_coefficient == 0)
   {
      vvalues_h = hypre_CTAlloc(HYPRE_Real, max_volume * stencil_size, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
      vvalues   = hypre_CTAlloc(HYPRE_Real, max_volume * stencil_size, memory_location);
#endif
   }
   else if (constant_coefficient == 1)
   {
      cvalues_h = hypre_CTAlloc(HYPRE_Real, stencil_size, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
      cvalues   = hypre_CTAlloc(HYPRE_Real, stencil_size, memory_location);
#endif
   }
   else if (constant_coefficient == 2)
   {
      cvalues_h = hypre_CTAlloc(HYPRE_Real, stencil_size - 1, HYPRE_MEMORY_HOST);
      vvalues_h = hypre_CTAlloc(HYPRE_Real, max_volume, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
      cvalues   = hypre_CTAlloc(HYPRE_Real, stencil_size - 1, memory_location);
      vvalues   = hypre_CTAlloc(HYPRE_Real, max_volume, memory_location);
#endif
   }

#if !defined(HYPRE_USING_GPU)
   vvalues = vvalues_h;
   cvalues = cvalues_h;
#endif

   if (sym)
   {
      if (constant_coefficient == 0)
      {
         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            if (ndim == 1)
            {
               for (d = 0; d < volume; d++)
               {
                  i = stencil_size * d;

                  vvalues_h[i] = west;
                  vvalues_h[i + 1] = center;
               }
            }
            else if (ndim == 2)
            {
               for (d = 0; d < volume; d++)
               {
                  i = stencil_size * d;

                  vvalues_h[i] = west;
                  vvalues_h[i + 1] = south;
                  vvalues_h[i + 2] = center;
               }
            }
            else if (ndim == 3)
            {
               for (d = 0; d < volume; d++)
               {
                  i = stencil_size * d;

                  vvalues_h[i] = west;
                  vvalues_h[i + 1] = south;
                  vvalues_h[i + 2] = bottom;
                  vvalues_h[i + 3] = center;
               }
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, stencil_size * volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, vvalues);
         }
      }
      else if (constant_coefficient == 1)
      {
         switch (ndim)
         {
            case 1:
               cvalues_h[0] = west;
               cvalues_h[1] = center;
               break;

            case 2:
               cvalues_h[0] = west;
               cvalues_h[1] = south;
               cvalues_h[2] = center;
               break;

            case 3:
               cvalues_h[0] = west;
               cvalues_h[1] = south;
               cvalues_h[2] = bottom;
               cvalues_h[3] = center;
               break;
         }

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size, stencil_indices, cvalues);
         }
      }
      else
      {
         hypre_assert(constant_coefficient == 2);

         /* stencil index for the center equals ndim, so it's easy to leave out */
         switch (ndim)
         {
            case 1:
               cvalues_h[0] = west;
               break;

            case 2:
               cvalues_h[0] = west;
               cvalues_h[1] = south;
               break;

            case 3:
               cvalues_h[0] = west;
               cvalues_h[1] = south;
               cvalues_h[2] = bottom;
               break;
         }

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size - 1,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size - 1, stencil_indices, cvalues);
         }

         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            for (i = 0; i < volume; i++)
            {
               vvalues_h[i] = center;
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, stencil_indices + ndim, vvalues);
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

      if (constant_coefficient == 0)
      {
         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            for (d = 0; d < volume; d++)
            {
               i = stencil_size * d;
               switch (ndim)
               {
                  case 1:
                     vvalues_h[i] = west;
                     vvalues_h[i + 1] = center;
                     vvalues_h[i + 2] = east;
                     break;

                  case 2:
                     vvalues_h[i] = west;
                     vvalues_h[i + 1] = south;
                     vvalues_h[i + 2] = center;
                     vvalues_h[i + 3] = east;
                     vvalues_h[i + 4] = north;
                     break;

                  case 3:
                     vvalues_h[i] = west;
                     vvalues_h[i + 1] = south;
                     vvalues_h[i + 2] = bottom;
                     vvalues_h[i + 3] = center;
                     vvalues_h[i + 4] = east;
                     vvalues_h[i + 5] = north;
                     vvalues_h[i + 6] = top;
                     break;
               }
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, stencil_size * volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, vvalues);
         }
      }
      else if (constant_coefficient == 1)
      {
         switch (ndim)
         {
            case 1:
               cvalues_h[0] = west;
               cvalues_h[1] = center;
               cvalues_h[2] = east;
               break;

            case 2:
               cvalues_h[0] = west;
               cvalues_h[1] = south;
               cvalues_h[2] = center;
               cvalues_h[3] = east;
               cvalues_h[4] = north;
               break;

            case 3:
               cvalues_h[0] = west;
               cvalues_h[1] = south;
               cvalues_h[2] = bottom;
               cvalues_h[3] = center;
               cvalues_h[4] = east;
               cvalues_h[5] = north;
               cvalues_h[6] = top;
               break;
         }

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size, stencil_indices, cvalues);
         }
      }
      else
      {
         hypre_assert(constant_coefficient == 2);

         switch (ndim)
         {
            /* no center in stencil_indices and values */
            case 1:
               stencil_indices[0] = 0;
               stencil_indices[1] = 2;

               cvalues_h[0] = west;
               cvalues_h[1] = east;
               break;

            case 2:
               stencil_indices[0] = 0;
               stencil_indices[1] = 1;
               stencil_indices[2] = 3;
               stencil_indices[3] = 4;

               cvalues_h[0] = west;
               cvalues_h[1] = south;
               cvalues_h[2] = east;
               cvalues_h[3] = north;
               break;

            case 3:
               stencil_indices[0] = 0;
               stencil_indices[1] = 1;
               stencil_indices[2] = 2;
               stencil_indices[3] = 4;
               stencil_indices[4] = 5;
               stencil_indices[5] = 6;

               cvalues_h[0] = west;
               cvalues_h[1] = south;
               cvalues_h[2] = bottom;
               cvalues_h[3] = east;
               cvalues_h[4] = north;
               cvalues_h[5] = top;
               break;
         }

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size - 1,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size - 1, stencil_indices, cvalues);
         }

         /* center is variable */
         stencil_indices[0] = ndim; /* refers to center */
         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            for (i = 0; i < volume; i++)
            {
               vvalues_h[i] = center;
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, stencil_indices, vvalues);
         }
      }
   }

#if defined(HYPRE_USING_GPU)
   hypre_TFree(cvalues, memory_location);
   hypre_TFree(vvalues, memory_location);
#endif
   hypre_TFree(stencil_indices, HYPRE_MEMORY_HOST);
   hypre_TFree(cvalues_h, HYPRE_MEMORY_HOST);
   hypre_TFree(vvalues_h, HYPRE_MEMORY_HOST);

   return 0;
}

/*-------------------------------------------------------------------------
 * Adds values to matrix based on 27pt (3d) stencil
 *
 * Models a diffusion equation with cross-derivative terms:
 *
 *  c_x  d²u/dx²  + c_y  d²u/dy²  + c_z  d²u/dz² +
 *  c_xy d²u/dxdy + c_xz d²u/dxdz + c_yz d²u/dydz = 0
 *
 * The cross coefficients are computed as the geometric average
 * of neighboring directional diffusion coefficients.
 *-------------------------------------------------------------------------*/

HYPRE_Int
SetValuesCrossMatrix( HYPRE_StructMatrix A,
                      HYPRE_StructGrid   grid,
                      HYPRE_Real         cx,
                      HYPRE_Real         cy,
                      HYPRE_Real         cz )
{
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(A);
#endif

   HYPRE_Int            *stencil_indices;
   HYPRE_Int             stencil_size;
   HYPRE_Int             constant_coefficient;
   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   HYPRE_Real           *vvalues = NULL, *vvalues_h = NULL;
   HYPRE_Real           *cvalues = NULL, *cvalues_h = NULL;
   HYPRE_Int             i, d, s, bi = 0;
   hypre_IndexRef        ilower, iupper;
   HYPRE_Real            center, cxy, cxz, cyz;
   HYPRE_Int             volume, max_volume, max_size;
   HYPRE_Int             ndim, sym;

   boxes =  hypre_StructGridBoxes(grid);
   ndim  =  hypre_StructGridNDim(grid);
   sym   =  hypre_StructMatrixSymmetric(A);
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);

   if (ndim != 3)
   {
      hypre_printf("%s valid only in 3D\n!", __func__);
      hypre_MPI_Abort(hypre_MPI_COMM_WORLD, 1);
   }

   bi = 0;

   cxy = hypre_sqrt(cx * cy) / 4.0;
   cxz = hypre_sqrt(cx * cz) / 4.0;
   cyz = hypre_sqrt(cy * cz) / 4.0;
   center = 2.0 * (cx + cy + cz);

   stencil_size = (sym) ? 14 : 27;
   stencil_indices = hypre_TAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
   for (s = 0; s < stencil_size; s++)
   {
      stencil_indices[s] = s;
   }

   /* Compute max. volume among boxes, so we allocate values only once */
   max_volume = 0;
   hypre_ForBoxI(bi, boxes)
   {
      box = hypre_BoxArrayBox(boxes, bi);
      max_volume = hypre_max(max_volume, hypre_BoxVolume(box));
   }
   max_size = max_volume * stencil_size;

   /* Allocate value arrays */
   if (constant_coefficient == 0)
   {
      vvalues_h = hypre_TAlloc(HYPRE_Real, max_size, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
      vvalues   = hypre_TAlloc(HYPRE_Real, max_size, memory_location);
#endif
   }
   else if (constant_coefficient == 1)
   {
      cvalues_h = hypre_TAlloc(HYPRE_Real, stencil_size, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
      cvalues   = hypre_TAlloc(HYPRE_Real, stencil_size, memory_location);
#endif
   }
   else if (constant_coefficient == 2)
   {
      cvalues_h = hypre_TAlloc(HYPRE_Real, stencil_size - 1, HYPRE_MEMORY_HOST);
      vvalues_h = hypre_TAlloc(HYPRE_Real, max_volume, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_GPU)
      cvalues   = hypre_TAlloc(HYPRE_Real, stencil_size - 1, memory_location);
      vvalues   = hypre_TAlloc(HYPRE_Real, max_volume, memory_location);
#endif
   }

#if !defined(HYPRE_USING_GPU)
   vvalues = vvalues_h;
   cvalues = cvalues_h;
#endif

   if (sym)
   {
      if (constant_coefficient == 0)
      {
         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            for (d = 0; d < volume; d++)
            {
               i = stencil_size * d;

               /* Follow the order given by SetStencilOffsets_3dim_27pt_sym */
               vvalues_h[i +  0] = center;  /* ( 0,  0,  0) */
               vvalues_h[i +  1] = -cx;     /* (-1,  0,  0) */
               vvalues_h[i +  2] = -cxy;    /* (-1, -1,  0) */
               vvalues_h[i +  3] = -cy;     /* ( 0, -1,  0) */
               vvalues_h[i +  4] = +cxy;    /* ( 1, -1,  0) */
               vvalues_h[i +  5] = -cz;     /* ( 0,  0, -1) */
               vvalues_h[i +  6] = -cxz;    /* (-1,  0, -1) */
               vvalues_h[i +  7] = +cxz;    /* ( 1,  0, -1) */
               vvalues_h[i +  8] = 0.0;     /* (-1, -1, -1) */
               vvalues_h[i +  9] = -cyz;    /* ( 0, -1, -1) */
               vvalues_h[i + 10] = 0.0;     /* ( 1, -1, -1) */
               vvalues_h[i + 11] = 0.0;     /* (-1,  1, -1) */
               vvalues_h[i + 12] = +cyz;    /* ( 0,  1, -1) */
               vvalues_h[i + 13] = 0.0;     /* ( 1,  1, -1) */
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, stencil_size * volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, vvalues);
         }
      }
      else if (constant_coefficient == 1)
      {
         cvalues_h[0]  = center;  /* ( 0,  0,  0) */
         cvalues_h[1]  = -cx;     /* (-1,  0,  0) */
         cvalues_h[2]  = -cxy;    /* (-1, -1,  0) */
         cvalues_h[3]  = -cy;     /* ( 0, -1,  0) */
         cvalues_h[4]  = +cxy;    /* ( 1, -1,  0) */
         cvalues_h[5]  = -cz;     /* ( 0,  0, -1) */
         cvalues_h[6]  = -cxz;    /* (-1,  0, -1) */
         cvalues_h[7]  = +cxz;    /* ( 1,  0, -1) */
         cvalues_h[8]  = 0.0;     /* (-1, -1, -1) */
         cvalues_h[9]  = -cyz;    /* ( 0, -1, -1) */
         cvalues_h[10] = 0.0;     /* ( 1, -1, -1) */
         cvalues_h[11] = 0.0;     /* (-1,  1, -1) */
         cvalues_h[12] = +cyz;    /* ( 0,  1, -1) */
         cvalues_h[13] = 0.0;     /* ( 1,  1, -1) */

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size, stencil_indices,
                                                cvalues);
         }
      }
      else
      {
         hypre_assert(constant_coefficient == 2);

         cvalues_h[0]  = -cx;     /* (-1,  0,  0) */
         cvalues_h[1]  = -cxy;    /* (-1, -1,  0) */
         cvalues_h[2]  = -cy;     /* ( 0, -1,  0) */
         cvalues_h[3]  = +cxy;    /* ( 1, -1,  0) */
         cvalues_h[4]  = -cz;     /* ( 0,  0, -1) */
         cvalues_h[5]  = -cxz;    /* (-1,  0, -1) */
         cvalues_h[6]  = +cxz;    /* ( 1,  0, -1) */
         cvalues_h[7]  = 0.0;     /* (-1, -1, -1) */
         cvalues_h[8]  = -cyz;    /* ( 0, -1, -1) */
         cvalues_h[9]  = 0.0;     /* ( 1, -1, -1) */
         cvalues_h[10] = 0.0;     /* (-1,  1, -1) */
         cvalues_h[11] = +cyz;    /* ( 0,  1, -1) */
         cvalues_h[12] = 0.0;     /* ( 1,  1, -1) */

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size - 1,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A,
                                                stencil_size - 1,
                                                stencil_indices + 1,
                                                cvalues);
         }

         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            for (i = 0; i < volume; i++)
            {
               vvalues_h[i] = center;
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1,
                                           stencil_indices, vvalues);
         }
      }
   }
   else
   {
      if (constant_coefficient == 0)
      {
         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            for (d = 0; d < volume; d++)
            {
               i = stencil_size * d;

               /* Follow the order given by SetStencilOffsets_3dim_27pt */
               vvalues_h[i +  0] = -cz;     /* ( 0,  0, -1) */
               vvalues_h[i +  1] = -cxz;    /* (-1,  0, -1) */
               vvalues_h[i +  2] = +cxz;    /* ( 1,  0, -1) */
               vvalues_h[i +  3] = 0.0;     /* (-1, -1, -1) */
               vvalues_h[i +  4] = -cyz;    /* ( 0, -1, -1) */
               vvalues_h[i +  5] = 0.0;     /* ( 1, -1, -1) */
               vvalues_h[i +  6] = 0.0;     /* (-1,  1, -1) */
               vvalues_h[i +  7] = +cyz;    /* ( 0,  1, -1) */
               vvalues_h[i +  8] = 0.0;     /* ( 1,  1, -1) */

               vvalues_h[i +  9] = center;  /* ( 0,  0,  0) */
               vvalues_h[i + 10] = -cx;     /* (-1,  0,  0) */
               vvalues_h[i + 11] = -cx;     /* ( 1,  0,  0) */
               vvalues_h[i + 12] = -cxy;    /* (-1, -1,  0) */
               vvalues_h[i + 13] = -cy;     /* ( 0, -1,  0) */
               vvalues_h[i + 14] = +cxy;    /* ( 1, -1,  0) */
               vvalues_h[i + 15] = +cxy;    /* (-1,  1,  0) */
               vvalues_h[i + 16] = -cy;     /* ( 0,  1,  0) */
               vvalues_h[i + 17] = -cxy;    /* ( 1,  1,  0) */

               vvalues_h[i + 18] = -cz;     /* ( 0,  0,  1) */
               vvalues_h[i + 19] = +cxz;    /* (-1,  0,  1) */
               vvalues_h[i + 20] = -cxz;    /* ( 1,  0,  1) */
               vvalues_h[i + 21] = 0.0;     /* (-1, -1,  1) */
               vvalues_h[i + 22] = +cyz;    /* ( 0, -1,  1) */
               vvalues_h[i + 23] = 0.0;     /* ( 1, -1,  1) */
               vvalues_h[i + 24] = 0.0;     /* (-1,  1,  1) */
               vvalues_h[i + 25] = -cyz;    /* ( 0,  1,  1) */
               vvalues_h[i + 26] = 0.0;     /* ( 1,  1,  1) */
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, stencil_size * volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, stencil_size,
                                           stencil_indices, vvalues);
         }
      }
      else if (constant_coefficient == 1)
      {
         cvalues_h[0]  = -cz;     /* ( 0,  0, -1) */
         cvalues_h[1]  = -cxz;    /* (-1,  0, -1) */
         cvalues_h[2]  = +cxz;    /* ( 1,  0, -1) */
         cvalues_h[3]  = 0.0;     /* (-1, -1, -1) */
         cvalues_h[4]  = -cyz;    /* ( 0, -1, -1) */
         cvalues_h[5]  = 0.0;     /* ( 1, -1, -1) */
         cvalues_h[6]  = 0.0;     /* (-1,  1, -1) */
         cvalues_h[7]  = +cyz;    /* ( 0,  1, -1) */
         cvalues_h[8]  = 0.0;     /* ( 1,  1, -1) */
         cvalues_h[9]  = center;  /* ( 0,  0,  0) */
         cvalues_h[10] = -cx;     /* (-1,  0,  0) */
         cvalues_h[11] = -cx;     /* ( 1,  0,  0) */
         cvalues_h[12] = -cxy;    /* (-1, -1,  0) */
         cvalues_h[13] = -cy;     /* ( 0, -1,  0) */
         cvalues_h[14] = +cxy;    /* ( 1, -1,  0) */
         cvalues_h[15] = +cxy;    /* (-1,  1,  0) */
         cvalues_h[16] = -cy;     /* ( 0,  1,  0) */
         cvalues_h[17] = -cxy;    /* ( 1,  1,  0) */
         cvalues_h[18] = -cz;     /* ( 0,  0,  1) */
         cvalues_h[19] = +cxz;    /* (-1,  0,  1) */
         cvalues_h[20] = -cxz;    /* ( 1,  0,  1) */
         cvalues_h[21] = 0.0;     /* (-1, -1,  1) */
         cvalues_h[22] = +cyz;    /* ( 0, -1,  1) */
         cvalues_h[23] = 0.0;     /* ( 1, -1,  1) */
         cvalues_h[24] = 0.0;     /* (-1,  1,  1) */
         cvalues_h[25] = -cyz;    /* ( 0,  1,  1) */
         cvalues_h[26] = 0.0;     /* ( 1,  1,  1) */

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size, stencil_indices,
                                                cvalues);
         }
      }
      else
      {
         hypre_assert(constant_coefficient == 2);

         for (s = 0; s < 9; s++) { stencil_indices[s] = s; }
         for (s = 0; s < 8; s++) { stencil_indices[s + 9] = s + 10; }
         for (s = 0; s < 9; s++) { stencil_indices[s + 17] = s + 18; }

         cvalues_h[0]  = -cz;     /* ( 0,  0, -1) */
         cvalues_h[1]  = -cxz;    /* (-1,  0, -1) */
         cvalues_h[2]  = +cxz;    /* ( 1,  0, -1) */
         cvalues_h[3]  = 0.0;     /* (-1, -1, -1) */
         cvalues_h[4]  = -cyz;    /* ( 0, -1, -1) */
         cvalues_h[5]  = 0.0;     /* ( 1, -1, -1) */
         cvalues_h[6]  = 0.0;     /* (-1,  1, -1) */
         cvalues_h[7]  = +cyz;    /* ( 0,  1, -1) */
         cvalues_h[8]  = 0.0;     /* ( 1,  1, -1) */
         cvalues_h[9]  = -cx;     /* (-1,  0,  0) */
         cvalues_h[10] = -cx;     /* ( 1,  0,  0) */
         cvalues_h[11] = -cxy;    /* (-1, -1,  0) */
         cvalues_h[12] = -cy;     /* ( 0, -1,  0) */
         cvalues_h[13] = +cxy;    /* ( 1, -1,  0) */
         cvalues_h[14] = +cxy;    /* (-1,  1,  0) */
         cvalues_h[15] = -cy;     /* ( 0,  1,  0) */
         cvalues_h[16] = -cxy;    /* ( 1,  1,  0) */
         cvalues_h[17] = -cz;     /* ( 0,  0,  1) */
         cvalues_h[18] = +cxz;    /* (-1,  0,  1) */
         cvalues_h[19] = -cxz;    /* ( 1,  0,  1) */
         cvalues_h[20] = 0.0;     /* (-1, -1,  1) */
         cvalues_h[21] = +cyz;    /* ( 0, -1,  1) */
         cvalues_h[22] = 0.0;     /* ( 1, -1,  1) */
         cvalues_h[23] = 0.0;     /* (-1,  1,  1) */
         cvalues_h[24] = -cyz;    /* ( 0,  1,  1) */
         cvalues_h[25] = 0.0;     /* ( 1,  1,  1) */

#if defined(HYPRE_USING_GPU)
         hypre_TMemcpy(cvalues, cvalues_h, HYPRE_Real, stencil_size - 1,
                       memory_location, HYPRE_MEMORY_HOST);
#endif

         if (hypre_BoxArraySize(boxes) > 0)
         {
            HYPRE_StructMatrixSetConstantValues(A, stencil_size - 1, stencil_indices,
                                                cvalues);
         }

         /* center is variable */
         stencil_indices[0] = 9; /* refers to center */
         hypre_ForBoxI(bi, boxes)
         {
            box    = hypre_BoxArrayBox(boxes, bi);
            volume = hypre_BoxVolume(box);

            for (i = 0; i < volume; i++)
            {
               vvalues_h[i] = center;
            }

#if defined(HYPRE_USING_GPU)
            hypre_TMemcpy(vvalues, vvalues_h, HYPRE_Real, volume,
                          memory_location, HYPRE_MEMORY_HOST);
#endif

            ilower = hypre_BoxIMin(box);
            iupper = hypre_BoxIMax(box);
            HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, stencil_indices,
                                           vvalues);
         }
      }
   }

#if defined(HYPRE_USING_GPU)
   hypre_TFree(cvalues, memory_location);
   hypre_TFree(vvalues, memory_location);
#endif
   hypre_TFree(stencil_indices, HYPRE_MEMORY_HOST);
   hypre_TFree(cvalues_h, HYPRE_MEMORY_HOST);
   hypre_TFree(vvalues_h, HYPRE_MEMORY_HOST);

   return 0;
}

/*********************************************************************************
 * this function sets to zero the stencil entries that are on the boundary
 * Grid, matrix and the period are needed.
 *********************************************************************************/

HYPRE_Int
SetStencilBndry( HYPRE_StructMatrix  A,
                 HYPRE_StructStencil stencil,
                 HYPRE_StructGrid    grid,
                 HYPRE_Int          *period )
{
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(A);
   hypre_BoxArray       *boxes;
   HYPRE_Int             size, i, j, d, ib;
   HYPRE_Int           **ilower;
   HYPRE_Int           **iupper;
   HYPRE_Int            *vol;
   HYPRE_Int            *istart, *iend;
   hypre_Box            *box;
   hypre_Box            *bbox;
   HYPRE_Real           *values;
   HYPRE_Int             ndim, sym;
   HYPRE_Int             constant_coefficient;
   HYPRE_Int             stencil_size = hypre_StructStencilSize(stencil);
   HYPRE_Int             stencil_indices[1] = {0};
   HYPRE_Int             stencil_sizes_27pt_neg[3] = {9, 9, 9};
   HYPRE_Int             stencil_indices_27pt_neg[3][9] =
   {
      {1, 3, 6, 10, 12, 15, 19, 21, 24},
      {3, 4, 5, 12, 13, 14, 21, 22, 23},
      {0, 1, 2, 3, 4, 5, 6, 7, 8}
   };
   HYPRE_Int             stencil_sizes_27pt_pos[3] = {9, 9, 9};
   HYPRE_Int             stencil_indices_27pt_pos[3][9] =
   {
      {2, 5, 8, 11, 14, 17, 20, 23, 26},
      {6, 7, 8, 15, 16, 17, 24, 25, 26},
      {18, 19, 20, 21, 22, 23, 24, 25, 26}
   };
   HYPRE_Int             stencil_sizes_27pt_sym_neg[3] = {5, 6, 9};
   HYPRE_Int             stencil_indices_27pt_sym_neg[3][9] =
   {
      {1, 2, 6, 8, 11, -1, -1, -1, -1},
      {2, 3, 4, 8, 9, 10, -1, -1, -1},
      {5, 6, 7, 8, 9, 10, 11, 12, 13}
   };
   HYPRE_Int             stencil_sizes_27pt_sym_pos[3] = {4, 3, 0};
   HYPRE_Int             stencil_indices_27pt_sym_pos[3][4] =
   {
      {4, 7, 10, 13},
      {11, 12, 13, -1},
      {-1, -1, -1, -1}
   };

   boxes  = hypre_StructGridBoxes(grid);
   bbox   = hypre_StructGridBoundingBox(grid);
   istart = hypre_BoxIMin(bbox);
   iend   = hypre_BoxIMax(bbox);
   size   = hypre_StructGridNumBoxes(grid);
   ndim   = hypre_StructGridNDim(grid);
   sym    =  hypre_StructMatrixSymmetric(A);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient > 0 ) { return 1; }
   /*...no space dependence if constant_coefficient==1,
     and space dependence only for diagonal if constant_coefficient==2 --
     and this function only touches off-diagonal entries */

   vol    = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
   ilower = hypre_CTAlloc(HYPRE_Int*, size, HYPRE_MEMORY_HOST);
   iupper = hypre_CTAlloc(HYPRE_Int*, size, HYPRE_MEMORY_HOST);
   for (i = 0; i < size; i++)
   {
      ilower[i] = hypre_CTAlloc(HYPRE_Int, ndim, HYPRE_MEMORY_HOST);
      iupper[i] = hypre_CTAlloc(HYPRE_Int, ndim, HYPRE_MEMORY_HOST);
   }

   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      for (d = 0; d < ndim; d++)
      {
         ilower[i][d] = hypre_BoxIMinD(box, d);
         iupper[i][d] = hypre_BoxIMaxD(box, d);
      }
      vol[i] = hypre_BoxVolume(box);
   }

   if (constant_coefficient == 0)
   {
      for (d = 0; d < ndim; d++)
      {
         for (ib = 0; ib < size; ib++)
         {
            values = hypre_CTAlloc(HYPRE_Real, vol[ib], memory_location);

            if (ilower[ib][d] == istart[d] && period[d] == 0)
            {
               j = iupper[ib][d];
               iupper[ib][d] = istart[d];
               if (stencil_size != 14 && stencil_size != 27)
               {
                  stencil_indices[0] = d;
                  HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                                 1, stencil_indices, values);
               }
               else if (stencil_size == 14 && ndim == 3 && sym)
               {
                  HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                                 stencil_sizes_27pt_sym_neg[d],
                                                 stencil_indices_27pt_sym_neg[d],
                                                 values);
               }
               else if (stencil_size == 27 && ndim == 3 && !sym)
               {
                  HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                                 stencil_sizes_27pt_neg[d],
                                                 stencil_indices_27pt_neg[d],
                                                 values);
               }
               iupper[ib][d] = j;
            }

            if (iupper[ib][d] == iend[d] && period[d] == 0)
            {
               j = ilower[ib][d];
               ilower[ib][d] = iend[d];
               if (stencil_size != 14 && stencil_size != 27)
               {
                  stencil_indices[0] = ndim + 1 + d;
                  HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                                 1, stencil_indices, values);
               }
               else if (stencil_size == 14 && ndim == 3 && sym)
               {
                  HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                                 stencil_sizes_27pt_sym_pos[d],
                                                 stencil_indices_27pt_sym_pos[d],
                                                 values);
               }
               else if (stencil_size == 27 && ndim == 3 && !sym)
               {
                  HYPRE_StructMatrixSetBoxValues(A, ilower[ib], iupper[ib],
                                                 stencil_sizes_27pt_pos[d],
                                                 stencil_indices_27pt_pos[d],
                                                 values);
               }
               ilower[ib][d] = j;
            }

            hypre_TFree(values, memory_location);
         }
      }
   }

   hypre_TFree(vol, HYPRE_MEMORY_HOST);
   for (ib = 0 ; ib < size ; ib++)
   {
      hypre_TFree(ilower[ib], HYPRE_MEMORY_HOST);
      hypre_TFree(iupper[ib], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(ilower, HYPRE_MEMORY_HOST);
   hypre_TFree(iupper, HYPRE_MEMORY_HOST);

   return 0;
}
