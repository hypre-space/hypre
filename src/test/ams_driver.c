/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   This test driver performs the following operations:

1. Read a linear system corresponding to a parallel finite element
   discretization of Maxwell's equations.

2. Call the AMS solver in HYPRE to solve that linear system.
*/

/* hypre/AMS prototypes */
#include "_hypre_parcsr_ls.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE.h"
#if defined(HYPRE_USING_CUSPARSE)
#include <cusparse.h>
#endif

void CheckIfFileExists(char *file)
{
   FILE *test;
   if (!(test = fopen(file, "r")))
   {
      hypre_MPI_Finalize();
      hypre_printf("Can't find the input file \"%s\"\n", file);
      exit(1);
   }
   fclose(test);
}

void AMSDriverMatrixRead(const char *file, HYPRE_ParCSRMatrix *A)
{
   FILE *test;
   char file0[100];
   sprintf(file0, "%s.D.0", file);
   if (!(test = fopen(file0, "r")))
   {
      sprintf(file0, "%s.00000", file);
      if (!(test = fopen(file0, "r")))
      {
         hypre_MPI_Finalize();
         hypre_printf("Can't find the input file \"%s\"\n", file);
         exit(1);
      }
      else /* Read in IJ format*/
      {
         HYPRE_IJMatrix ij_A;
         void *object;
         HYPRE_IJMatrixRead(file, hypre_MPI_COMM_WORLD, HYPRE_PARCSR, &ij_A);
         HYPRE_IJMatrixGetObject(ij_A, &object);
         *A = (HYPRE_ParCSRMatrix) object;
         hypre_IJMatrixObject((hypre_IJMatrix *)ij_A) = NULL;
         HYPRE_IJMatrixDestroy(ij_A);
      }
   }
   else /* Read in ParCSR format*/
   {
      HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, file, A);
   }
   fclose(test);
}

void AMSDriverVectorRead(const char *file, HYPRE_ParVector *x)
{
   FILE *test;
   char file0[100];
   sprintf(file0, "%s.0", file);
   if (!(test = fopen(file0, "r")))
   {
      sprintf(file0, "%s.00000", file);
      if (!(test = fopen(file0, "r")))
      {
         hypre_MPI_Finalize();
         hypre_printf("Can't find the input file \"%s\"\n", file);
         exit(1);
      }
      else /* Read in IJ format*/
      {
         HYPRE_IJVector ij_x;
         void *object;
         HYPRE_IJVectorRead(file, hypre_MPI_COMM_WORLD, HYPRE_PARCSR, &ij_x);
         HYPRE_IJVectorGetObject(ij_x, &object);
         *x = (HYPRE_ParVector) object;
         hypre_IJVectorObject((hypre_IJVector *)ij_x) = NULL;
         HYPRE_IJVectorDestroy(ij_x);
      }
   }
   else /* Read in ParCSR format*/
   {
      HYPRE_ParVectorRead(hypre_MPI_COMM_WORLD, file, x);
   }
   fclose(test);
}

hypre_int
main (hypre_int argc,
      char *argv[])
{
   HYPRE_Int num_procs, myid;
   HYPRE_Int time_index;

   HYPRE_Int solver_id;
   HYPRE_Int maxit, pcg_maxit, cycle_type, rlx_type, coarse_rlx_type, rlx_sweeps, dim;
   HYPRE_Real rlx_weight, rlx_omega;
   HYPRE_Int amg_coarsen_type, amg_rlx_type, amg_agg_levels, amg_interp_type, amg_Pmax;
   HYPRE_Int h1_method, singular_problem, coordinates;
   HYPRE_Real tol, theta;
   HYPRE_Real rtol;
   HYPRE_Int rr;
   HYPRE_Int zero_cond;
   HYPRE_Int blockSize;
   HYPRE_Solver solver, precond;

   HYPRE_ParCSRMatrix A = 0, G = 0, Aalpha = 0, Abeta = 0, M = 0;
   HYPRE_ParVector x0 = 0, b = 0;
   HYPRE_ParVector Gx = 0, Gy = 0, Gz = 0;
   HYPRE_ParVector x = 0, y = 0, z = 0;

   HYPRE_ParVector interior_nodes = 0;

   /* default execution policy and memory space */
#if defined(HYPRE_TEST_USING_HOST)
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
#else
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;
   HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
#endif

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   hypre_bind_device_id(-1, myid, num_procs, hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   HYPRE_Initialize();
   HYPRE_DeviceInitialize();

   /* default memory location */
   HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(HYPRE_USING_GPU)
#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= 11000
   /* CUSPARSE_SPMV_ALG_DEFAULT doesn't provide deterministic results */
   HYPRE_SetSpMVUseVendor(0);
#endif
   /* use vendor implementation for SpGEMM */
   HYPRE_SetSpGemmUseVendor(0);
   /* use cuRand for PMIS */
   HYPRE_SetUseGpuRand(1);
#endif

   /* Set defaults */
   solver_id = 3;
   maxit = 200;
   pcg_maxit = 50;
   tol = 1e-6;
   dim = 3;
   coordinates = 0;
   h1_method = 0;
   singular_problem = 0;
   rlx_sweeps = 1;
   rlx_weight = 1.0; rlx_omega = 1.0;
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      cycle_type = 1; amg_coarsen_type = 8; amg_agg_levels = 1; amg_rlx_type = 18;
      coarse_rlx_type = 18, rlx_type = 1; /* PMIS */
   }
   else
   {
      cycle_type = 1; amg_coarsen_type = 10; amg_agg_levels = 1; amg_rlx_type = 8;
      coarse_rlx_type = 8, rlx_type = 2; /* HMIS-1 */
   }

   /* cycle_type = 1; amg_coarsen_type = 10; amg_agg_levels = 0; amg_rlx_type = 3; */ /* HMIS-0 */
   /* cycle_type = 1; amg_coarsen_type = 8; amg_agg_levels = 1; amg_rlx_type = 3;  */ /* PMIS-1 */
   /* cycle_type = 1; amg_coarsen_type = 8; amg_agg_levels = 0; amg_rlx_type = 3;  */ /* PMIS-0 */
   /* cycle_type = 7; amg_coarsen_type = 6; amg_agg_levels = 0; amg_rlx_type = 6;  */ /* Falgout-0 */
   amg_interp_type = 6; amg_Pmax = 4;     /* long-range interpolation */
   /* amg_interp_type = 0; amg_Pmax = 0; */  /* standard interpolation */
   theta = 0.25;
   blockSize = 5;
   rtol = 0;
   rr = 0;
   zero_cond = 0;

   /* Parse command line */
   {
      HYPRE_Int arg_index = 0;
      HYPRE_Int print_usage = 0;

      while (arg_index < argc)
      {
         if ( strcmp(argv[arg_index], "-solver") == 0 )
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-maxit") == 0 )
         {
            arg_index++;
            maxit = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-pcg_maxit") == 0 )
         {
            arg_index++;
            pcg_maxit = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-tol") == 0 )
         {
            arg_index++;
            tol = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-type") == 0 )
         {
            arg_index++;
            cycle_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlx") == 0 )
         {
            arg_index++;
            rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxn") == 0 )
         {
            arg_index++;
            rlx_sweeps = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxw") == 0 )
         {
            arg_index++;
            rlx_weight = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rlxo") == 0 )
         {
            arg_index++;
            rlx_omega = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-ctype") == 0 )
         {
            arg_index++;
            amg_coarsen_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-amgrlx") == 0 )
         {
            arg_index++;
            amg_rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-crlx") == 0 )
         {
            arg_index++;
            coarse_rlx_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-agg") == 0 )
         {
            arg_index++;
            amg_agg_levels = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-itype") == 0 )
         {
            arg_index++;
            amg_interp_type = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-pmax") == 0 )
         {
            arg_index++;
            amg_Pmax = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-dim") == 0 )
         {
            arg_index++;
            dim = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-coord") == 0 )
         {
            arg_index++;
            coordinates = 1;
         }
         else if ( strcmp(argv[arg_index], "-h1") == 0 )
         {
            arg_index++;
            h1_method = 1;
         }
         else if ( strcmp(argv[arg_index], "-sing") == 0 )
         {
            arg_index++;
            singular_problem = 1;
         }
         else if ( strcmp(argv[arg_index], "-theta") == 0 )
         {
            arg_index++;
            theta = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-bsize") == 0 )
         {
            arg_index++;
            blockSize = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rtol") == 0 )
         {
            arg_index++;
            rtol = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-rr") == 0 )
         {
            arg_index++;
            rr = atoi(argv[arg_index++]);
         }
         else if ( strcmp(argv[arg_index], "-zc") == 0 )
         {
            arg_index++;
            zero_cond = 1;
         }
         else if ( strcmp(argv[arg_index], "-help") == 0 )
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if (argc == 1)
      {
         print_usage = 1;
      }

      if ((print_usage) && (myid == 0))
      {
         hypre_printf("                                                                 \n");
         hypre_printf("Usage: mpirun -np <np> %s [<options>]                            \n", argv[0]);
         hypre_printf("                                                                 \n");
         hypre_printf("  Hypre solvers options:                                         \n");
         hypre_printf("    -solver <ID>         : solver ID                             \n");
         hypre_printf("                           0  - AMG                              \n");
         hypre_printf("                           1  - AMG-PCG                          \n");
         hypre_printf("                           2  - AMS                              \n");
         hypre_printf("                           3  - AMS-PCG (default)                \n");
         hypre_printf("                           4  - DS-PCG                           \n");
         hypre_printf("                           5  - AME eigensolver                  \n");
         hypre_printf("    -maxit <num>         : maximum number of iterations (200)    \n");
         hypre_printf("    -pcg_maxit <num>     : maximum number of PCG iterations (50) \n");
         hypre_printf("    -tol <num>           : convergence tolerance (1e-6)          \n");
         hypre_printf("                                                                 \n");
         hypre_printf("  AMS solver options:                                            \n");
         hypre_printf("    -dim <num>           : space dimension                       \n");
         hypre_printf("    -type <num>          : 3-level cycle type (0-8, 11-14)       \n");
         hypre_printf("    -theta <num>         : BoomerAMG threshold (0.25)            \n");
         hypre_printf("    -ctype <num>         : BoomerAMG coarsening type             \n");
         hypre_printf("    -agg <num>           : Levels of BoomerAMG agg. coarsening   \n");
         hypre_printf("    -amgrlx <num>        : BoomerAMG relaxation type             \n");
         hypre_printf("    -itype <num>         : BoomerAMG interpolation type          \n");
         hypre_printf("    -pmax <num>          : BoomerAMG interpolation truncation    \n");
         hypre_printf("    -rlx <num>           : relaxation type                       \n");
         hypre_printf("    -rlxn <num>          : number of relaxation sweeps           \n");
         hypre_printf("    -rlxw <num>          : damping parameter (usually <=1)       \n");
         hypre_printf("    -rlxo <num>          : SOR parameter (usuallyin (0,2))       \n");
         hypre_printf("    -coord               : use coordinate vectors                \n");
         hypre_printf("    -h1                  : use block-diag Poisson solves         \n");
         hypre_printf("    -sing                : curl-curl only (singular) problem     \n");
         hypre_printf("                                                                 \n");
         hypre_printf("  AME eigensolver options:                                       \n");
         hypre_printf("    -bsize<num>          : number of eigenvalues to compute      \n");
         hypre_printf("                                                                 \n");
      }

      if (print_usage)
      {
         hypre_MPI_Finalize();
         return (0);
      }
   }

   /* RL: XXX force to use l1-jac for GPU
    * TODO: change it back when GPU SpTrSV is fixed */
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      amg_rlx_type = 18;
   }

   AMSDriverMatrixRead("mfem.A", &A);
   AMSDriverVectorRead("mfem.x0", &x0);
   AMSDriverVectorRead("mfem.b", &b);
   AMSDriverMatrixRead("mfem.G", &G);

   /* Vectors Gx, Gy and Gz */
   if (!coordinates)
   {
      AMSDriverVectorRead("mfem.Gx", &Gx);
      AMSDriverVectorRead("mfem.Gy", &Gy);
      if (dim == 3)
      {
         AMSDriverVectorRead("mfem.Gz", &Gz);
      }
   }

   /* Vectors x, y and z */
   if (coordinates)
   {
      AMSDriverVectorRead("mfem.x", &x);
      AMSDriverVectorRead("mfem.y", &y);
      if (dim == 3)
      {
         AMSDriverVectorRead("mfem.z", &z);
      }
   }

   /* Poisson matrices */
   if (h1_method)
   {
      AMSDriverMatrixRead("mfem.Aalpha", &Aalpha);
      AMSDriverMatrixRead("mfem.Abeta", &Abeta);
   }

   if (zero_cond)
   {
      AMSDriverVectorRead("mfem.inodes", &interior_nodes);
   }

   if (!myid)
   {
      hypre_printf("Problem size: %d\n\n",
                   hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix*)A));
   }

   hypre_ParCSRMatrixMigrate(A,      hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParCSRMatrixMigrate(G,      hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParCSRMatrixMigrate(Aalpha, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParCSRMatrixMigrate(Abeta,  hypre_HandleMemoryLocation(hypre_handle()));

   hypre_ParVectorMigrate(x0, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(b,  hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(Gx, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(Gy, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(Gz, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(x,  hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(y,  hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(z,  hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(interior_nodes, hypre_HandleMemoryLocation(hypre_handle()));

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   /* AMG */
   if (solver_id == 0)
   {
      HYPRE_Int num_iterations;
      HYPRE_Real final_res_norm;

      /* Start timing */
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
      HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
      HYPRE_BoomerAMGSetRelaxType(solver, rlx_type); /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
      HYPRE_BoomerAMGSetTol(solver, tol);       /* conv. tolerance */
      HYPRE_BoomerAMGSetMaxIter(solver, maxit); /* maximum number of iterations */
      HYPRE_BoomerAMGSetStrongThreshold(solver, theta);

      HYPRE_BoomerAMGSetup(solver, A, b, x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_BoomerAMGSolve(solver, A, b, x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver */
      HYPRE_BoomerAMGDestroy(solver);
   }

   /* AMS */
   if (solver_id == 2)
   {
      /* Start timing */
      time_index = hypre_InitializeTiming("AMS Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_AMSCreate(&solver);

      /* Set AMS parameters */
      HYPRE_AMSSetDimension(solver, dim);
      HYPRE_AMSSetMaxIter(solver, maxit);
      HYPRE_AMSSetTol(solver, tol);
      HYPRE_AMSSetCycleType(solver, cycle_type);
      HYPRE_AMSSetPrintLevel(solver, 1);
      HYPRE_AMSSetDiscreteGradient(solver, G);

      /* Vectors Gx, Gy and Gz */
      if (!coordinates)
      {
         HYPRE_AMSSetEdgeConstantVectors(solver, Gx, Gy, Gz);
      }

      /* Vectors x, y and z */
      if (coordinates)
      {
         HYPRE_AMSSetCoordinateVectors(solver, x, y, z);
      }

      /* Poisson matrices */
      if (h1_method)
      {
         HYPRE_AMSSetAlphaPoissonMatrix(solver, Aalpha);
         HYPRE_AMSSetBetaPoissonMatrix(solver, Abeta);
      }

      if (singular_problem)
      {
         HYPRE_AMSSetBetaPoissonMatrix(solver, NULL);
      }

      /* Smoothing and AMG options */
      HYPRE_AMSSetSmoothingOptions(solver, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
      HYPRE_AMSSetAlphaAMGOptions(solver, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                  amg_interp_type, amg_Pmax);
      HYPRE_AMSSetBetaAMGOptions(solver, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                 amg_interp_type, amg_Pmax);
      HYPRE_AMSSetAlphaAMGCoarseRelaxType(solver, coarse_rlx_type);
      HYPRE_AMSSetBetaAMGCoarseRelaxType(solver, coarse_rlx_type);

      HYPRE_AMSSetup(solver, A, b, x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      time_index = hypre_InitializeTiming("AMS Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_AMSSolve(solver, A, b, x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Destroy solver */
      HYPRE_AMSDestroy(solver);
   }

   /* PCG solvers */
   else if (solver_id == 1 || solver_id == 3 || solver_id == 4)
   {
      HYPRE_Int num_iterations;
      HYPRE_Real final_res_norm;

      /* Start timing */
      if (solver_id == 1)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-PCG Setup");
      }
      else if (solver_id == 3)
      {
         time_index = hypre_InitializeTiming("AMS-PCG Setup");
      }
      else if (solver_id == 4)
      {
         time_index = hypre_InitializeTiming("DS-PCG Setup");
      }
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, maxit); /* max iterations */
      HYPRE_PCGSetTol(solver, tol); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 0); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      /* PCG with AMG preconditioner */
      if (solver_id == 1)
      {
         /* Now set up the AMG preconditioner and specify any parameters */
         HYPRE_BoomerAMGCreate(&precond);
         HYPRE_BoomerAMGSetPrintLevel(precond, 1);  /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType(precond, 6); /* Falgout coarsening */
         HYPRE_BoomerAMGSetRelaxType(precond, rlx_type);   /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps(precond, 1);   /* Sweeeps on each level */
         HYPRE_BoomerAMGSetMaxLevels(precond, 20);  /* maximum number of levels */
         HYPRE_BoomerAMGSetTol(precond, 0.0);      /* conv. tolerance (if needed) */
         HYPRE_BoomerAMGSetMaxIter(precond, 1);     /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold(precond, theta);

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             precond);
      }
      /* PCG with AMS preconditioner */
      if (solver_id == 3)
      {
         /* Now set up the AMS preconditioner and specify any parameters */
         HYPRE_AMSCreate(&precond);
         HYPRE_AMSSetDimension(precond, dim);
         HYPRE_AMSSetMaxIter(precond, 1);
         HYPRE_AMSSetTol(precond, 0.0);
         HYPRE_AMSSetCycleType(precond, cycle_type);
         HYPRE_AMSSetPrintLevel(precond, 0);
         HYPRE_AMSSetDiscreteGradient(precond, G);

         if (zero_cond)
         {
            HYPRE_AMSSetInteriorNodes(precond, interior_nodes);
            HYPRE_AMSSetProjectionFrequency(precond, 5);
         }
         HYPRE_PCGSetResidualTol(solver, rtol);
         HYPRE_PCGSetRecomputeResidualP(solver, rr);

         /* Vectors Gx, Gy and Gz */
         if (!coordinates)
         {
            HYPRE_AMSSetEdgeConstantVectors(precond, Gx, Gy, Gz);
         }

         /* Vectors x, y and z */
         if (coordinates)
         {
            HYPRE_AMSSetCoordinateVectors(precond, x, y, z);
         }

         /* Poisson matrices */
         if (h1_method)
         {
            HYPRE_AMSSetAlphaPoissonMatrix(precond, Aalpha);
            HYPRE_AMSSetBetaPoissonMatrix(precond, Abeta);
         }

         if (singular_problem)
         {
            HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
         }

         /* Smoothing and AMG options */
         HYPRE_AMSSetSmoothingOptions(precond, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
         HYPRE_AMSSetAlphaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                     amg_interp_type, amg_Pmax);
         HYPRE_AMSSetBetaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                    amg_interp_type, amg_Pmax);
         HYPRE_AMSSetAlphaAMGCoarseRelaxType(precond, coarse_rlx_type);
         HYPRE_AMSSetBetaAMGCoarseRelaxType(precond, coarse_rlx_type);

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_AMSSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_AMSSetup,
                             precond);
      }
      /* PCG with diagonal scaling preconditioner */
      else if (solver_id == 4)
      {
         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             NULL);
      }

      /* Setup */
      HYPRE_ParCSRPCGSetup(solver, A, b, x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      if (solver_id == 1)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-PCG Solve");
      }
      else if (solver_id == 3)
      {
         time_index = hypre_InitializeTiming("AMS-PCG Solve");
      }
      else if (solver_id == 4)
      {
         time_index = hypre_InitializeTiming("DS-PCG Solve");
      }
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_ParCSRPCGSolve(solver, A, b, x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
      if (solver_id == 1)
      {
         HYPRE_BoomerAMGDestroy(precond);
      }
      else if (solver_id == 3)
      {
         HYPRE_AMSDestroy(precond);
      }
   }

   if (solver_id == 5)
   {
      AMSDriverMatrixRead("mfem.M", &M);

      hypre_ParCSRMatrixMigrate(M, hypre_HandleMemoryLocation(hypre_handle()));

      time_index = hypre_InitializeTiming("AME Setup");
      hypre_BeginTiming(time_index);

      /* Create AMS preconditioner and specify any parameters */
      HYPRE_AMSCreate(&precond);
      HYPRE_AMSSetDimension(precond, dim);
      HYPRE_AMSSetMaxIter(precond, 1);
      HYPRE_AMSSetTol(precond, 0.0);
      HYPRE_AMSSetCycleType(precond, cycle_type);
      HYPRE_AMSSetPrintLevel(precond, 0);
      HYPRE_AMSSetDiscreteGradient(precond, G);

      /* Vectors Gx, Gy and Gz */
      if (!coordinates)
      {
         HYPRE_AMSSetEdgeConstantVectors(precond, Gx, Gy, Gz);
      }

      /* Vectors x, y and z */
      if (coordinates)
      {
         HYPRE_AMSSetCoordinateVectors(precond, x, y, z);
      }

      /* Poisson matrices */
      if (h1_method)
      {
         HYPRE_AMSSetAlphaPoissonMatrix(precond, Aalpha);
         HYPRE_AMSSetBetaPoissonMatrix(precond, Abeta);
      }

      if (singular_problem)
      {
         HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
      }

      /* Smoothing and AMG options */
      HYPRE_AMSSetSmoothingOptions(precond, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
      HYPRE_AMSSetAlphaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                  amg_interp_type, amg_Pmax);
      HYPRE_AMSSetBetaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                 amg_interp_type, amg_Pmax);
      HYPRE_AMSSetAlphaAMGCoarseRelaxType(precond, coarse_rlx_type);
      HYPRE_AMSSetBetaAMGCoarseRelaxType(precond, coarse_rlx_type);

      /* Set up the AMS preconditioner */
      HYPRE_AMSSetup(precond, A, b, x0);

      /* Create AME object */
      HYPRE_AMECreate(&solver);

      /* Set main parameters */
      HYPRE_AMESetAMSSolver(solver, precond);
      HYPRE_AMESetMassMatrix(solver, M);
      HYPRE_AMESetBlockSize(solver, blockSize);

      /* Set additional parameters */
      HYPRE_AMESetMaxIter(solver, maxit); /* max iterations */
      HYPRE_AMESetMaxPCGIter(solver, pcg_maxit); /* max iterations */
      HYPRE_AMESetTol(solver, tol); /* conv. tolerance */
      if (myid == 0)
      {
         HYPRE_AMESetPrintLevel(solver, 1);   /* print solve info */
      }
      else
      {
         HYPRE_AMESetPrintLevel(solver, 0);
      }

      /* Setup */
      HYPRE_AMESetup(solver);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("AME Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_AMESolve(solver);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Destroy solver and preconditioner */
      HYPRE_AMEDestroy(solver);
      HYPRE_AMSDestroy(precond);
   }

   /* Save the solution */
   /* HYPRE_ParVectorPrint(x0,"x.ams"); */

   /* Clean-up */
   HYPRE_ParCSRMatrixDestroy(A);
   HYPRE_ParVectorDestroy(x0);
   HYPRE_ParVectorDestroy(b);
   HYPRE_ParCSRMatrixDestroy(G);

   if (M) { HYPRE_ParCSRMatrixDestroy(M); }

   if (Gx) { HYPRE_ParVectorDestroy(Gx); }
   if (Gy) { HYPRE_ParVectorDestroy(Gy); }
   if (Gz) { HYPRE_ParVectorDestroy(Gz); }

   if (x) { HYPRE_ParVectorDestroy(x); }
   if (y) { HYPRE_ParVectorDestroy(y); }
   if (z) { HYPRE_ParVectorDestroy(z); }

   if (Aalpha) { HYPRE_ParCSRMatrixDestroy(Aalpha); }
   if (Abeta) { HYPRE_ParCSRMatrixDestroy(Abeta); }

   if (zero_cond)
   {
      HYPRE_ParVectorDestroy(interior_nodes);
   }

   /* Finalize Hypre */
   HYPRE_Finalize();

   /* Finalize MPI */
   hypre_MPI_Finalize();

   if (HYPRE_GetError() && !myid)
   {
      hypre_fprintf(stderr, "hypre_error_flag = %d\n", HYPRE_GetError());
   }

   return 0;
}
