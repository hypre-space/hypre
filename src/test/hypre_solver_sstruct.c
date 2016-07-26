/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.52 $
 ***********************************************************************EHEADER*/

#include "hypre_solver.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * Split Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSplitHelp()
{
   hypre_printf("SplitOptions: <SolverStdOptions> [<options>]\n");
   hypre_printf("\n");
   hypre_printf("  -pfmg          : Use PFMG for Struct Solver\n");
   hypre_printf("  -smg           : Use SMG for Struct Solver\n");
   hypre_printf("  -jac           : Use Jacobi for Struct Solver\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSplitSetup(
   HYPRE_SStructSolver solver,
   char      *argv[],
   HYPRE_Int  argc,
   HYPRE_Int  precond_bool,
   HYPRE_Real tol,
   HYPRE_Real atol,
   HYPRE_Int  max_iter,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x )
{
   HYPRE_Int  argi;

   /* Set general options */
   hypre_DriveSolverStdOptions(argv, argc, &tol, &atol, &max_iter);
   HYPRE_SStructSplitSetTol(solver, tol);
   HYPRE_SStructSplitSetMaxIter(solver, max_iter);

   /* Parse command line */
   argi = 0;
   while (argi < argc)
   {
      if ( strcmp(argv[argi], "-pfmg") == 0 )
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_PFMG);
         argi++;
      }
      else if ( strcmp(argv[argi], "-smg") == 0 )
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_SMG);
         argi++;
      }
      else if ( strcmp(argv[argi], "-jac") == 0 )
      {
         HYPRE_SStructSplitSetStructSolver(solver, HYPRE_Jacobi);
         argi++;
      }
   }

   /* Preconditioner options */
   if (precond_bool)
   {
      HYPRE_SStructSplitSetTol(solver, 0.0);
      HYPRE_SStructSplitSetMaxIter(solver, 1);
      HYPRE_SStructSplitSetZeroGuess(solver);
   }
   else
   {
      HYPRE_SStructSplitSetup(solver, A, b, x);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * SysPFMG Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSysPFMGHelp()
{
   hypre_printf("SysPFMGOptions: <SolverStdOptions> [<options>]\n");
   hypre_printf("\n");
   hypre_printf("  -v <n_pre> <n_post>: Number of pre and post relax\n");
   hypre_printf("  -relax <r>         : Relaxation type\n");
   hypre_printf("                        0 - Jacobi\n");
   hypre_printf("                        1 - Weighted Jacobi (default)\n");
   hypre_printf("                        2 - R/B Gauss-Seidel\n");
   hypre_printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
   hypre_printf("  -w <weight>        : Jacobi weight\n");
   hypre_printf("  -skip <s>          : Skip relaxation (0 or 1)\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSysPFMGSetup(
   HYPRE_SStructSolver solver,
   char      *argv[],
   HYPRE_Int  argc,
   HYPRE_Int  precond_bool,
   HYPRE_Real tol,
   HYPRE_Real atol,
   HYPRE_Int  max_iter,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x )
{
   HYPRE_Int  n_pre, n_post;
   HYPRE_Int  relax;
   HYPRE_Real jacobi_weight;
   HYPRE_Int  skip;

   HYPRE_Int  argi;

   /* Set general options */
   hypre_DriveSolverStdOptions(argv, argc, &tol, &atol, &max_iter);
   HYPRE_SStructSysPFMGSetTol(solver, tol);
   HYPRE_SStructSysPFMGSetMaxIter(solver, max_iter);

   /* Set command-line options */
   argi = 0;
   while (argi < argc)
   {
      if ( strcmp(argv[argi], "-v") == 0 )
      {
         argi++;
         n_pre = atoi(argv[argi++]);
         n_post = atoi(argv[argi++]);
         HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
      }
      else if ( strcmp(argv[argi], "-relax") == 0 )
      {
         argi++;
         relax = atoi(argv[argi++]);
         HYPRE_SStructSysPFMGSetRelaxType(solver, relax);
         /* weighted Jacobi = 1; red-black GS = 2 */
      }
      else if ( strcmp(argv[argi], "-w") == 0 )
      {
         argi++;
         jacobi_weight= atof(argv[argi++]);
         HYPRE_SStructSysPFMGSetJacobiWeight(solver, jacobi_weight);
      }
      else if ( strcmp(argv[argi], "-skip") == 0 )
      {
         argi++;
         skip = atoi(argv[argi++]);
         HYPRE_SStructSysPFMGSetSkipRelax(solver, skip);
      }
      else
      {
         argi++;
      }
   }

   /* Other options */
   HYPRE_SStructSysPFMGSetRelChange(solver, 0);
   /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
   HYPRE_SStructSysPFMGSetPrintLevel(solver, 1);
   HYPRE_SStructSysPFMGSetLogging(solver, 1);

   /* Preconditioner options */
   if (precond_bool)
   {
      HYPRE_SStructSysPFMGSetTol(solver, 0.0);
      HYPRE_SStructSysPFMGSetMaxIter(solver, 1);
      HYPRE_SStructSysPFMGSetZeroGuess(solver);
      HYPRE_SStructSysPFMGSetPrintLevel(solver, 0);
      HYPRE_SStructSysPFMGSetLogging(solver, 0);
   }
   else
   {
      HYPRE_SStructSysPFMGSetup(solver, A, b, x);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * SStruct Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSolverSStructHelp()
{
   hypre_printf("SolverSStructOptions: <SolverStdOptions> [<KrylovOptions>] [<option>]\n");
   hypre_printf("\n");
   hypre_printf("  -split   { <SplitOptions> }\n");
   hypre_printf("  -syspfmg { <SysPFMGOptions> }\n");
   hypre_printf("  -diag    { }\n");
   hypre_printf("\n");
   hypre_DriveSplitHelp();
   hypre_DriveSysPFMGHelp();

   return 0;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_DriveSolverSStructCreate(
   char      *argv[],
   HYPRE_Int  argc,
   hypre_DriveSolver *krylov_ptr,
   hypre_DriveSolver *precond_ptr )
{
   hypre_DriveSolver krylov;
   hypre_DriveSolver precond;
   HYPRE_Int         argi, argn, tmp_argi;

   /* Parse command line */

   /* Check for Krylov solver */
   hypre_DriveKrylovCreate(argv, argc, &krylov);

   /* Check for solver/preconditioner */
   hypre_DriveSolverCreate(&precond);
   ArgInit(argv, &argi, &argn);
   while (argi < argc)
   {
      if ( strcmp(argv[argi], "-split") == 0 )
      {
         precond.id = SPLIT;
      }
      else if ( strcmp(argv[argi], "-syspfmg") == 0 )
      {
         precond.id = SYSPFMG;
      }
      else if ( strcmp(argv[argi], "-diag") == 0 )
      {
         precond.id = DIAG;
      }
      ArgNext(argv, &argi, &argn);

      if (precond.id != NONE)
      {
         ArgStripBraces(argv, argi, argn, &precond.argv, &tmp_argi, &precond.argc);
         break;
      }
   }

   /* If nothing is specified, set a default solver */
   if ( (krylov.id == NONE) && (precond.id == NONE) )
   {
      krylov.id = GMRES;
   }

   /* Create the solver */

   switch (krylov.id)
   {
      case PCG:
      {
         HYPRE_SStructPCGCreate(hypre_MPI_COMM_WORLD, (HYPRE_SStructSolver *)&krylov.solver);
      }
      break;

      case GMRES:
      {
         HYPRE_SStructGMRESCreate(hypre_MPI_COMM_WORLD, (HYPRE_SStructSolver *)&krylov.solver);
      }
      break;

      case BiCGSTAB:
      {
         HYPRE_SStructBiCGSTABCreate(hypre_MPI_COMM_WORLD, (HYPRE_SStructSolver *)&krylov.solver);
      }
      break;

      case FlexGMRES:
      {
         HYPRE_SStructFlexGMRESCreate(hypre_MPI_COMM_WORLD, (HYPRE_SStructSolver *)&krylov.solver);
      }
      break;

      case LGMRES:
      {
         HYPRE_SStructLGMRESCreate(hypre_MPI_COMM_WORLD, (HYPRE_SStructSolver *)&krylov.solver);
      }
      break;
   }

   switch (precond.id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitCreate(hypre_MPI_COMM_WORLD, (HYPRE_SStructSolver *)&precond.solver);
         precond.solve = (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve;
         precond.setup = (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup;
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGCreate(hypre_MPI_COMM_WORLD, (HYPRE_SStructSolver *)&precond.solver);
         precond.solve = (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve;
         precond.setup = (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup;
      }
      break;

      case DIAG:
      {
         precond.solver = NULL;
         precond.solve  = (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale;
         precond.setup  = (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup;
      }
      break;
   }

   *krylov_ptr  = krylov;
   *precond_ptr = precond;

   return (0);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_DriveSolverSStructSetup(
   hypre_DriveSolver   solver,
   HYPRE_Int           precond_bool,
   HYPRE_Real          tol,
   HYPRE_Real          atol,
   HYPRE_Int           max_iter,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x )
{
   HYPRE_SStructSolver  sstruct_solver = (HYPRE_SStructSolver) solver.solver;

   switch (solver.id)
   {
      case SPLIT:
      {
         hypre_DriveSplitSetup(sstruct_solver, solver.argv, solver.argc,
                               precond_bool, tol, atol, max_iter, A, b, x);
      }
      break;

      case SYSPFMG:
      {
         hypre_DriveSysPFMGSetup(sstruct_solver, solver.argv, solver.argc,
                                 precond_bool, tol, atol, max_iter, A, b, x);
      }
      break;
   }

   return (0);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_DriveSolverSStructSolve(
   hypre_DriveSolver   solver,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x )
{
   HYPRE_SStructSolver  sstruct_solver = (HYPRE_SStructSolver) solver.solver;

   switch (solver.id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitSolve(sstruct_solver, A, b, x);
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGSolve(sstruct_solver, A, b, x);
      }
      break;
   }

   return (0);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_DriveSolverSStructGetStats(
   hypre_DriveSolver  solver,
   HYPRE_Int         *num_iterations_ptr,
   HYPRE_Real        *final_res_norm_ptr )
{
   HYPRE_SStructSolver  sstruct_solver = (HYPRE_SStructSolver) solver.solver;

   HYPRE_Int   num_iterations;
   HYPRE_Real  final_res_norm;
                        
   switch (solver.id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitGetNumIterations(sstruct_solver, &num_iterations);
         HYPRE_SStructSplitGetFinalRelativeResidualNorm(sstruct_solver, &final_res_norm);
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGGetNumIterations(sstruct_solver, &num_iterations);
         HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(sstruct_solver, &final_res_norm);
      }
      break;
   }

   *num_iterations_ptr = num_iterations;
   *final_res_norm_ptr = final_res_norm;

   return (0);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_DriveSolverSStructDestroy(
   hypre_DriveSolver  krylov,
   hypre_DriveSolver  precond )
{
   HYPRE_SStructSolver  sstruct_krylov_solver  = (HYPRE_SStructSolver) krylov.solver;
   HYPRE_SStructSolver  sstruct_precond_solver = (HYPRE_SStructSolver) precond.solver;

   switch (krylov.id)
   {
      case PCG:
      {
         HYPRE_SStructPCGDestroy(sstruct_krylov_solver);
      }
      break;

      case GMRES:
      {
         HYPRE_SStructGMRESDestroy(sstruct_krylov_solver);
      }
      break;

      case BiCGSTAB:
      {
         HYPRE_SStructBiCGSTABDestroy(sstruct_krylov_solver);
      }
      break;

      case FlexGMRES:
      {
         HYPRE_SStructFlexGMRESDestroy(sstruct_krylov_solver);
      }
      break;

      case LGMRES:
      {
         HYPRE_SStructLGMRESDestroy(sstruct_krylov_solver);
      }
      break;
   }

   switch (precond.id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitDestroy(sstruct_precond_solver);
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGDestroy(sstruct_precond_solver);
      }
      break;
   }

   return (0);
}

