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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSolverCreate(
   hypre_DriveSolver *solver )
{
   (solver -> id)     = NONE;
   (solver -> solver) = NULL;
   (solver -> solve)  = NULL;
   (solver -> setup)  = NULL;
   (solver -> argv)   = NULL;
   (solver -> argc)   = 0;

   return 0;
}

HYPRE_Int
hypre_DriveSolverStdHelp()
{
   hypre_printf("SolverStdOptions: [<options>]\n");
   hypre_printf("\n");
   hypre_printf("  -tol <val>         : convergence tolerance (default 1e-6)\n");
   hypre_printf("  -atol <val>        : absolute tolerance\n");
   hypre_printf("  -max_iter <val>    : max iterations\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSolverStdDefaults(
   HYPRE_Real *tol_ptr,
   HYPRE_Real *atol_ptr,
   HYPRE_Int  *max_iter_ptr )
{
   *tol_ptr      = 1.0e-6;
   *atol_ptr     = 0.0;
   *max_iter_ptr = 100;

   return 0;
}

HYPRE_Int
hypre_DriveSolverStdOptions(
   char       *argv[],
   HYPRE_Int   argc,
   HYPRE_Real *tol_ptr,
   HYPRE_Real *atol_ptr,
   HYPRE_Int  *max_iter_ptr )
{
   HYPRE_Int  argi, argn;

   ArgInit(argv, &argi, &argn);
   while (argi < argc)
   {
      if ( strcmp(argv[argi], "-tol") == 0 )
      {
         argi++;
         *tol_ptr = atof(argv[argi++]);
      }
      else if ( strcmp(argv[argi], "-atol") == 0 )
      {
         argi++;
         *atol_ptr = atof(argv[argi++]);
      }
      else if ( strcmp(argv[argi], "-max_iter") == 0 )
      {
         argi++;
         *max_iter_ptr = atoi(argv[argi++]);
      }
      else
      {
         ArgNext(argv, &argi, &argn);
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Krylov Solvers
 *--------------------------------------------------------------------------*/

/* PCG Solver */

HYPRE_Int
hypre_DrivePCGHelp()
{
   hypre_printf("PCGOptions: <SolverStdOptions> [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DrivePCGSet(
   HYPRE_Solver solver,
   char      *argv[],
   HYPRE_Int  argc,
   HYPRE_Real tol,
   HYPRE_Real atol,
   HYPRE_Int  max_iter )
{
   /* Set general options */
   hypre_DriveSolverStdOptions(argv, argc, &tol, &atol, &max_iter);
   HYPRE_PCGSetTol(solver, tol);
   HYPRE_PCGSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_PCGSetTwoNorm(solver, 1);
   HYPRE_PCGSetRelChange(solver, 0);
   HYPRE_PCGSetPrintLevel(solver, 1);

   return 0;
}

/* GMRES Solver */

HYPRE_Int
hypre_DriveGMRESHelp()
{
   hypre_printf("GMRESOptions: <SolverStdOptions> [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveGMRESSet(
   HYPRE_Solver solver,
   char      *argv[],
   HYPRE_Int  argc,
   HYPRE_Real tol,
   HYPRE_Real atol,
   HYPRE_Int  max_iter )
{
   /* Set general options */
   hypre_DriveSolverStdOptions(argv, argc, &tol, &atol, &max_iter);
   HYPRE_GMRESSetTol(solver, tol);
   HYPRE_GMRESSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_GMRESSetPrintLevel(solver, 1);
   HYPRE_GMRESSetLogging(solver, 1);
   HYPRE_GMRESSetKDim(solver, 5);

   return 0;
}

/* BiCGSTAB Solver */

HYPRE_Int
hypre_DriveBiCGSTABHelp()
{
   hypre_printf("BiCGSTABOptions: <SolverStdOptions> [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveBiCGSTABSet(
   HYPRE_Solver solver,
   char      *argv[],
   HYPRE_Int  argc,
   HYPRE_Real tol,
   HYPRE_Real atol,
   HYPRE_Int  max_iter )
{
   /* Set general options */
   hypre_DriveSolverStdOptions(argv, argc, &tol, &atol, &max_iter);
   HYPRE_BiCGSTABSetTol(solver, tol);
   HYPRE_BiCGSTABSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_BiCGSTABSetPrintLevel(solver, 1);
   HYPRE_BiCGSTABSetLogging(solver, 1);

   return 0;
}

/* FlexGMRES Solver */

HYPRE_Int
hypre_DriveFlexGMRESHelp()
{
   hypre_printf("FlexGMRESOptions: <SolverStdOptions> [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveFlexGMRESSet(
   HYPRE_Solver solver,
   char      *argv[],
   HYPRE_Int  argc,
   HYPRE_Real tol,
   HYPRE_Real atol,
   HYPRE_Int  max_iter )
{
   /* Set general options */
   hypre_DriveSolverStdOptions(argv, argc, &tol, &atol, &max_iter);
   HYPRE_FlexGMRESSetTol(solver, tol);
   HYPRE_FlexGMRESSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_FlexGMRESSetPrintLevel(solver, 1);
   HYPRE_FlexGMRESSetLogging(solver, 1);
   HYPRE_FlexGMRESSetKDim(solver, 5);

   return 0;
}

/* LGMRES Solver */

HYPRE_Int
hypre_DriveLGMRESHelp()
{
   hypre_printf("LGMRESOptions: <SolverStdOptions> [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveLGMRESSet(
   HYPRE_Solver solver,
   char      *argv[],
   HYPRE_Int  argc,
   HYPRE_Real tol,
   HYPRE_Real atol,
   HYPRE_Int  max_iter )
{
   /* Set general options */
   hypre_DriveSolverStdOptions(argv, argc, &tol, &atol, &max_iter);
   HYPRE_LGMRESSetTol(solver, tol);
   HYPRE_LGMRESSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_LGMRESSetPrintLevel(solver, 1);
   HYPRE_LGMRESSetLogging(solver, 1);
   HYPRE_LGMRESSetKDim(solver, 10);
   HYPRE_LGMRESSetAugDim(solver, 2);

   return 0;
}

/* Krylov Solver */

HYPRE_Int
hypre_DriveKrylovHelp()
{
   hypre_printf("KrylovOptions: [<option>]\n");
   hypre_printf("\n");
   hypre_printf("  -pcg       { <PCGOptions> }\n");
   hypre_printf("  -gmres     { <GMRESOptions> }\n");
   hypre_printf("  -bicgstab  { <BiCGSTABOptions> }\n");
   hypre_printf("  -flexgmres { <FlexGMRESOptions> }\n");
   hypre_printf("  -lgmres    { <LGMRESOptions> }\n");
   hypre_printf("\n");
   hypre_DrivePCGHelp();
   hypre_DriveGMRESHelp();
   hypre_DriveBiCGSTABHelp();
   hypre_DriveFlexGMRESHelp();
   hypre_DriveLGMRESHelp();

   return 0;
}

HYPRE_Int
hypre_DriveKrylovCreate(
   char      *argv[],
   HYPRE_Int  argc,
   hypre_DriveSolver *solver_ptr )
{
   hypre_DriveSolver solver;
   HYPRE_Int         argi, argn, tmp_argi;

   hypre_DriveSolverCreate(&solver);

   ArgInit(argv, &argi, &argn);
   while (argi < argc)
   {
      if ( strcmp(argv[argi], "-pcg") == 0 )
      {
         solver.id = PCG;
      }
      else if ( strcmp(argv[argi], "-gmres") == 0 )
      {
         solver.id = GMRES;
      }
      else if ( strcmp(argv[argi], "-bicgstab") == 0 )
      {
         solver.id = BiCGSTAB;
      }
      else if ( strcmp(argv[argi], "-flexgmres") == 0 )
      {
         solver.id = FlexGMRES;
      }
      else if ( strcmp(argv[argi], "-lgmres") == 0 )
      {
         solver.id = LGMRES;
      }
      ArgNext(argv, &argi, &argn);

      if (solver.id != NONE)
      {
         ArgStripBraces(argv, argi, argn, &solver.argv, &tmp_argi, &solver.argc);
         break;
      }
   }

   *solver_ptr = solver;

   return 0;
}

HYPRE_Int
hypre_DriveKrylovSetup(
   hypre_DriveSolver solver,
   hypre_DriveSolver precond,
   HYPRE_Real   tol,
   HYPRE_Real   atol,
   HYPRE_Int    max_iter,
   HYPRE_Matrix A,
   HYPRE_Vector b,
   HYPRE_Vector x )
{
   switch (solver.id)
   {
      case PCG:
      {
         hypre_DrivePCGSet(solver.solver, solver.argv, solver.argc, tol, atol, max_iter);
         if (precond.solve != NULL)
         {
            HYPRE_PCGSetPrecond(solver.solver, precond.solve, precond.setup, precond.solver);
         }
         HYPRE_PCGSetup(solver.solver, A, b, x);
      }
      break;

      case GMRES:
      {
         hypre_DriveGMRESSet(solver.solver, solver.argv, solver.argc, tol, atol, max_iter);
         if (precond.solve != NULL)
         {
            HYPRE_GMRESSetPrecond(solver.solver, precond.solve, precond.setup, precond.solver);
         }
         HYPRE_GMRESSetup(solver.solver, A, b, x);
      }
      break;

      case BiCGSTAB:
      {
         hypre_DriveBiCGSTABSet(solver.solver, solver.argv, solver.argc, tol, atol, max_iter);
         if (precond.solve != NULL)
         {
            HYPRE_BiCGSTABSetPrecond(solver.solver, precond.solve, precond.setup, precond.solver);
         }
         HYPRE_BiCGSTABSetup(solver.solver, A, b, x);
      }
      break;

      case FlexGMRES:
      {
         hypre_DriveFlexGMRESSet(solver.solver, solver.argv, solver.argc, tol, atol, max_iter);
         if (precond.solve != NULL)
         {
            HYPRE_FlexGMRESSetPrecond(solver.solver, precond.solve, precond.setup, precond.solver);
         }
         HYPRE_FlexGMRESSetup(solver.solver, A, b, x);
      }
      break;

      case LGMRES:
      {
         hypre_DriveLGMRESSet(solver.solver, solver.argv, solver.argc, tol, atol, max_iter);
         if (precond.solve != NULL)
         {
            HYPRE_LGMRESSetPrecond(solver.solver, precond.solve, precond.setup, precond.solver);
         }
         HYPRE_LGMRESSetup(solver.solver, A, b, x);
      }
      break;
   }

   return 0;
}

HYPRE_Int
hypre_DriveKrylovSolve(
   hypre_DriveSolver solver,
   HYPRE_Matrix A,
   HYPRE_Vector b,
   HYPRE_Vector x )
{
   switch (solver.id)
   {
      case PCG:
      {
         HYPRE_PCGSolve(solver.solver, A, b, x);
      }
      break;

      case GMRES:
      {
         HYPRE_GMRESSolve(solver.solver, A, b, x);
      }
      break;

      case BiCGSTAB:
      {
         HYPRE_BiCGSTABSolve(solver.solver, A, b, x);
      }
      break;

      case FlexGMRES:
      {
         HYPRE_FlexGMRESSolve(solver.solver, A, b, x);
      }
      break;

      case LGMRES:
      {
         HYPRE_LGMRESSolve(solver.solver, A, b, x);
      }
      break;
   }

   return 0;
}

HYPRE_Int
hypre_DriveKrylovGetStats(
   hypre_DriveSolver solver,
   HYPRE_Int   *num_iterations_ptr,
   HYPRE_Real  *final_res_norm_ptr )
{
   HYPRE_Int   num_iterations;
   HYPRE_Real  final_res_norm;

   switch (solver.id)
   {
      case PCG:
      {
         HYPRE_PCGGetNumIterations(solver.solver, &num_iterations);
         HYPRE_PCGGetFinalRelativeResidualNorm(solver.solver, &final_res_norm);
      }
      break;

      case GMRES:
      {
         HYPRE_GMRESGetNumIterations(solver.solver, &num_iterations);
         HYPRE_GMRESGetFinalRelativeResidualNorm(solver.solver, &final_res_norm);
      }
      break;

      case BiCGSTAB:
      {
         HYPRE_BiCGSTABGetNumIterations(solver.solver, &num_iterations);
         HYPRE_BiCGSTABGetFinalRelativeResidualNorm(solver.solver, &final_res_norm);
      }
      break;

      case FlexGMRES:
      {
         HYPRE_FlexGMRESGetNumIterations(solver.solver, &num_iterations);
         HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver.solver, &final_res_norm);
      }
      break;

      case LGMRES:
      {
         HYPRE_LGMRESGetNumIterations(solver.solver, &num_iterations);
         HYPRE_LGMRESGetFinalRelativeResidualNorm(solver.solver, &final_res_norm);
      }
      break;
   }

   *num_iterations_ptr = num_iterations;
   *final_res_norm_ptr = final_res_norm;

   return 0;
}

