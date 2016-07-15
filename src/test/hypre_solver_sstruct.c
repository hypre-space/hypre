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

#define NONE      0
#define SPLIT     1
#define SYSPFMG   2
#define DIAG      9
#define PCG       101
#define GMRES     102
#define BiCGSTAB  103
#define FlexGMRES 104
#define LGMRES    105

static HYPRE_Int  argn_notset = 0;
static HYPRE_Int *argn_ref = &argn_notset;
#define ArgSet(argi, argn, index)  argi = argn = index; argn_ref = &argn
#define ArgInc()                   (*argn_ref)++

/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Split Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSplitHelp()
{
   hypre_printf("Split Options: [<gen options>] [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSplitSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Int  preconditioner,
   HYPRE_Int  split_solver,
   HYPRE_SStructSolver solver )
{
   HYPRE_Real tol, atol;
   HYPRE_Int  max_iter;

   /* Set general options */
   hypre_DriveSolverGeneralOptions(argv, argi, argn, &tol, &atol, &max_iter);
   HYPRE_SStructSplitSetTol(solver, tol);
   HYPRE_SStructSplitSetMaxIter(solver, max_iter);

   /* Preconditioner options */
   if (preconditioner)
   {
      HYPRE_SStructSplitSetTol(solver, 0.0);
      HYPRE_SStructSplitSetMaxIter(solver, 1);
      HYPRE_SStructSplitSetZeroGuess(solver);
   }

   HYPRE_SStructSplitSetStructSolver(solver, split_solver);

   return 0;
}

/*--------------------------------------------------------------------------
 * SysPFMG Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSysPFMGHelp()
{
   hypre_printf("SysPFMG Options: [<gen options>] [<options>]\n");
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
hypre_DriveSysPFMGSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Int  preconditioner,
   HYPRE_SStructSolver solver )
{
   HYPRE_Real tol, atol;
   HYPRE_Int  max_iter;

   HYPRE_Int  n_pre, n_post;
   HYPRE_Int  relax;
   HYPRE_Real jacobi_weight;
   HYPRE_Int  skip;

   HYPRE_Int  arg_index;

   /* Set general options */
   hypre_DriveSolverGeneralOptions(argv, argi, argn, &tol, &atol, &max_iter);
   HYPRE_SStructSysPFMGSetTol(solver, tol);
   HYPRE_SStructSysPFMGSetMaxIter(solver, max_iter);

   /* Set command-line options */
   arg_index = argi;
   while (arg_index < argn)
   {
      if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
         HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
         HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
      }
      else if ( strcmp(argv[arg_index], "-relax") == 0 )
      {
         arg_index++;
         relax = atoi(argv[arg_index++]);
         HYPRE_SStructSysPFMGSetRelaxType(solver, relax);
         /* weighted Jacobi = 1; red-black GS = 2 */
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         jacobi_weight= atof(argv[arg_index++]);
         HYPRE_SStructSysPFMGSetJacobiWeight(solver, jacobi_weight);
      }
      else if ( strcmp(argv[arg_index], "-skip") == 0 )
      {
         arg_index++;
         skip = atoi(argv[arg_index++]);
         HYPRE_SStructSysPFMGSetSkipRelax(solver, skip);
      }
      else
      {
         arg_index++;
      }
   }

   /* Other options */
   HYPRE_SStructSysPFMGSetRelChange(solver, 0);
   /*HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
   HYPRE_SStructSysPFMGSetPrintLevel(solver, 1);
   HYPRE_SStructSysPFMGSetLogging(solver, 1);

   /* Preconditioner options */
   if (preconditioner)
   {
      HYPRE_SStructSysPFMGSetTol(solver, 0.0);
      HYPRE_SStructSysPFMGSetMaxIter(solver, 1);
      HYPRE_SStructSysPFMGSetZeroGuess(solver);
      HYPRE_SStructSysPFMGSetPrintLevel(solver, 0);
      HYPRE_SStructSysPFMGSetLogging(solver, 0);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * SStruct Solver Options
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSolverSStructHelp( )
{
   hypre_printf("Solver SStruct Options: [<options>]\n");
   hypre_printf("\n");
   hypre_printf("  -smgsplit  <options>\n");
   hypre_printf("  -pfmgsplit <options>\n");
   hypre_printf("  -syspfmg   <options>\n");
   hypre_printf("  -pcg       <options>\n");
   hypre_printf("  -gmres     <options>\n");
   hypre_printf("  -bicgstab  <options>\n");
   hypre_printf("  -flexgmres <options>\n");
   hypre_printf("  -lgmres    <options>\n");
   hypre_printf("\n");
   hypre_DriveSplitHelp();
   hypre_DriveSysPFMGHelp();
   hypre_DrivePCGHelp();
   hypre_DriveGMRESHelp();
   hypre_DriveBiCGSTABHelp();
   hypre_DriveFlexGMRESHelp();
   hypre_DriveLGMRESHelp();

   return 0;
}

/*--------------------------------------------------------------------------
 * Driver for semi-structured (SStruct) solver interface
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_DriveSolveSStruct(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x )
{
   HYPRE_SStructSolver   solver, precond;

   HYPRE_Int             arg_solver_id, solver_id, precond_id;
                        
   HYPRE_Int             num_iterations;
   HYPRE_Real            final_res_norm;
                         
   HYPRE_Int             num_procs, myid;
   HYPRE_Int             time_index;

   HYPRE_Int             arg_index;

   HYPRE_Int             split_argi,     split_argn;
   HYPRE_Int             syspfmg_argi,   syspfmg_argn;
   HYPRE_Int             pcg_argi,       pcg_argn;
   HYPRE_Int             gmres_argi,     gmres_argn;
   HYPRE_Int             bicgstab_argi,  bicgstab_argn;
   HYPRE_Int             flexgmres_argi, flexgmres_argn;
   HYPRE_Int             lgmres_argi,    lgmres_argn;
                        
   HYPRE_Int             split_solver;

   HYPRE_PtrToSolverFcn  precond_solve, precond_setup;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   solver_id     = NONE;
   precond_id    = NONE;

   precond_solve = NULL;
   precond_setup = NULL;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   arg_index = argi;

   while (arg_index < argn)
   {
      arg_solver_id = NONE;

      if ( strcmp(argv[arg_index], "-smgsplit") == 0 )
      {
         arg_index++;
         arg_solver_id = SPLIT;
         ArgSet(split_argi, split_argn, arg_index);
         split_solver = HYPRE_SMG;
      }
      else if ( strcmp(argv[arg_index], "-pfmgsplit") == 0 )
      {
         arg_index++;
         arg_solver_id = SPLIT;
         ArgSet(split_argi, split_argn, arg_index);
         split_solver = HYPRE_PFMG;
      }
      else if ( strcmp(argv[arg_index], "-syspfmg") == 0 )
      {
         arg_index++;
         arg_solver_id = SYSPFMG;
         ArgSet(syspfmg_argi, syspfmg_argn, arg_index);
      }
      else if ( strcmp(argv[arg_index], "-diag") == 0 )
      {
         arg_index++;
         precond_id = DIAG;
      }
      else if ( strcmp(argv[arg_index], "-pcg") == 0 )
      {
         arg_index++;
         arg_solver_id = PCG;
         ArgSet(pcg_argi, pcg_argn, arg_index);
      }
      else if ( strcmp(argv[arg_index], "-gmres") == 0 )
      {
         arg_index++;
         arg_solver_id = GMRES;
         ArgSet(gmres_argi, gmres_argn, arg_index);
      }
      else if ( strcmp(argv[arg_index], "-bicgstab") == 0 )
      {
         arg_index++;
         arg_solver_id = BiCGSTAB;
         ArgSet(bicgstab_argi, bicgstab_argn, arg_index);
      }
      else if ( strcmp(argv[arg_index], "-flexgmres") == 0 )
      {
         arg_index++;
         arg_solver_id = FlexGMRES;
         ArgSet(flexgmres_argi, flexgmres_argn, arg_index);
      }
      else if ( strcmp(argv[arg_index], "-lgmres") == 0 )
      {
         arg_index++;
         arg_solver_id = LGMRES;
         ArgSet(lgmres_argi, lgmres_argn, arg_index);
      }
      else
      {
         arg_index++;
         ArgInc();
      }

      /* First solver argument is the solver, second is the preconditioner */
      if (arg_solver_id != NONE)
      {
         if (solver_id == NONE)
         {
            solver_id = arg_solver_id;
         }
         else if (precond_id == NONE)
         {
            precond_id = arg_solver_id;
         }
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if (solver_id == NONE)
   {
      solver_id = GMRES;
   }

   /*-----------------------------------------------------------
    * Set up the solver
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct Solver Setup");
   hypre_BeginTiming(time_index);

   switch (precond_id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitCreate(hypre_MPI_COMM_WORLD, &precond);
         hypre_DriveSplitSet(argv, split_argi, split_argn, 1,
                             split_solver, precond);
         precond_solve = (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve;
         precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup;
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGCreate(hypre_MPI_COMM_WORLD, &precond);
         hypre_DriveSysPFMGSet(argv, syspfmg_argi, syspfmg_argn, 1, precond);
         precond_solve = (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSolve;
         precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_SStructSysPFMGSetup;
      }
      break;

      case DIAG:
      {
         precond = NULL;
         precond_solve = (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScale;
         precond_setup = (HYPRE_PtrToSolverFcn) HYPRE_SStructDiagScaleSetup;
      }
      break;
   }

   switch (solver_id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitCreate(hypre_MPI_COMM_WORLD, &solver);
         hypre_DriveSplitSet(argv, split_argi, split_argn, 0,
                             split_solver, solver);
         HYPRE_SStructSplitSetup(solver, A, b, x);
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGCreate(hypre_MPI_COMM_WORLD, &solver);
         hypre_DriveSysPFMGSet(argv, syspfmg_argi, syspfmg_argn, 0, solver);
         HYPRE_SStructSysPFMGSetup(solver, A, b, x);
      }
      break;

      case PCG:
      {
         HYPRE_SStructPCGCreate(hypre_MPI_COMM_WORLD, &solver);
         hypre_DrivePCGSet(argv, pcg_argi, pcg_argn,
                           (HYPRE_Solver) precond, precond_solve,
                           precond_setup, (HYPRE_Solver) solver);
         HYPRE_PCGSetup((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case GMRES:
      {
         HYPRE_SStructGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
         hypre_DriveGMRESSet(argv, gmres_argi, gmres_argn,
                             (HYPRE_Solver) precond, precond_solve,
                             precond_setup, (HYPRE_Solver) solver);
         HYPRE_GMRESSetup((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                          (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case BiCGSTAB:
      {
         HYPRE_SStructBiCGSTABCreate(hypre_MPI_COMM_WORLD, &solver);
         hypre_DriveBiCGSTABSet(argv, bicgstab_argi, bicgstab_argn,
                                (HYPRE_Solver) precond, precond_solve,
                                precond_setup, (HYPRE_Solver) solver);
         HYPRE_BiCGSTABSetup((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                             (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case FlexGMRES:
      {
         HYPRE_SStructFlexGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
         hypre_DriveFlexGMRESSet(argv, flexgmres_argi, flexgmres_argn,
                                 (HYPRE_Solver) precond, precond_solve,
                                 precond_setup, (HYPRE_Solver) solver);
         HYPRE_FlexGMRESSetup((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                              (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case LGMRES:
      {
         HYPRE_SStructLGMRESCreate(hypre_MPI_COMM_WORLD, &solver);
         hypre_DriveLGMRESSet(argv, lgmres_argi, lgmres_argn,
                              (HYPRE_Solver) precond, precond_solve,
                              precond_setup, (HYPRE_Solver) solver);
         HYPRE_LGMRESSetup((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                           (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Solve the system
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct Solver Solve");
   hypre_BeginTiming(time_index);

   switch (solver_id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitSolve(solver, A, b, x);
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGSolve(solver, A, b, x);
      }
      break;

      case PCG:
      {
         HYPRE_PCGSolve((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                        (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case GMRES:
      {
         HYPRE_GMRESSolve((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                          (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case BiCGSTAB:
      {
         HYPRE_BiCGSTABSolve((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                             (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case FlexGMRES:
      {
         HYPRE_FlexGMRESSolve((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                              (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;

      case LGMRES:
      {
         HYPRE_LGMRESSolve((HYPRE_Solver) solver, (HYPRE_Matrix) A,
                           (HYPRE_Vector) b, (HYPRE_Vector) x);
      }
      break;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Clean up
    *-----------------------------------------------------------*/

   switch (precond_id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitDestroy(precond);
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGDestroy(precond);
      }
      break;
   }

   switch (solver_id)
   {
      case SPLIT:
      {
         HYPRE_SStructSplitGetNumIterations(solver, &num_iterations);
         HYPRE_SStructSplitGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_SStructSplitDestroy(solver);
      }
      break;

      case SYSPFMG:
      {
         HYPRE_SStructSysPFMGGetNumIterations(solver, &num_iterations);
         HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
         HYPRE_SStructSysPFMGDestroy(solver);
      }
      break;

      case PCG:
      {
         HYPRE_PCGGetNumIterations((HYPRE_Solver) solver, &num_iterations);
         HYPRE_PCGGetFinalRelativeResidualNorm(
            (HYPRE_Solver) solver, &final_res_norm);
         HYPRE_SStructPCGDestroy(solver);
      }
      break;

      case GMRES:
      {
         HYPRE_GMRESGetNumIterations((HYPRE_Solver) solver, &num_iterations);
         HYPRE_GMRESGetFinalRelativeResidualNorm(
            (HYPRE_Solver) solver, &final_res_norm);
         HYPRE_SStructGMRESDestroy(solver);
      }
      break;

      case BiCGSTAB:
      {
         HYPRE_BiCGSTABGetNumIterations((HYPRE_Solver) solver, &num_iterations);
         HYPRE_BiCGSTABGetFinalRelativeResidualNorm(
            (HYPRE_Solver) solver, &final_res_norm);
         HYPRE_SStructBiCGSTABDestroy(solver);
      }
      break;

      case FlexGMRES:
      {
         HYPRE_FlexGMRESGetNumIterations((HYPRE_Solver) solver, &num_iterations);
         HYPRE_FlexGMRESGetFinalRelativeResidualNorm(
            (HYPRE_Solver) solver, &final_res_norm);
         HYPRE_SStructFlexGMRESDestroy(solver);
      }
      break;

      case LGMRES:
      {
         HYPRE_LGMRESGetNumIterations((HYPRE_Solver) solver, &num_iterations);
         HYPRE_LGMRESGetFinalRelativeResidualNorm(
            (HYPRE_Solver) solver, &final_res_norm);
         HYPRE_SStructLGMRESDestroy(solver);
      }
      break;
   }

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

   return (0);
}
