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

HYPRE_Int *argn_ref;
#define ArgSet(argi, argn, index)  argi = argn = index; argn_ref = &argn
#define ArgInc()                  *argn_ref++

/*--------------------------------------------------------------------------
 * General Solver Options
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSolverGeneralHelp()
{
   hypre_printf("Solver General Options: [<options>]\n");
   hypre_printf("\n");
   hypre_printf("  -tol <val>         : convergence tolerance (default 1e-6)\n");
   hypre_printf("  -atol <val>        : absolute tolerance\n");
   hypre_printf("  -max_iter <val>    : max iterations\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSolverGeneralOptions(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Real *tol_ptr,
   HYPRE_Real *atol_ptr,
   HYPRE_Int  *max_iter_ptr )
{
   HYPRE_Real tol      = 1.0e-6;
   HYPRE_Real atol     = 0.0;
   HYPRE_Int  max_iter = 100;

   HYPRE_Int  arg_index;

   arg_index = argi;

   while (arg_index < argn)
   {
      if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atol") == 0 )
      {
         arg_index++;
         atol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-max_iter") == 0 )
      {
         arg_index++;
         max_iter = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   *tol_ptr      = tol;
   *atol_ptr     = atol;
   *max_iter_ptr = max_iter;

   return 0;
}

/*--------------------------------------------------------------------------
 * PCG Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DrivePCGHelp()
{
   hypre_printf("PCG Options: [<gen options>] [<options>] [-diag | <precond>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DrivePCGSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver )
{
   HYPRE_Real tol, atol;
   HYPRE_Int  max_iter;

   /* Set general options */
   hypre_DriveSolverGeneralOptions(argv, argi, argn, &tol, &atol, &max_iter);
   HYPRE_PCGSetTol(solver, tol);
   HYPRE_PCGSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_PCGSetTwoNorm(solver, 1);
   HYPRE_PCGSetRelChange(solver, 0);
   HYPRE_PCGSetPrintLevel(solver, 1);

   if (precond_solve != NULL)
   {
      HYPRE_PCGSetPrecond(solver, precond_solve, precond_setup, precond);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * GMRES Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveGMRESHelp()
{
   hypre_printf("GMRES Options: [<gen options>] [<options>] [-diag | <precond>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveGMRESSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver )
{
   HYPRE_Real tol, atol;
   HYPRE_Int  max_iter;

   /* Set general options */
   hypre_DriveSolverGeneralOptions(argv, argi, argn, &tol, &atol, &max_iter);
   HYPRE_GMRESSetTol(solver, tol);
   HYPRE_GMRESSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_GMRESSetPrintLevel(solver, 1);
   HYPRE_GMRESSetLogging(solver, 1);
   HYPRE_GMRESSetKDim(solver, 5);

   if (precond_solve != NULL)
   {
      HYPRE_GMRESSetPrecond(solver, precond_solve, precond_setup, precond);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * BiCGSTAB Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveBiCGSTABHelp()
{
   hypre_printf("BiCGSTAB Options: [<gen options>] [<options>] [-diag | <precond>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveBiCGSTABSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver )
{
   HYPRE_Real tol, atol;
   HYPRE_Int  max_iter;

   /* Set general options */
   hypre_DriveSolverGeneralOptions(argv, argi, argn, &tol, &atol, &max_iter);
   HYPRE_BiCGSTABSetTol(solver, tol);
   HYPRE_BiCGSTABSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_BiCGSTABSetPrintLevel(solver, 1);
   HYPRE_BiCGSTABSetLogging(solver, 1);

   if (precond_solve != NULL)
   {
      HYPRE_BiCGSTABSetPrecond(solver, precond_solve, precond_setup, precond);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * FlexGMRES Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveFlexGMRESHelp()
{
   hypre_printf("FlexGMRES Options: [<gen options>] [<options>] [-diag | <precond>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveFlexGMRESSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver )
{
   HYPRE_Real tol, atol;
   HYPRE_Int  max_iter;

   /* Set general options */
   hypre_DriveSolverGeneralOptions(argv, argi, argn, &tol, &atol, &max_iter);
   HYPRE_FlexGMRESSetTol(solver, tol);
   HYPRE_FlexGMRESSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_FlexGMRESSetPrintLevel(solver, 1);
   HYPRE_FlexGMRESSetLogging(solver, 1);
   HYPRE_FlexGMRESSetKDim(solver, 5);

   if (precond_solve != NULL)
   {
      HYPRE_FlexGMRESSetPrecond(solver, precond_solve, precond_setup, precond);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * LGMRES Solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveLGMRESHelp()
{
   hypre_printf("LGMRES Options: [<gen options>] [<options>] [-diag | <precond>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveLGMRESSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver )
{
   HYPRE_Real tol, atol;
   HYPRE_Int  max_iter;

   /* Set general options */
   hypre_DriveSolverGeneralOptions(argv, argi, argn, &tol, &atol, &max_iter);
   HYPRE_LGMRESSetTol(solver, tol);
   HYPRE_LGMRESSetMaxIter(solver, max_iter);

   /* Other options */
   HYPRE_LGMRESSetPrintLevel(solver, 1);
   HYPRE_LGMRESSetLogging(solver, 1);
   HYPRE_LGMRESSetKDim(solver, 10);
   HYPRE_LGMRESSetAugDim(solver, 2);

   if (precond_solve != NULL)
   {
      HYPRE_LGMRESSetPrecond(solver, precond_solve, precond_setup, precond);
   }

   return 0;
}

