/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef HYPRE_DRIVE_SOLVER_HEADER
#define HYPRE_DRIVE_SOLVER_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * Prototypes for driver solver
 *--------------------------------------------------------------------------*/

/* hypre_solver.c */

HYPRE_Int
hypre_DriveSolverGeneralHelp();

HYPRE_Int
hypre_DriveSolverGeneralOptions(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Real *tol_ptr,
   HYPRE_Real *atol_ptr,
   HYPRE_Int  *max_iter_ptr);

HYPRE_Int
hypre_DrivePCGHelp();

HYPRE_Int
hypre_DrivePCGSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver);

HYPRE_Int
hypre_DriveGMRESHelp();

HYPRE_Int
hypre_DriveGMRESSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver);

HYPRE_Int
hypre_DriveBiCGSTABHelp();

HYPRE_Int
hypre_DriveBiCGSTABSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver);

HYPRE_Int
hypre_DriveFlexGMRESHelp();

HYPRE_Int
hypre_DriveFlexGMRESSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver);

HYPRE_Int
hypre_DriveLGMRESHelp();

HYPRE_Int
hypre_DriveLGMRESSet(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_Solver precond,
   HYPRE_PtrToSolverFcn precond_solve,
   HYPRE_PtrToSolverFcn precond_setup,
   HYPRE_Solver solver);

/* hypre_solver_struct.c */

HYPRE_Int
hypre_DriveSolverStructHelp();

HYPRE_Int
hypre_DriveSolveStruct(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_StructMatrix A,
   HYPRE_StructVector b,
   HYPRE_StructVector x);

/* hypre_solver_sstruct.c */

HYPRE_Int
hypre_DriveSolverSStructHelp();

HYPRE_Int
hypre_DriveSolveSStruct(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x);

/* hypre_solver_parcsr.c */

HYPRE_Int
hypre_DriveSolverParCSRHelp();

HYPRE_Int
hypre_DriveSolveParCSR(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_ParCSRMatrix A,
   HYPRE_ParVector    b,
   HYPRE_ParVector    x);

#endif
