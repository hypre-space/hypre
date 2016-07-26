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

#include "hypre_drive.h"

#include "HYPRE_sstruct_ls.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * DriveSolver IDs
 *--------------------------------------------------------------------------*/

#define NONE      0
#define PCG       1
#define GMRES     2
#define BiCGSTAB  3
#define FlexGMRES 4
#define LGMRES    5
#define DIAG      9

#define AMG       101

#define PFMG      201
#define SMG       202

#define SPLIT     301
#define SYSPFMG   302

/*--------------------------------------------------------------------------
 * DriveSolver data type
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int             id;
   HYPRE_Solver          solver;
   HYPRE_PtrToSolverFcn  solve;
   HYPRE_PtrToSolverFcn  setup;

   char                **argv;
   HYPRE_Int             argc;

} hypre_DriveSolver;

/*--------------------------------------------------------------------------
 * Prototypes for hypre drive solver
 *--------------------------------------------------------------------------*/

/* hypre_solver.c */

HYPRE_Int
hypre_DriveSolverCreate(
   hypre_DriveSolver *solver );

HYPRE_Int
hypre_DriveSolverStdHelp();

HYPRE_Int
hypre_DriveSolverStdDefaults(
   HYPRE_Real *tol_ptr,
   HYPRE_Real *atol_ptr,
   HYPRE_Int  *max_iter_ptr );

HYPRE_Int
hypre_DriveSolverStdOptions(
   char       *argv[],
   HYPRE_Int   argn,
   HYPRE_Real *tol_ptr,
   HYPRE_Real *atol_ptr,
   HYPRE_Int  *max_iter_ptr );

HYPRE_Int
hypre_DriveKrylovHelp();

HYPRE_Int
hypre_DriveKrylovCreate(
   char      *argv[],
   HYPRE_Int  argc,
   hypre_DriveSolver *solver_ptr );

HYPRE_Int
hypre_DriveKrylovSetup(
   hypre_DriveSolver solver,
   hypre_DriveSolver precond,
   HYPRE_Real   tol,
   HYPRE_Real   atol,
   HYPRE_Int    max_iter,
   HYPRE_Matrix A,
   HYPRE_Vector b,
   HYPRE_Vector x );

HYPRE_Int
hypre_DriveKrylovSolve(
   hypre_DriveSolver solver,
   HYPRE_Matrix A,
   HYPRE_Vector b,
   HYPRE_Vector x );

HYPRE_Int
hypre_DriveKrylovGetStats(
   hypre_DriveSolver solver,
   HYPRE_Int   *num_iterations_ptr,
   HYPRE_Real  *final_res_norm_ptr );

/* hypre_solver_struct.c */

HYPRE_Int
hypre_DriveSolverStructHelp();

HYPRE_Int
hypre_DriveSolverStructCreate(
   char      *argv[],
   HYPRE_Int  argc,
   hypre_DriveSolver *krylov,
   hypre_DriveSolver *precond );

HYPRE_Int
hypre_DriveSolverStructSetup(
   hypre_DriveSolver  solver,
   HYPRE_Int          precond_bool,
   HYPRE_Real         tol,
   HYPRE_Real         atol,
   HYPRE_Int          max_iter,
   HYPRE_StructMatrix A,
   HYPRE_StructVector b,
   HYPRE_StructVector x );

HYPRE_Int
hypre_DriveSolverStructSolve(
   hypre_DriveSolver  solver,
   HYPRE_StructMatrix A,
   HYPRE_StructVector b,
   HYPRE_StructVector x );

HYPRE_Int
hypre_DriveSolverStructGetStats(
   hypre_DriveSolver  solver,
   HYPRE_Int         *num_iterations_ptr,
   HYPRE_Real        *final_res_norm_ptr );

HYPRE_Int
hypre_DriveSolverStructDestroy(
   hypre_DriveSolver  solver );

/* hypre_solver_sstruct.c */

HYPRE_Int
hypre_DriveSolverSStructHelp();

HYPRE_Int
hypre_DriveSolverSStructCreate(
   char      *argv[],
   HYPRE_Int  argc,
   hypre_DriveSolver *krylov_ptr,
   hypre_DriveSolver *precond_ptr );

HYPRE_Int
hypre_DriveSolverSStructSetup(
   hypre_DriveSolver   solver,
   HYPRE_Int           precond_bool,
   HYPRE_Real          tol,
   HYPRE_Real          atol,
   HYPRE_Int           max_iter,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x );

HYPRE_Int
hypre_DriveSolverSStructSolve(
   hypre_DriveSolver   solver,
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x );

HYPRE_Int
hypre_DriveSolverSStructGetStats(
   hypre_DriveSolver  solver,
   HYPRE_Int         *num_iterations_ptr,
   HYPRE_Real        *final_res_norm_ptr );

HYPRE_Int
hypre_DriveSolverSStructDestroy(
   hypre_DriveSolver  krylov,
   hypre_DriveSolver  precond );

/* hypre_solver_parcsr.c */

HYPRE_Int
hypre_DriveSolverParCSRHelp();

HYPRE_Int
hypre_DriveSolverParCSRCreate(
   char      *argv[],
   HYPRE_Int  argc,
   hypre_DriveSolver *krylov,
   hypre_DriveSolver *precond );

HYPRE_Int
hypre_DriveSolverParCSRSetup(
   hypre_DriveSolver  solver,
   HYPRE_Int          precond_bool,
   HYPRE_Real         tol,
   HYPRE_Real         atol,
   HYPRE_Int          max_iter,
   HYPRE_ParCSRMatrix A,
   HYPRE_ParVector    b,
   HYPRE_ParVector    x );

HYPRE_Int
hypre_DriveSolverParCSRSolve(
   hypre_DriveSolver  solver,
   HYPRE_ParCSRMatrix A,
   HYPRE_ParVector    b,
   HYPRE_ParVector    x );

HYPRE_Int
hypre_DriveSolverParCSRGetStats(
   hypre_DriveSolver  solver,
   HYPRE_Int         *num_iterations_ptr,
   HYPRE_Real        *final_res_norm_ptr );

HYPRE_Int
hypre_DriveSolverParCSRDestroy(
   hypre_DriveSolver  solver );

#endif
