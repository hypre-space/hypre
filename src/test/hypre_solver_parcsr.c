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

/*--------------------------------------------------------------------------
 * ParCSR Solver Options
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSolverParCSRHelp( )
{
   hypre_printf("SolverParCSROptions: [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSolverParCSRCreate(
   char      *argv[],
   HYPRE_Int  argc,
   hypre_DriveSolver *krylov,
   hypre_DriveSolver *precond )
{
   return 0;
}

HYPRE_Int
hypre_DriveSolverParCSRSetup(
   hypre_DriveSolver  solver,
   HYPRE_Int          precond_bool,
   HYPRE_Real         tol,
   HYPRE_Real         atol,
   HYPRE_Int          max_iter,
   HYPRE_ParCSRMatrix A,
   HYPRE_ParVector    b,
   HYPRE_ParVector    x )
{
   return 0;
}

HYPRE_Int
hypre_DriveSolverParCSRSolve(
   hypre_DriveSolver  solver,
   HYPRE_ParCSRMatrix A,
   HYPRE_ParVector    b,
   HYPRE_ParVector    x )
{
   return 0;
}

HYPRE_Int
hypre_DriveSolverParCSRGetStats(
   hypre_DriveSolver  solver,
   HYPRE_Int         *num_iterations_ptr,
   HYPRE_Real        *final_res_norm_ptr )
{
   return 0;
}

HYPRE_Int
hypre_DriveSolverParCSRDestroy(
   hypre_DriveSolver  solver )
{
   return 0;
}
