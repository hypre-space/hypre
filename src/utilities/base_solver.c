/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SolverSetup(HYPRE_Solver solver,
                  HYPRE_Matrix A,
                  HYPRE_Vector b,
                  HYPRE_Vector x)
{
   hypre_Solver *base = (hypre_Solver *) solver;

   if (!base || !hypre_SolverSetup(base))
   {
      return hypre_error_flag;
   }

   return hypre_SolverSetup(base)(solver, A, b, x);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SolverSolve(HYPRE_Solver solver,
                  HYPRE_Matrix A,
                  HYPRE_Vector b,
                  HYPRE_Vector x)
{
   hypre_Solver *base = (hypre_Solver *) solver;

   if (!base || !hypre_SolverSolve(base))
   {
      return hypre_error_flag;
   }

   return hypre_SolverSolve(base)(solver, A, b, x);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SolverDestroy(HYPRE_Solver solver)
{
   hypre_Solver *base = (hypre_Solver *) solver;

   if (!base || !hypre_SolverDestroy(base))
   {
      return hypre_error_flag;
   }

   return hypre_SolverDestroy(base)(solver);
}
