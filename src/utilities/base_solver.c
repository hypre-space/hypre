/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#if 0
typedef struct hypre_BaseSolverWrapper_struct
{
   hypre_Solver          base;
   HYPRE_Solver          inner;
   HYPRE_PtrToSolverFcn  inner_setup;
   HYPRE_PtrToSolverFcn  inner_solve;
   HYPRE_PtrToDestroyFcn inner_destroy;
} hypre_BaseSolverWrapper;

static HYPRE_Int
hypre_BaseSolverWrapperSetup(HYPRE_Solver solver,
                             HYPRE_Matrix A,
                             HYPRE_Vector b,
                             HYPRE_Vector x)
{
   hypre_BaseSolverWrapper *wrapper = (hypre_BaseSolverWrapper *) solver;

   if (!wrapper || !wrapper->inner_setup)
   {
      return hypre_error_flag;
   }

   return wrapper->inner_setup(wrapper->inner, A, b, x);
}

static HYPRE_Int
hypre_BaseSolverWrapperSolve(HYPRE_Solver solver,
                             HYPRE_Matrix A,
                             HYPRE_Vector b,
                             HYPRE_Vector x)
{
   hypre_BaseSolverWrapper *wrapper = (hypre_BaseSolverWrapper *) solver;

   if (!wrapper || !wrapper->inner_solve)
   {
      return hypre_error_flag;
   }

   return wrapper->inner_solve(wrapper->inner, A, b, x);
}

static HYPRE_Int
hypre_BaseSolverWrapperDestroy(HYPRE_Solver solver)
{
   hypre_BaseSolverWrapper *wrapper = (hypre_BaseSolverWrapper *) solver;

   if (!wrapper)
   {
      return hypre_error_flag;
   }

   if (wrapper->inner_destroy)
   {
      wrapper->inner_destroy(wrapper->inner);
      wrapper->inner = NULL;
   }

   hypre_TFree(wrapper, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SolverWrap(HYPRE_PtrToSolverFcn  setup,
                 HYPRE_PtrToSolverFcn  solve,
                 HYPRE_PtrToDestroyFcn destroy,
                 HYPRE_Solver          inner,
                 HYPRE_Solver         *wrapped)
{
   hypre_BaseSolverWrapper *wrapper = NULL;

   if (!wrapped)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   wrapper = hypre_CTAlloc(hypre_BaseSolverWrapper, 1, HYPRE_MEMORY_HOST);
   if (!wrapper)
   {
      return hypre_error_flag;
   }

   wrapper->base.setup   = hypre_BaseSolverWrapperSetup;
   wrapper->base.solve   = hypre_BaseSolverWrapperSolve;
   wrapper->base.destroy = hypre_BaseSolverWrapperDestroy;

   wrapper->inner         = inner;
   wrapper->inner_setup   = setup;
   wrapper->inner_solve   = solve;
   wrapper->inner_destroy = destroy;

   *wrapped = (HYPRE_Solver) wrapper;
   return hypre_error_flag;
}
#endif

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
