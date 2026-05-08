/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_BASE_SOLVER_HEADER
#define HYPRE_BASE_SOLVER_HEADER

/******************************************************************************
 *
 * Base private solver struct
 *
 *****************************************************************************/

typedef struct
{
   HYPRE_PtrToSolverFcn   setup;
   HYPRE_PtrToSolverFcn   solve;
   HYPRE_PtrToDestroyFcn  destroy;

   /* Common parameters */
   HYPRE_Int              is_setup; /* 1 after a successful Setup call */
} hypre_Solver;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Solver structure
 *--------------------------------------------------------------------------*/

#define hypre_SolverSetup(data)          ((data) -> setup)
#define hypre_SolverSolve(data)          ((data) -> solve)
#define hypre_SolverDestroy(data)        ((data) -> destroy)
#define hypre_SolverSetupIsDone(data)    ((data) -> is_setup)
#define hypre_SolverSetIsSetup(data)     ((data) -> is_setup = 1)
#define hypre_SolverResetIsSetup(data)   ((data) -> is_setup = 0)

#endif /* HYPRE_BASE_SOLVER_HEADER */
