/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_BASE_HEADER
#define HYPRE_BASE_HEADER

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

} hypre_Solver;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_Solver structure
 *--------------------------------------------------------------------------*/

#define hypre_SolverSetup(data)       ((data) -> setup)
#define hypre_SolverSolve(data)       ((data) -> solve)
#define hypre_SolverDestroy(data)     ((data) -> destroy)

#endif /* HYPRE_BASE_HEADER */
