/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_BASE_HEADER
#define HYPRE_BASE_HEADER

/* Base public solver struct */
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;

/* Base public matrix struct */
struct hypre_Matrix_struct;
typedef struct hypre_Matrix_struct *HYPRE_Matrix;

/* Base public vector struct */
struct hypre_Vector_struct;
typedef struct hypre_Vector_struct *HYPRE_Vector;

/* Base function pointers */
typedef HYPRE_Int (*HYPRE_PtrToSolverFcn)(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector, HYPRE_Vector);
typedef HYPRE_Int (*HYPRE_PtrToDestroyFcn)(HYPRE_Solver);

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
