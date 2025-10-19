/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_LOBPCG_HEADER
#define HYPRE_LOBPCG_HEADER

#include "HYPRE_krylov.h"

#include "_hypre_fortran_matrix.h"
#include "_hypre_lobpcg_multivector.h"
#include "_hypre_lobpcg_interpreter.h"
#include "_hypre_lobpcg_temp_multivector.h"

#ifdef HYPRE_MIXED_PRECISION
#include "_hypre_krylov_mup_def.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   void*  (*MatvecCreate)     ( void *A, void *x );
   HYPRE_Int (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                                void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int (*MatvecDestroy) ( void *matvec_data );

   void*  (*MatMultiVecCreate)     ( void *A, void *x );
   HYPRE_Int (*MatMultiVec)        ( void *data, HYPRE_Complex alpha, void *A,
                                     void *x, HYPRE_Complex beta, void *y );
   HYPRE_Int (*MatMultiVecDestroy) ( void *data );

} HYPRE_MatvecFunctions;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup Eigensolvers Eigensolvers
 *
 * A basic interface for eigensolvers. These eigensolvers support many of the
 * matrix/vector storage schemes in hypre.  They should be used in conjunction
 * with the storage-specific interfaces.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name EigenSolvers
 *
 * @{
 **/

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name LOBPCG Eigensolver
 *
 * @{
 **/

/**
 * LOBPCG constructor.
 */
HYPRE_Int HYPRE_LOBPCGCreate(mv_InterfaceInterpreter *interpreter,
                             HYPRE_MatvecFunctions   *mvfunctions,
                             HYPRE_Solver            *solver);

/**
 * LOBPCG destructor.
 */
HYPRE_Int HYPRE_LOBPCGDestroy(HYPRE_Solver solver);

/**
 * (Optional) Set the preconditioner to use.  If not called, preconditioning is
 * not used.
 **/
HYPRE_Int HYPRE_LOBPCGSetPrecond(HYPRE_Solver         solver,
                                 HYPRE_PtrToSolverFcn precond,
                                 HYPRE_PtrToSolverFcn precond_setup,
                                 HYPRE_Solver         precond_solver);

/**
 **/
HYPRE_Int HYPRE_LOBPCGGetPrecond(HYPRE_Solver  solver,
                                 HYPRE_Solver *precond_data_ptr);

/**
 * Set up \e A and the preconditioner (if there is one).
 **/
HYPRE_Int HYPRE_LOBPCGSetup(HYPRE_Solver solver,
                            HYPRE_Matrix A,
                            HYPRE_Vector b,
                            HYPRE_Vector x);

/**
 * (Optional) Set up \e B.  If not called, B = I.
 **/
HYPRE_Int HYPRE_LOBPCGSetupB(HYPRE_Solver solver,
                             HYPRE_Matrix B,
                             HYPRE_Vector x);

/**
 * (Optional) Set the preconditioning to be applied to Tx = b, not Ax = b.
 **/
HYPRE_Int HYPRE_LOBPCGSetupT(HYPRE_Solver solver,
                             HYPRE_Matrix T,
                             HYPRE_Vector x);

/**
 * Solve A x = lambda B x, y'x = 0.
 **/
HYPRE_Int HYPRE_LOBPCGSolve(HYPRE_Solver       solver,
                            mv_MultiVectorPtr  y,
                            mv_MultiVectorPtr  x,
                            HYPRE_Real        *lambda );

/**
 * (Optional) Set the absolute convergence tolerance.
 **/
HYPRE_Int HYPRE_LOBPCGSetTol(HYPRE_Solver solver,
                             HYPRE_Real   tol);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
HYPRE_Int HYPRE_LOBPCGSetRTol(HYPRE_Solver solver,
                              HYPRE_Real   tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_LOBPCGSetMaxIter(HYPRE_Solver solver,
                                 HYPRE_Int          max_iter);

/**
 * Define which initial guess for inner PCG iterations to use: \e mode = 0:
 * use zero initial guess, otherwise use RHS.
 **/
HYPRE_Int HYPRE_LOBPCGSetPrecondUsageMode(HYPRE_Solver solver,
                                          HYPRE_Int          mode);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_LOBPCGSetPrintLevel(HYPRE_Solver solver,
                                    HYPRE_Int          level);

/* Returns the pointer to residual norms matrix (blockSize x 1) */
utilities_FortranMatrix*
HYPRE_LOBPCGResidualNorms(HYPRE_Solver solver);

/* Returns the pointer to residual norms history matrix (blockSize x maxIter) */
utilities_FortranMatrix*
HYPRE_LOBPCGResidualNormsHistory(HYPRE_Solver solver);

/* Returns the pointer to eigenvalue history matrix (blockSize x maxIter) */
utilities_FortranMatrix*
HYPRE_LOBPCGEigenvaluesHistory(HYPRE_Solver solver);

/* Returns the number of iterations performed by LOBPCG */
HYPRE_Int HYPRE_LOBPCGIterations(HYPRE_Solver solver);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#ifdef HYPRE_MIXED_PRECISION
/* The following is for user compiles and the order is important.  The first
 * header ensures that we do not change prototype names in user files or in the
 * second header file.  The second header contains all the prototypes needed by
 * users for mixed precision. */
#ifndef hypre_MP_BUILD
#include "_hypre_krylov_mup_undef.h"
#include "HYPRE_lobpcg_mup.h"
#endif
#endif

#endif
