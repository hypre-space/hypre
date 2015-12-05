/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/


#ifndef hypre_LOBPCG_SOLVER
#define hypre_LOBPCG_SOLVER

#include "HYPRE_krylov.h"

#include "fortran_matrix.h"
#include "multivector.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Eigensolvers
 *
 * These eigensolvers support many of the matrix/vector storage schemes in
 * hypre.  They should be used in conjunction with the storage-specific
 * interfaces.
 *
 * @memo A basic interface for eigensolvers
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name EigenSolvers
 **/
/*@{*/

#ifndef HYPRE_SOLVER_STRUCT
#define HYPRE_SOLVER_STRUCT
   struct hypre_Solver_struct;
/**
 * The solver object.
 **/
   typedef struct hypre_Solver_struct *HYPRE_Solver;
#endif

#ifndef HYPRE_MATRIX_STRUCT
#define HYPRE_MATRIX_STRUCT
   struct hypre_Matrix_struct;
/**
 * The matrix object.
 **/
   typedef struct hypre_Matrix_struct *HYPRE_Matrix;
#endif

#ifndef HYPRE_VECTOR_STRUCT
#define HYPRE_VECTOR_STRUCT
   struct hypre_Vector_struct;
/**
 * The vector object.
 **/
   typedef struct hypre_Vector_struct *HYPRE_Vector;
#endif

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name LOBPCG Eigensolver
 **/
/*@{*/

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
 * Set up {\tt A} and the preconditioner (if there is one).
 **/
HYPRE_Int HYPRE_LOBPCGSetup(HYPRE_Solver solver, 
                      HYPRE_Matrix A,
                      HYPRE_Vector b,
                      HYPRE_Vector x);

/**
 * (Optional) Set up {\tt B}.  If not called, B = I.
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
                      double            *lambda );

/**
 * (Optional) Set the absolute convergence tolerance.
 **/
HYPRE_Int HYPRE_LOBPCGSetTol(HYPRE_Solver solver,
                       double       tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_LOBPCGSetMaxIter(HYPRE_Solver solver,
                           HYPRE_Int          max_iter);

/**
 * Define which initial guess for inner PCG iterations to use: {\tt mode} = 0:
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

void hypre_LOBPCGMultiOperatorB(void *data,
                                void *x,
                                void *y);

void lobpcg_MultiVectorByMultiVector(mv_MultiVectorPtr        x,
                                     mv_MultiVectorPtr        y,
                                     utilities_FortranMatrix *xy);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*@}*/

#ifdef __cplusplus
}
#endif

#endif
