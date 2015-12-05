/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.1 $
 ***********************************************************************EHEADER*/

#ifndef HYPRE_KRYLOV_HEADER
#define HYPRE_KRYLOV_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Krylov Solvers
 *
 * These solvers support many of the matrix/vector storage schemes in hypre.
 * They should be used in conjunction with the storage-specific interfaces,
 * particularly the specific Create() and Destroy() functions.
 *
 * @memo A basic interface for Krylov solvers
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Krylov Solvers
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

typedef int (*HYPRE_PtrToSolverFcn)(HYPRE_Solver,
                                    HYPRE_Matrix,
                                    HYPRE_Vector,
                                    HYPRE_Vector);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name PCG Solver
 **/
/*@{*/

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_PCGSetup(HYPRE_Solver solver,
                   HYPRE_Matrix A,
                   HYPRE_Vector b,
                   HYPRE_Vector x);

/**
 * Solve the system.
 **/
int HYPRE_PCGSolve(HYPRE_Solver solver,
                   HYPRE_Matrix A,
                   HYPRE_Vector b,
                   HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_PCGSetTol(HYPRE_Solver solver,
                    double       tol);

/*
 * RE-VISIT
 **/
int HYPRE_PCGSetAbsoluteTolFactor(HYPRE_Solver solver, double abstolf);

/*
 * RE-VISIT
 **/
int HYPRE_PCGSetConvergenceFactorTol(HYPRE_Solver solver, double cf_tol);

/*
 * RE-VISIT
 **/
int HYPRE_PCGSetStopCrit(HYPRE_Solver solver, int stop_crit);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_PCGSetMaxIter(HYPRE_Solver solver,
                        int          max_iter);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
int HYPRE_PCGSetTwoNorm(HYPRE_Solver solver,
                        int          two_norm);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_PCGSetRelChange(HYPRE_Solver solver,
                          int          rel_change);

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_PCGSetPrecond(HYPRE_Solver         solver,
                        HYPRE_PtrToSolverFcn precond,
                        HYPRE_PtrToSolverFcn precond_setup,
                        HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_PCGSetLogging(HYPRE_Solver solver,
                        int          logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_PCGSetPrintLevel(HYPRE_Solver solver,
                           int          level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_PCGGetNumIterations(HYPRE_Solver  solver,
                              int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_PCGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                          double       *norm);

/**
 * Return the residual.
 **/
int HYPRE_PCGGetResidual(HYPRE_Solver  solver,
                         void        **residual);

/**
 **/
int HYPRE_PCGGetTol(HYPRE_Solver solver, double *tol);

/*
 * RE-VISIT
 **/
int HYPRE_PCGGetAbsoluteTolFactor(HYPRE_Solver solver, double *abstolf);

/*
 * RE-VISIT
 **/
int HYPRE_PCGGetConvergenceFactorTol(HYPRE_Solver solver, double *cf_tol);

/*
 * RE-VISIT
 **/
int HYPRE_PCGGetStopCrit(HYPRE_Solver solver, int *stop_crit);

/**
 **/
int HYPRE_PCGGetMaxIter(HYPRE_Solver solver, int *max_iter);

/**
 **/
int HYPRE_PCGGetTwoNorm(HYPRE_Solver solver, int *two_norm);

/**
 **/
int HYPRE_PCGGetRelChange(HYPRE_Solver solver, int *rel_change);

/**
 **/
int HYPRE_PCGGetPrecond(HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr);

/**
 **/
int HYPRE_PCGGetLogging(HYPRE_Solver solver, int *level);

/**
 **/
int HYPRE_PCGGetPrintLevel(HYPRE_Solver solver, int *level);

/**
 **/
int HYPRE_PCGGetConverged(HYPRE_Solver solver, int *converged);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name GMRES Solver
 **/
/*@{*/

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int 
HYPRE_GMRESSetup(HYPRE_Solver solver,
                 HYPRE_Matrix A,
                 HYPRE_Vector b,
                 HYPRE_Vector x);


/**
 * Solve the system.
 **/
int 
HYPRE_GMRESSolve(HYPRE_Solver solver,
                 HYPRE_Matrix A,
                 HYPRE_Vector b,
                 HYPRE_Vector x);


/**
 * (Optional) Set the convergence tolerance.
 **/
int
HYPRE_GMRESSetTol(HYPRE_Solver solver,
                  double       tol);

/*
 * RE-VISIT
 **/
int HYPRE_GMRESSetConvergenceFactorTol(HYPRE_Solver solver, double cf_tol);

/*
 * RE-VISIT
 **/
int HYPRE_GMRESSetStopCrit(HYPRE_Solver solver, int stop_crit);

/*
 * RE-VISIT
 **/
int HYPRE_GMRESSetMinIter(HYPRE_Solver solver, int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int
HYPRE_GMRESSetMaxIter(HYPRE_Solver solver,
                      int          max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
int HYPRE_GMRESSetKDim(HYPRE_Solver solver,
                       int          k_dim);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_GMRESSetRelChange(HYPRE_Solver solver, int rel_change);

/**
 * (Optional) Set the preconditioner to use.
 **/
int
HYPRE_GMRESSetPrecond(HYPRE_Solver         solver,
                      HYPRE_PtrToSolverFcn precond,
                      HYPRE_PtrToSolverFcn precond_setup,
                      HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int
HYPRE_GMRESSetLogging(HYPRE_Solver solver,
                      int          logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int
HYPRE_GMRESSetPrintLevel(HYPRE_Solver solver,
                         int          level);

/**
 * Return the number of iterations taken.
 **/
int
HYPRE_GMRESGetNumIterations(HYPRE_Solver  solver,
                            int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int
HYPRE_GMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                        double       *norm);

/**
 * Return the residual.
 **/
int
HYPRE_GMRESGetResidual(HYPRE_Solver   solver,
                       void         **residual);

/**
 **/
int HYPRE_GMRESGetTol(HYPRE_Solver solver, double *tol);

/*
 * RE-VISIT
 **/
int HYPRE_GMRESGetConvergenceFactorTol(HYPRE_Solver solver, double *cf_tol);

/*
 * RE-VISIT
 **/
int HYPRE_GMRESGetStopCrit(HYPRE_Solver solver, int *stop_crit);

/*
 * RE-VISIT
 **/
int HYPRE_GMRESGetMinIter(HYPRE_Solver solver, int *min_iter);

/**
 **/
int HYPRE_GMRESGetMaxIter(HYPRE_Solver solver, int *max_iter);

/**
 **/
int HYPRE_GMRESGetKDim(HYPRE_Solver solver, int *k_dim);

/**
 **/
int HYPRE_GMRESGetRelChange(HYPRE_Solver solver, int *rel_change);

/**
 **/
int HYPRE_GMRESGetPrecond(HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr);

/**
 **/
int HYPRE_GMRESGetLogging(HYPRE_Solver solver, int *level);

/**
 **/
int HYPRE_GMRESGetPrintLevel(HYPRE_Solver solver, int *level);

/**
 **/
int HYPRE_GMRESGetConverged(HYPRE_Solver solver, int *converged);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name BiCGSTAB Solver
 **/
/*@{*/

/*
 * RE-VISIT
 **/
int HYPRE_BiCGSTABDestroy(HYPRE_Solver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int 
HYPRE_BiCGSTABSetup(HYPRE_Solver solver,
                    HYPRE_Matrix A,
                    HYPRE_Vector b,
                    HYPRE_Vector x);

/**
 * Solve the system.
 **/
int 
HYPRE_BiCGSTABSolve(HYPRE_Solver solver,
                    HYPRE_Matrix A,
                    HYPRE_Vector b,
                    HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int
HYPRE_BiCGSTABSetTol(HYPRE_Solver solver,
                     double       tol);

/*
 * RE-VISIT
 **/
int HYPRE_BiCGSTABSetConvergenceFactorTol(HYPRE_Solver solver, double cf_tol);

/*
 * RE-VISIT
 **/
int HYPRE_BiCGSTABSetStopCrit(HYPRE_Solver solver, int stop_crit);

/*
 * RE-VISIT
 **/
int HYPRE_BiCGSTABSetMinIter(HYPRE_Solver solver, int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int
HYPRE_BiCGSTABSetMaxIter(HYPRE_Solver solver,
                         int          max_iter);

/**
 * (Optional) Set the preconditioner to use.
 **/
int
HYPRE_BiCGSTABSetPrecond(HYPRE_Solver         solver,
                         HYPRE_PtrToSolverFcn precond,
                         HYPRE_PtrToSolverFcn precond_setup,
                         HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int
HYPRE_BiCGSTABSetLogging(HYPRE_Solver solver,
                         int          logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int
HYPRE_BiCGSTABSetPrintLevel(HYPRE_Solver solver,
                            int          level);

/**
 * Return the number of iterations taken.
 **/
int
HYPRE_BiCGSTABGetNumIterations(HYPRE_Solver  solver,
                               int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int
HYPRE_BiCGSTABGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                           double       *norm);

/**
 * Return the residual.
 **/
int
HYPRE_BiCGSTABGetResidual(HYPRE_Solver  solver,
                          void        **residual);

/**
 **/
int HYPRE_BiCGSTABGetPrecond(HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name CGNR Solver
 **/
/*@{*/

/*
 * RE-VISIT
 **/
int HYPRE_CGNRDestroy(HYPRE_Solver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int 
HYPRE_CGNRSetup(HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x);

/**
 * Solve the system.
 **/
int 
HYPRE_CGNRSolve(HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int
HYPRE_CGNRSetTol(HYPRE_Solver solver,
                 double       tol);

/*
 * RE-VISIT
 **/
int HYPRE_CGNRSetStopCrit(HYPRE_Solver solver, int stop_crit);

/*
 * RE-VISIT
 **/
int HYPRE_CGNRSetMinIter(HYPRE_Solver solver, int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int
HYPRE_CGNRSetMaxIter(HYPRE_Solver solver,
                     int          max_iter);

/**
 * (Optional) Set the preconditioner to use.
 **/
int
HYPRE_CGNRSetPrecond(HYPRE_Solver         solver,
                     HYPRE_PtrToSolverFcn precond,
                     HYPRE_PtrToSolverFcn precondT,
                     HYPRE_PtrToSolverFcn precond_setup,
                     HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int
HYPRE_CGNRSetLogging(HYPRE_Solver solver,
                     int          logging);

#if 0 /* need to add */
/*
 * (Optional) Set the amount of printing to do to the screen.
 **/
int
HYPRE_CGNRSetPrintLevel(HYPRE_Solver solver,
                        int          level);
#endif

/**
 * Return the number of iterations taken.
 **/
int
HYPRE_CGNRGetNumIterations(HYPRE_Solver  solver,
                           int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int
HYPRE_CGNRGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                       double       *norm);

#if 0 /* need to add */
/*
 * Return the residual.
 **/
int
HYPRE_CGNRGetResidual(HYPRE_Solver  solver,
                      void        **residual);
#endif

/**
 **/
int HYPRE_CGNRGetPrecond(HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*@}*/

#ifdef __cplusplus
}
#endif

#endif
