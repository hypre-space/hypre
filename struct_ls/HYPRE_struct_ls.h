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


#ifndef HYPRE_STRUCT_LS_HEADER
#define HYPRE_STRUCT_LS_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_struct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Solvers
 *
 * These solvers use matrix/vector storage schemes that are tailored
 * to structured grid problems.
 *
 * @memo Linear solvers for structured grids
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Solvers
 **/
/*@{*/

struct hypre_StructSolver_struct;
/**
 * The solver object.
 **/
typedef struct hypre_StructSolver_struct *HYPRE_StructSolver;

typedef int (*HYPRE_PtrToStructSolverFcn)(HYPRE_StructSolver,
                                          HYPRE_StructMatrix,
                                          HYPRE_StructVector,
                                          HYPRE_StructVector);

#ifndef HYPRE_MODIFYPC
#define HYPRE_MODIFYPC
/* if pc not defined, then may need HYPRE_SOLVER also */

 #ifndef HYPRE_SOLVER_STRUCT
 #define HYPRE_SOLVER_STRUCT
 struct hypre_Solver_struct;
 typedef struct hypre_Solver_struct *HYPRE_Solver;
 #endif

typedef int (*HYPRE_PtrToModifyPCFcn)(HYPRE_Solver,
                                      int,
                                      double);
#endif

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Jacobi Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructJacobiCreate(MPI_Comm            comm,
                             HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_StructJacobiDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_StructJacobiSetup(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * Solve the system.
 **/
int HYPRE_StructJacobiSolve(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_StructJacobiSetTol(HYPRE_StructSolver solver,
                             double             tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_StructJacobiSetMaxIter(HYPRE_StructSolver solver,
                                 int                max_iter);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
int HYPRE_StructJacobiSetZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
int HYPRE_StructJacobiSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_StructJacobiGetNumIterations(HYPRE_StructSolver  solver,
                                       int                *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_StructJacobiGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                   double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PFMG Solver
 *
 * PFMG is a semicoarsening multigrid solver that uses pointwise relaxation.
 * For periodic problems, users should try to set the grid size in periodic
 * dimensions to be as close to a power-of-two as possible.  That is, if the
 * grid size in a periodic dimension is given by $N = 2^m * M$ where $M$ is not
 * a power-of-two, then $M$ should be as small as possible.  Large values of $M$
 * will generally result in slower convergence rates.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructPFMGCreate(MPI_Comm            comm,
                           HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_StructPFMGDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_StructPFMGSetup(HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x);

/**
 * Solve the system.
 **/
int HYPRE_StructPFMGSolve(HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_StructPFMGSetTol(HYPRE_StructSolver solver,
                           double             tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_StructPFMGSetMaxIter(HYPRE_StructSolver solver,
                               int                max_iter);

/**
 * (Optional) Set maximum number of multigrid grid levels.
 **/
int HYPRE_StructPFMGSetMaxLevels(HYPRE_StructSolver solver, 
                                 int                max_levels);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_StructPFMGSetRelChange(HYPRE_StructSolver solver,
                                 int                rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
int HYPRE_StructPFMGSetZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
int HYPRE_StructPFMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Set relaxation type.
 *
 * Current relaxation methods set by {\tt relax\_type} are:
 *
 * \begin{tabular}{l@{ -- }l}
 * 0 & Jacobi \\
 * 1 & Weighted Jacobi (default) \\
 * 2 & Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation) \\
 * 3 & Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation) \\
 * \end{tabular}
 **/
int HYPRE_StructPFMGSetRelaxType(HYPRE_StructSolver solver,
                                 int                relax_type);

/*
 * (Optional) Set Jacobi weight (this is purposely not documented)
 */
int HYPRE_StructPFMGSetJacobiWeight(HYPRE_StructSolver solver,
                                    double             weight);
int HYPRE_StructPFMGGetJacobiWeight(HYPRE_StructSolver solver,
                                    double            *weight);


/**
 * (Optional) Set type of coarse-grid operator to use.
 *
 * Current operators set by {\tt rap\_type} are:
 *
 * \begin{tabular}{l@{ -- }l}
 * 0 & Galerkin (default) \\
 * 1 & non-Galerkin 5-pt or 7-pt stencils \\
 * \end{tabular}
 *
 * Both operators are constructed algebraically.  The non-Galerkin option
 * maintains a 5-pt stencil in 2D and a 7-pt stencil in 3D on all grid levels.
 * The stencil coefficients are computed by averaging techniques.
 **/
int HYPRE_StructPFMGSetRAPType(HYPRE_StructSolver solver,
                               int                rap_type);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
int HYPRE_StructPFMGSetNumPreRelax(HYPRE_StructSolver solver,
                                   int                num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
int HYPRE_StructPFMGSetNumPostRelax(HYPRE_StructSolver solver,
                                    int                num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.  This can
 * greatly improve efficiency by eliminating unnecessary relaxations when the
 * underlying problem is isotropic.
 **/
int HYPRE_StructPFMGSetSkipRelax(HYPRE_StructSolver solver,
                                 int                skip_relax);

/*
 * RE-VISIT
 **/
int HYPRE_StructPFMGSetDxyz(HYPRE_StructSolver  solver,
                            double             *dxyz);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_StructPFMGSetLogging(HYPRE_StructSolver solver,
                               int                logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_StructPFMGSetPrintLevel(HYPRE_StructSolver solver,
                                  int                print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_StructPFMGGetNumIterations(HYPRE_StructSolver  solver,
                                     int                *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_StructPFMGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                 double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct SMG Solver
 *
 * SMG is a semicoarsening multigrid solver that uses plane smoothing (in 3D).
 * The plane smoother calls a 2D SMG algorithm with line smoothing, and the line
 * smoother is cyclic reduction (1D SMG).  For periodic problems, the grid size
 * in periodic dimensions currently must be a power-of-two.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructSMGCreate(MPI_Comm            comm,
                          HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_StructSMGDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_StructSMGSetup(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/**
 * Solve the system.
 **/
int HYPRE_StructSMGSolve(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/*
 * RE-VISIT
 **/
int HYPRE_StructSMGSetMemoryUse(HYPRE_StructSolver solver,
                                int                memory_use);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_StructSMGSetTol(HYPRE_StructSolver solver,
                          double             tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_StructSMGSetMaxIter(HYPRE_StructSolver solver,
                              int                max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_StructSMGSetRelChange(HYPRE_StructSolver solver,
                                int                rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
int HYPRE_StructSMGSetZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
int HYPRE_StructSMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
int HYPRE_StructSMGSetNumPreRelax(HYPRE_StructSolver solver,
                                  int                num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
int HYPRE_StructSMGSetNumPostRelax(HYPRE_StructSolver solver,
                                   int                num_post_relax);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_StructSMGSetLogging(HYPRE_StructSolver solver,
                              int                logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_StructSMGSetPrintLevel(HYPRE_StructSolver solver,
                                  int                print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_StructSMGGetNumIterations(HYPRE_StructSolver  solver,
                                    int                *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_StructSMGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PCG Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{PCG Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructPCGCreate(MPI_Comm            comm,
                          HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_StructPCGDestroy(HYPRE_StructSolver solver);

int HYPRE_StructPCGSetup(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

int HYPRE_StructPCGSolve(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

int HYPRE_StructPCGSetTol(HYPRE_StructSolver solver,
                          double             tol);

int HYPRE_StructPCGSetAbsoluteTol(HYPRE_StructSolver solver,
                                  double             tol);

int HYPRE_StructPCGSetMaxIter(HYPRE_StructSolver solver,
                              int                max_iter);

int HYPRE_StructPCGSetTwoNorm(HYPRE_StructSolver solver,
                              int                two_norm);

int HYPRE_StructPCGSetRelChange(HYPRE_StructSolver solver,
                                int                rel_change);

int HYPRE_StructPCGSetPrecond(HYPRE_StructSolver         solver,
                              HYPRE_PtrToStructSolverFcn precond,
                              HYPRE_PtrToStructSolverFcn precond_setup,
                              HYPRE_StructSolver         precond_solver);

int HYPRE_StructPCGSetLogging(HYPRE_StructSolver solver,
                              int                logging);

int HYPRE_StructPCGSetPrintLevel(HYPRE_StructSolver solver,
                                 int                level);

int HYPRE_StructPCGGetNumIterations(HYPRE_StructSolver  solver,
                                    int                *num_iterations);

int HYPRE_StructPCGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                double             *norm);

int HYPRE_StructPCGGetResidual(HYPRE_StructSolver   solver,
                               void               **residual);

/**
 * Setup routine for diagonal preconditioning.
 **/
int HYPRE_StructDiagScaleSetup(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector y,
                               HYPRE_StructVector x);

/**
 * Solve routine for diagonal preconditioning.
 **/
int HYPRE_StructDiagScale(HYPRE_StructSolver solver,
                          HYPRE_StructMatrix HA,
                          HYPRE_StructVector Hy,
                          HYPRE_StructVector Hx);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{GMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructGMRESCreate(MPI_Comm            comm,
                            HYPRE_StructSolver *solver);


/**
 * Destroy a solver object.
 **/
int HYPRE_StructGMRESDestroy(HYPRE_StructSolver solver);

int HYPRE_StructGMRESSetup(HYPRE_StructSolver solver,
                           HYPRE_StructMatrix A,
                           HYPRE_StructVector b,
                           HYPRE_StructVector x);

int HYPRE_StructGMRESSolve(HYPRE_StructSolver solver,
                           HYPRE_StructMatrix A,
                           HYPRE_StructVector b,
                           HYPRE_StructVector x);

int HYPRE_StructGMRESSetTol(HYPRE_StructSolver solver,
                            double             tol);

int HYPRE_StructGMRESSetAbsoluteTol(HYPRE_StructSolver solver,
                                    double             tol);

int HYPRE_StructGMRESSetMaxIter(HYPRE_StructSolver solver,
                                int                max_iter);

int HYPRE_StructGMRESSetKDim(HYPRE_StructSolver solver,
                             int                k_dim);

int HYPRE_StructGMRESSetPrecond(HYPRE_StructSolver         solver,
                                HYPRE_PtrToStructSolverFcn precond,
                                HYPRE_PtrToStructSolverFcn precond_setup,
                                HYPRE_StructSolver         precond_solver);

int HYPRE_StructGMRESSetLogging(HYPRE_StructSolver solver,
                                int                logging);

int HYPRE_StructGMRESSetPrintLevel(HYPRE_StructSolver solver,
                                   int                level);

int HYPRE_StructGMRESGetNumIterations(HYPRE_StructSolver  solver,
                                      int                *num_iterations);

int HYPRE_StructGMRESGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                  double             *norm);

int HYPRE_StructGMRESGetResidual(HYPRE_StructSolver   solver,
                                 void               **residual);
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{FlexGMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructFlexGMRESCreate(MPI_Comm            comm,
                                HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_StructFlexGMRESDestroy(HYPRE_StructSolver solver);

int HYPRE_StructFlexGMRESSetup(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

int HYPRE_StructFlexGMRESSolve(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

int HYPRE_StructFlexGMRESSetTol(HYPRE_StructSolver solver,
                                double             tol);

int HYPRE_StructFlexGMRESSetAbsoluteTol(HYPRE_StructSolver solver,
                                        double             tol);

int HYPRE_StructFlexGMRESSetMaxIter(HYPRE_StructSolver solver,
                                    int                max_iter);

int HYPRE_StructFlexGMRESSetKDim(HYPRE_StructSolver solver,
                                 int                k_dim);

int HYPRE_StructFlexGMRESSetPrecond(HYPRE_StructSolver         solver,
                                    HYPRE_PtrToStructSolverFcn precond,
                                    HYPRE_PtrToStructSolverFcn precond_setup,
                                    HYPRE_StructSolver         precond_solver);

int HYPRE_StructFlexGMRESSetLogging(HYPRE_StructSolver solver,
                                    int                logging);

int HYPRE_StructFlexGMRESSetPrintLevel(HYPRE_StructSolver solver,
                                       int                level);

int HYPRE_StructFlexGMRESGetNumIterations(HYPRE_StructSolver  solver,
                                          int                *num_iterations);

int HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                      double             *norm);

int HYPRE_StructFlexGMRESGetResidual(HYPRE_StructSolver   solver,
                                     void               **residual);

int HYPRE_StructFlexGMRESSetModifyPC(HYPRE_StructSolver     solver,
                                     HYPRE_PtrToModifyPCFcn modify_pc);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{LGMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructLGMRESCreate(MPI_Comm            comm,
                             HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_StructLGMRESDestroy(HYPRE_StructSolver solver);

int HYPRE_StructLGMRESSetup(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

int HYPRE_StructLGMRESSolve(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

int HYPRE_StructLGMRESSetTol(HYPRE_StructSolver solver,
                             double             tol);

int HYPRE_StructLGMRESSetAbsoluteTol(HYPRE_StructSolver solver,
                                     double             tol);
   
int HYPRE_StructLGMRESSetMaxIter(HYPRE_StructSolver solver,
                                 int                max_iter);

int HYPRE_StructLGMRESSetKDim(HYPRE_StructSolver solver,
                              int                k_dim);

int HYPRE_StructLGMRESSetAugDim(HYPRE_StructSolver solver,
                                int                aug_dim);

int HYPRE_StructLGMRESSetPrecond(HYPRE_StructSolver         solver,
                                 HYPRE_PtrToStructSolverFcn precond,
                                 HYPRE_PtrToStructSolverFcn precond_setup,
                                 HYPRE_StructSolver         precond_solver);

int HYPRE_StructLGMRESSetLogging(HYPRE_StructSolver solver,
                                 int                logging);

int HYPRE_StructLGMRESSetPrintLevel(HYPRE_StructSolver solver,
                                    int                level);

int HYPRE_StructLGMRESGetNumIterations(HYPRE_StructSolver  solver,
                                       int                *num_iterations);

int HYPRE_StructLGMRESGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                   double             *norm);

int HYPRE_StructLGMRESGetResidual(HYPRE_StructSolver   solver,
                                  void               **residual);
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{BiCGSTAB Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructBiCGSTABCreate(MPI_Comm            comm,
                               HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_StructBiCGSTABDestroy(HYPRE_StructSolver solver);

int HYPRE_StructBiCGSTABSetup(HYPRE_StructSolver solver,
                              HYPRE_StructMatrix A,
                              HYPRE_StructVector b,
                              HYPRE_StructVector x);

int HYPRE_StructBiCGSTABSolve(HYPRE_StructSolver solver,
                              HYPRE_StructMatrix A,
                              HYPRE_StructVector b,
                              HYPRE_StructVector x);

int HYPRE_StructBiCGSTABSetTol(HYPRE_StructSolver solver,
                               double             tol);

int HYPRE_StructBiCGSTABSetAbsoluteTol(HYPRE_StructSolver solver,
                                       double             tol);

int HYPRE_StructBiCGSTABSetMaxIter(HYPRE_StructSolver solver,
                                   int                max_iter);

int HYPRE_StructBiCGSTABSetPrecond(HYPRE_StructSolver         solver,
                                   HYPRE_PtrToStructSolverFcn precond,
                                   HYPRE_PtrToStructSolverFcn precond_setup,
                                   HYPRE_StructSolver         precond_solver);

int HYPRE_StructBiCGSTABSetLogging(HYPRE_StructSolver solver,
                                   int                logging);

int HYPRE_StructBiCGSTABSetPrintLevel(HYPRE_StructSolver solver,
                                      int                level);

int HYPRE_StructBiCGSTABGetNumIterations(HYPRE_StructSolver  solver,
                                         int                *num_iterations);

int HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                     double             *norm);

int HYPRE_StructBiCGSTABGetResidual( HYPRE_StructSolver   solver,
                                     void               **residual);
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Hybrid Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_StructHybridCreate(MPI_Comm            comm,
                             HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_StructHybridDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_StructHybridSetup(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * Solve the system.
 **/
int HYPRE_StructHybridSolve(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_StructHybridSetTol(HYPRE_StructSolver solver,
                             double             tol);

/**
 * (Optional) Set an accepted convergence tolerance for diagonal scaling (DS).
 * The solver will switch preconditioners if the convergence of DS is slower
 * than {\tt cf\_tol}.
 **/
int HYPRE_StructHybridSetConvergenceTol(HYPRE_StructSolver solver,
                                        double             cf_tol);

/**
 * (Optional) Set maximum number of iterations for diagonal scaling (DS).  The
 * solver will switch preconditioners if DS reaches {\tt ds\_max\_its}.
 **/
int HYPRE_StructHybridSetDSCGMaxIter(HYPRE_StructSolver solver,
                                     int                ds_max_its);

/**
 * (Optional) Set maximum number of iterations for general preconditioner (PRE).
 * The solver will stop if PRE reaches {\tt pre\_max\_its}.
 **/
int HYPRE_StructHybridSetPCGMaxIter(HYPRE_StructSolver solver,
                                    int                pre_max_its);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
int HYPRE_StructHybridSetTwoNorm(HYPRE_StructSolver solver,
                                 int                two_norm);

int HYPRE_StructHybridSetStopCrit(HYPRE_StructSolver solver,
                                 int                stop_crit);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_StructHybridSetRelChange(HYPRE_StructSolver solver,
                                   int                rel_change);

/**
 * (Optional) Set the type of Krylov solver to use.
 *
 * Current krylov methods set by {\tt solver\_type} are:
 *
 * \begin{tabular}{l@{ -- }l}
 * 0 & PCG (default) \\
 * 1 & GMRES \\
 * 2 & BiCGSTAB \\
 * \end{tabular}
 **/
int HYPRE_StructHybridSetSolverType(HYPRE_StructSolver solver,
                                    int                solver_type);

/**
 * (Optional) Set the maximum size of the Krylov space when using GMRES.
 **/
int HYPRE_StructHybridSetKDim(HYPRE_StructSolver solver,
                              int                k_dim);

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_StructHybridSetPrecond(HYPRE_StructSolver         solver,
                                 HYPRE_PtrToStructSolverFcn precond,
                                 HYPRE_PtrToStructSolverFcn precond_setup,
                                 HYPRE_StructSolver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_StructHybridSetLogging(HYPRE_StructSolver solver,
                                 int                logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_StructHybridSetPrintLevel(HYPRE_StructSolver solver,
                                    int                print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_StructHybridGetNumIterations(HYPRE_StructSolver  solver,
                                       int                *num_its);

/**
 * Return the number of diagonal scaling iterations taken.
 **/
int HYPRE_StructHybridGetDSCGNumIterations(HYPRE_StructSolver  solver,
                                           int                *ds_num_its);

/**
 * Return the number of general preconditioning iterations taken.
 **/
int HYPRE_StructHybridGetPCGNumIterations(HYPRE_StructSolver  solver,
                                          int                *pre_num_its);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_StructHybridGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                   double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name Struct SparseMSG Solver
 **/

int HYPRE_StructSparseMSGCreate(MPI_Comm            comm,
                                HYPRE_StructSolver *solver);

int HYPRE_StructSparseMSGDestroy(HYPRE_StructSolver solver);

int HYPRE_StructSparseMSGSetup(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

int HYPRE_StructSparseMSGSolve(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

int HYPRE_StructSparseMSGSetTol(HYPRE_StructSolver solver,
                                double             tol);

int HYPRE_StructSparseMSGSetMaxIter(HYPRE_StructSolver solver,
                                    int                max_iter);

int HYPRE_StructSparseMSGSetJump(HYPRE_StructSolver solver,
                                 int                jump);

int HYPRE_StructSparseMSGSetRelChange(HYPRE_StructSolver solver,
                                      int                rel_change);

int HYPRE_StructSparseMSGSetZeroGuess(HYPRE_StructSolver solver);

int HYPRE_StructSparseMSGSetNonZeroGuess(HYPRE_StructSolver solver);

int HYPRE_StructSparseMSGSetRelaxType(HYPRE_StructSolver solver,
                                      int                relax_type);

int HYPRE_StructSparseMSGSetJacobiWeight(HYPRE_StructSolver solver,
                                         double             weight);

int HYPRE_StructSparseMSGSetNumPreRelax(HYPRE_StructSolver solver,
                                        int                num_pre_relax);

int HYPRE_StructSparseMSGSetNumPostRelax(HYPRE_StructSolver solver,
                                         int                num_post_relax);

int HYPRE_StructSparseMSGSetNumFineRelax(HYPRE_StructSolver solver,
                                         int                num_fine_relax);

int HYPRE_StructSparseMSGSetLogging(HYPRE_StructSolver solver,
                                    int                logging);

int HYPRE_StructSparseMSGSetPrintLevel(HYPRE_StructSolver solver,
                                          int         print_level);


int HYPRE_StructSparseMSGGetNumIterations(HYPRE_StructSolver  solver,
                                          int                *num_iterations);

int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                      double             *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* These includes shouldn't be here. (RDF) */
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "_hypre_struct_mv.h"

/**
 * @name Struct LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{LOBPCG Eigensolver}.
 **/
/*@{*/

/**
 * Load interface interpreter. Vector part loaded with hypre_StructKrylov
 * functions and multivector part loaded with mv_TempMultiVector functions.
 **/
int
HYPRE_StructSetupInterpreter(mv_InterfaceInterpreter *i);

/**
 * Load Matvec interpreter with hypre_StructKrylov functions.
 **/
int
HYPRE_StructSetupMatvec(HYPRE_MatvecFunctions *mv);

/* The next routines should not be here (lower-case prefix). (RDF) */

/*
 * Set hypre_StructPVector to random values.
 **/
int
hypre_StructVectorSetRandomValues(hypre_StructVector *vector, int seed);

/*
 * Same as hypre_StructVectorSetRandomValues except uses void pointer.
 **/
int
hypre_StructSetRandomValues(void *v, int seed);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*@}*/

#ifdef __cplusplus
}
#endif

#endif

