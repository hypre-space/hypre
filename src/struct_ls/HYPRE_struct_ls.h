/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_STRUCT_LS_HEADER
#define HYPRE_STRUCT_LS_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_struct_mv.h"
#include "HYPRE_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup StructSolvers Struct Solvers
 *
 * Linear solvers for structured grids. These solvers use matrix/vector storage
 * schemes that are tailored to structured grid problems.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Solvers
 *
 * @{
 **/

struct hypre_StructSolver_struct;
/**
 * The solver object.
 **/
typedef struct hypre_StructSolver_struct *HYPRE_StructSolver;

typedef HYPRE_Int (*HYPRE_PtrToStructSolverFcn)(HYPRE_StructSolver,
                                                HYPRE_StructMatrix,
                                                HYPRE_StructVector,
                                                HYPRE_StructVector);

#ifndef HYPRE_MODIFYPC
#define HYPRE_MODIFYPC
/* if pc not defined, then may need HYPRE_SOLVER also */

typedef HYPRE_Int (*HYPRE_PtrToModifyPCFcn)(HYPRE_Solver,
                                            HYPRE_Int,
                                            HYPRE_Real);
#endif

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Jacobi Solver
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructJacobiCreate(MPI_Comm            comm,
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
HYPRE_Int HYPRE_StructJacobiDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_StructJacobiSetup(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_StructJacobiSolve(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_StructJacobiSetTol(HYPRE_StructSolver solver,
                                   HYPRE_Real         tol);


HYPRE_Int HYPRE_StructJacobiGetTol(HYPRE_StructSolver solver,
                                   HYPRE_Real *tol );

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_StructJacobiSetMaxIter(HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructJacobiGetMaxIter(HYPRE_StructSolver solver,
                                       HYPRE_Int *max_iter );

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
HYPRE_Int HYPRE_StructJacobiSetZeroGuess(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructJacobiGetZeroGuess(HYPRE_StructSolver solver,
                                         HYPRE_Int *zeroguess );

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
HYPRE_Int HYPRE_StructJacobiSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_StructJacobiGetNumIterations(HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_StructJacobiGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                         HYPRE_Real         *norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PFMG Solver
 *
 * PFMG is a semicoarsening multigrid solver that uses pointwise relaxation.
 * For periodic problems, users should try to set the grid size in periodic
 * dimensions to be as close to a power-of-two as possible.  That is, if the
 * grid size in a periodic dimension is given by \f$N = 2^m * M\f$ where \f$M\f$
 * is not a power-of-two, then \f$M\f$ should be as small as possible.  Large
 * values of \f$M\f$ will generally result in slower convergence rates.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructPFMGCreate(MPI_Comm            comm,
                                 HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructPFMGDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_StructPFMGSetup(HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_StructPFMGSolve(HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_StructPFMGSetTol(HYPRE_StructSolver solver,
                                 HYPRE_Real         tol);

HYPRE_Int HYPRE_StructPFMGGetTol (HYPRE_StructSolver solver,
                                  HYPRE_Real *tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_StructPFMGSetMaxIter(HYPRE_StructSolver solver,
                                     HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructPFMGGetMaxIter(HYPRE_StructSolver solver,
                                     HYPRE_Int *max_iter);

/**
 * (Optional) Set maximum number of multigrid grid levels.
 **/
HYPRE_Int HYPRE_StructPFMGSetMaxLevels(HYPRE_StructSolver solver,
                                       HYPRE_Int          max_levels);

HYPRE_Int HYPRE_StructPFMGGetMaxLevels (HYPRE_StructSolver solver,
                                        HYPRE_Int *max_levels );

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int HYPRE_StructPFMGSetRelChange(HYPRE_StructSolver solver,
                                       HYPRE_Int          rel_change);

HYPRE_Int HYPRE_StructPFMGGetRelChange (HYPRE_StructSolver solver,
                                        HYPRE_Int *rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
HYPRE_Int HYPRE_StructPFMGSetZeroGuess(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructPFMGGetZeroGuess(HYPRE_StructSolver solver,
                                       HYPRE_Int *zeroguess);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
HYPRE_Int HYPRE_StructPFMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Set relaxation type.
 *
 * Current relaxation methods set by \e relax_type are:
 *
 *    - 0 : Jacobi
 *    - 1 : Weighted Jacobi (default)
 *    - 2 : Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation)
 *    - 3 : Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation)
 **/
HYPRE_Int HYPRE_StructPFMGSetRelaxType(HYPRE_StructSolver solver,
                                       HYPRE_Int          relax_type);

HYPRE_Int HYPRE_StructPFMGGetRelaxType(HYPRE_StructSolver solver,
                                       HYPRE_Int *relax_type);

/*
 * (Optional) Set Jacobi weight (this is purposely not documented)
 */
HYPRE_Int HYPRE_StructPFMGSetJacobiWeight(HYPRE_StructSolver solver,
                                          HYPRE_Real         weight);
HYPRE_Int HYPRE_StructPFMGGetJacobiWeight(HYPRE_StructSolver solver,
                                          HYPRE_Real        *weight);


/**
 * (Optional) Set type of coarse-grid operator to use.
 *
 * Current operators set by \e rap_type are:
 *
 *    - 0 : Galerkin (default)
 *    - 1 : non-Galerkin 5-pt or 7-pt stencils
 *
 * Both operators are constructed algebraically.  The non-Galerkin option
 * maintains a 5-pt stencil in 2D and a 7-pt stencil in 3D on all grid levels.
 * The stencil coefficients are computed by averaging techniques.
 **/
HYPRE_Int HYPRE_StructPFMGSetRAPType(HYPRE_StructSolver solver,
                                     HYPRE_Int          rap_type);

HYPRE_Int HYPRE_StructPFMGGetRAPType(HYPRE_StructSolver solver,
                                     HYPRE_Int *rap_type );

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
HYPRE_Int HYPRE_StructPFMGSetNumPreRelax(HYPRE_StructSolver solver,
                                         HYPRE_Int          num_pre_relax);

HYPRE_Int HYPRE_StructPFMGGetNumPreRelax(HYPRE_StructSolver solver,
                                         HYPRE_Int *num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
HYPRE_Int HYPRE_StructPFMGSetNumPostRelax(HYPRE_StructSolver solver,
                                          HYPRE_Int          num_post_relax);

HYPRE_Int HYPRE_StructPFMGGetNumPostRelax(HYPRE_StructSolver solver,
                                          HYPRE_Int *num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.  This can
 * greatly improve efficiency by eliminating unnecessary relaxations when the
 * underlying problem is isotropic.
 **/
HYPRE_Int HYPRE_StructPFMGSetSkipRelax(HYPRE_StructSolver solver,
                                       HYPRE_Int          skip_relax);

HYPRE_Int HYPRE_StructPFMGGetSkipRelax(HYPRE_StructSolver solver,
                                       HYPRE_Int *skip_relax);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_StructPFMGSetDxyz(HYPRE_StructSolver  solver,
                                  HYPRE_Real         *dxyz);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_StructPFMGSetLogging(HYPRE_StructSolver solver,
                                     HYPRE_Int          logging);

HYPRE_Int HYPRE_StructPFMGGetLogging(HYPRE_StructSolver solver,
                                     HYPRE_Int *logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_StructPFMGSetPrintLevel(HYPRE_StructSolver solver,
                                        HYPRE_Int          print_level);

HYPRE_Int HYPRE_StructPFMGGetPrintLevel(HYPRE_StructSolver solver,
                                        HYPRE_Int *print_level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_StructPFMGGetNumIterations(HYPRE_StructSolver  solver,
                                           HYPRE_Int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_StructPFMGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                       HYPRE_Real         *norm);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
HYPRE_Int
HYPRE_StructPFMGSetDeviceLevel( HYPRE_StructSolver  solver,
                                HYPRE_Int   device_level  );
#endif
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct SMG Solver
 *
 * SMG is a semicoarsening multigrid solver that uses plane smoothing (in 3D).
 * The plane smoother calls a 2D SMG algorithm with line smoothing, and the line
 * smoother is cyclic reduction (1D SMG).  For periodic problems, the grid size
 * in periodic dimensions currently must be a power-of-two.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructSMGCreate(MPI_Comm            comm,
                                HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructSMGDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_StructSMGSetup(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_StructSMGSolve(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_StructSMGSetMemoryUse(HYPRE_StructSolver solver,
                                      HYPRE_Int          memory_use);

HYPRE_Int HYPRE_StructSMGGetMemoryUse(HYPRE_StructSolver solver,
                                      HYPRE_Int *memory_use);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_StructSMGSetTol(HYPRE_StructSolver solver,
                                HYPRE_Real         tol);

HYPRE_Int HYPRE_StructSMGGetTol(HYPRE_StructSolver solver,
                                HYPRE_Real *tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_StructSMGSetMaxIter(HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructSMGGetMaxIter(HYPRE_StructSolver solver,
                                    HYPRE_Int *max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int HYPRE_StructSMGSetRelChange(HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);

HYPRE_Int HYPRE_StructSMGGetRelChange(HYPRE_StructSolver solver,
                                      HYPRE_Int *rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
HYPRE_Int HYPRE_StructSMGSetZeroGuess(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructSMGGetZeroGuess(HYPRE_StructSolver solver,
                                      HYPRE_Int *zeroguess);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using \e SetZeroGuess.
 **/
HYPRE_Int HYPRE_StructSMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
HYPRE_Int HYPRE_StructSMGSetNumPreRelax(HYPRE_StructSolver solver,
                                        HYPRE_Int          num_pre_relax);

HYPRE_Int HYPRE_StructSMGGetNumPreRelax(HYPRE_StructSolver solver,
                                        HYPRE_Int *num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
HYPRE_Int HYPRE_StructSMGSetNumPostRelax(HYPRE_StructSolver solver,
                                         HYPRE_Int          num_post_relax);

HYPRE_Int HYPRE_StructSMGGetNumPostRelax(HYPRE_StructSolver solver,
                                         HYPRE_Int *num_post_relax);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_StructSMGSetLogging(HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);

HYPRE_Int HYPRE_StructSMGGetLogging(HYPRE_StructSolver solver,
                                    HYPRE_Int *logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_StructSMGSetPrintLevel(HYPRE_StructSolver solver,
                                       HYPRE_Int          print_level);

HYPRE_Int HYPRE_StructSMGGetPrintLevel(HYPRE_StructSolver solver,
                                       HYPRE_Int *print_level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_StructSMGGetNumIterations(HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_StructSMGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                      HYPRE_Real         *norm);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
HYPRE_Int
HYPRE_StructSMGSetDeviceLevel( HYPRE_StructSolver  solver,
                               HYPRE_Int   device_level  );
#endif

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct CycRed Solver
 *
 * CycRed is a cyclic reduction solver that simultaneously solves a collection
 * of 1D tridiagonal systems embedded in a d-dimensional grid.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructCycRedCreate(MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructCycRedDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_StructCycRedSetup(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_StructCycRedSolve(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

/**
 *
 * (Optional) Set the dimension number for the embedded 1D tridiagonal systems.
 * The default is \e tdim = 0.
 **/
HYPRE_Int HYPRE_StructCycRedSetTDim(HYPRE_StructSolver solver,
                                    HYPRE_Int          tdim);

/**
 * (Optional) Set the base index and stride for the embedded 1D systems.  The
 * stride must be equal one in the dimension corresponding to the 1D systems
 * (see \ref HYPRE_StructCycRedSetTDim).
 **/
HYPRE_Int HYPRE_StructCycRedSetBase(HYPRE_StructSolver solver,
                                    HYPRE_Int          ndim,
                                    HYPRE_Int         *base_index,
                                    HYPRE_Int         *base_stride);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PCG Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructPCGCreate(MPI_Comm            comm,
                                HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructPCGDestroy(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructPCGSetup(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

HYPRE_Int HYPRE_StructPCGSolve(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

HYPRE_Int HYPRE_StructPCGSetTol(HYPRE_StructSolver solver,
                                HYPRE_Real         tol);

HYPRE_Int HYPRE_StructPCGSetAbsoluteTol(HYPRE_StructSolver solver,
                                        HYPRE_Real         tol);

HYPRE_Int HYPRE_StructPCGSetMaxIter(HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructPCGSetTwoNorm(HYPRE_StructSolver solver,
                                    HYPRE_Int          two_norm);

HYPRE_Int HYPRE_StructPCGSetRelChange(HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);

HYPRE_Int HYPRE_StructPCGSetPrecond(HYPRE_StructSolver         solver,
                                    HYPRE_PtrToStructSolverFcn precond,
                                    HYPRE_PtrToStructSolverFcn precond_setup,
                                    HYPRE_StructSolver         precond_solver);

HYPRE_Int HYPRE_StructPCGSetLogging(HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);

HYPRE_Int HYPRE_StructPCGSetPrintLevel(HYPRE_StructSolver solver,
                                       HYPRE_Int          level);

HYPRE_Int HYPRE_StructPCGGetNumIterations(HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_StructPCGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                      HYPRE_Real         *norm);

HYPRE_Int HYPRE_StructPCGGetResidual(HYPRE_StructSolver   solver,
                                     void               **residual);

/**
 * Setup routine for diagonal preconditioning.
 **/
HYPRE_Int HYPRE_StructDiagScaleSetup(HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector y,
                                     HYPRE_StructVector x);

/**
 * Solve routine for diagonal preconditioning.
 **/
HYPRE_Int HYPRE_StructDiagScale(HYPRE_StructSolver solver,
                                HYPRE_StructMatrix HA,
                                HYPRE_StructVector Hy,
                                HYPRE_StructVector Hx);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructGMRESCreate(MPI_Comm            comm,
                                  HYPRE_StructSolver *solver);


/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructGMRESDestroy(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructGMRESSetup(HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);

HYPRE_Int HYPRE_StructGMRESSolve(HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);

HYPRE_Int HYPRE_StructGMRESSetTol(HYPRE_StructSolver solver,
                                  HYPRE_Real         tol);

HYPRE_Int HYPRE_StructGMRESSetAbsoluteTol(HYPRE_StructSolver solver,
                                          HYPRE_Real         tol);

HYPRE_Int HYPRE_StructGMRESSetMaxIter(HYPRE_StructSolver solver,
                                      HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructGMRESSetKDim(HYPRE_StructSolver solver,
                                   HYPRE_Int          k_dim);

HYPRE_Int HYPRE_StructGMRESSetPrecond(HYPRE_StructSolver         solver,
                                      HYPRE_PtrToStructSolverFcn precond,
                                      HYPRE_PtrToStructSolverFcn precond_setup,
                                      HYPRE_StructSolver         precond_solver);

HYPRE_Int HYPRE_StructGMRESSetLogging(HYPRE_StructSolver solver,
                                      HYPRE_Int          logging);

HYPRE_Int HYPRE_StructGMRESSetPrintLevel(HYPRE_StructSolver solver,
                                         HYPRE_Int          level);

HYPRE_Int HYPRE_StructGMRESGetNumIterations(HYPRE_StructSolver  solver,
                                            HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_StructGMRESGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                        HYPRE_Real         *norm);

HYPRE_Int HYPRE_StructGMRESGetResidual(HYPRE_StructSolver   solver,
                                       void               **residual);
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructFlexGMRESCreate(MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructFlexGMRESDestroy(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructFlexGMRESSetup(HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);

HYPRE_Int HYPRE_StructFlexGMRESSolve(HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);

HYPRE_Int HYPRE_StructFlexGMRESSetTol(HYPRE_StructSolver solver,
                                      HYPRE_Real         tol);

HYPRE_Int HYPRE_StructFlexGMRESSetAbsoluteTol(HYPRE_StructSolver solver,
                                              HYPRE_Real         tol);

HYPRE_Int HYPRE_StructFlexGMRESSetMaxIter(HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructFlexGMRESSetKDim(HYPRE_StructSolver solver,
                                       HYPRE_Int          k_dim);

HYPRE_Int HYPRE_StructFlexGMRESSetPrecond(HYPRE_StructSolver         solver,
                                          HYPRE_PtrToStructSolverFcn precond,
                                          HYPRE_PtrToStructSolverFcn precond_setup,
                                          HYPRE_StructSolver         precond_solver);

HYPRE_Int HYPRE_StructFlexGMRESSetLogging(HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);

HYPRE_Int HYPRE_StructFlexGMRESSetPrintLevel(HYPRE_StructSolver solver,
                                             HYPRE_Int          level);

HYPRE_Int HYPRE_StructFlexGMRESGetNumIterations(HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                            HYPRE_Real         *norm);

HYPRE_Int HYPRE_StructFlexGMRESGetResidual(HYPRE_StructSolver   solver,
                                           void               **residual);

HYPRE_Int HYPRE_StructFlexGMRESSetModifyPC(HYPRE_StructSolver     solver,
                                           HYPRE_PtrToModifyPCFcn modify_pc);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructLGMRESCreate(MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructLGMRESDestroy(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructLGMRESSetup(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

HYPRE_Int HYPRE_StructLGMRESSolve(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

HYPRE_Int HYPRE_StructLGMRESSetTol(HYPRE_StructSolver solver,
                                   HYPRE_Real         tol);

HYPRE_Int HYPRE_StructLGMRESSetAbsoluteTol(HYPRE_StructSolver solver,
                                           HYPRE_Real         tol);

HYPRE_Int HYPRE_StructLGMRESSetMaxIter(HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructLGMRESSetKDim(HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);

HYPRE_Int HYPRE_StructLGMRESSetAugDim(HYPRE_StructSolver solver,
                                      HYPRE_Int          aug_dim);

HYPRE_Int HYPRE_StructLGMRESSetPrecond(HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);

HYPRE_Int HYPRE_StructLGMRESSetLogging(HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);

HYPRE_Int HYPRE_StructLGMRESSetPrintLevel(HYPRE_StructSolver solver,
                                          HYPRE_Int          level);

HYPRE_Int HYPRE_StructLGMRESGetNumIterations(HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_StructLGMRESGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                         HYPRE_Real         *norm);

HYPRE_Int HYPRE_StructLGMRESGetResidual(HYPRE_StructSolver   solver,
                                        void               **residual);
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructBiCGSTABCreate(MPI_Comm            comm,
                                     HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructBiCGSTABDestroy(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructBiCGSTABSetup(HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);

HYPRE_Int HYPRE_StructBiCGSTABSolve(HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);

HYPRE_Int HYPRE_StructBiCGSTABSetTol(HYPRE_StructSolver solver,
                                     HYPRE_Real         tol);

HYPRE_Int HYPRE_StructBiCGSTABSetAbsoluteTol(HYPRE_StructSolver solver,
                                             HYPRE_Real         tol);

HYPRE_Int HYPRE_StructBiCGSTABSetMaxIter(HYPRE_StructSolver solver,
                                         HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructBiCGSTABSetPrecond(HYPRE_StructSolver         solver,
                                         HYPRE_PtrToStructSolverFcn precond,
                                         HYPRE_PtrToStructSolverFcn precond_setup,
                                         HYPRE_StructSolver         precond_solver);

HYPRE_Int HYPRE_StructBiCGSTABSetLogging(HYPRE_StructSolver solver,
                                         HYPRE_Int          logging);

HYPRE_Int HYPRE_StructBiCGSTABSetPrintLevel(HYPRE_StructSolver solver,
                                            HYPRE_Int          level);

HYPRE_Int HYPRE_StructBiCGSTABGetNumIterations(HYPRE_StructSolver  solver,
                                               HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                           HYPRE_Real         *norm);

HYPRE_Int HYPRE_StructBiCGSTABGetResidual( HYPRE_StructSolver   solver,
                                           void               **residual);
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Hybrid Solver
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_StructHybridCreate(MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_StructHybridDestroy(HYPRE_StructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_StructHybridSetup(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_StructHybridSolve(HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_StructHybridSetTol(HYPRE_StructSolver solver,
                                   HYPRE_Real         tol);

/**
 * (Optional) Set an accepted convergence tolerance for diagonal scaling (DS).
 * The solver will switch preconditioners if the convergence of DS is slower
 * than \e cf_tol.
 **/
HYPRE_Int HYPRE_StructHybridSetConvergenceTol(HYPRE_StructSolver solver,
                                              HYPRE_Real         cf_tol);

/**
 * (Optional) Set maximum number of iterations for diagonal scaling (DS).  The
 * solver will switch preconditioners if DS reaches \e ds_max_its.
 **/
HYPRE_Int HYPRE_StructHybridSetDSCGMaxIter(HYPRE_StructSolver solver,
                                           HYPRE_Int          ds_max_its);

/**
 * (Optional) Set maximum number of iterations for general preconditioner (PRE).
 * The solver will stop if PRE reaches \e pre_max_its.
 **/
HYPRE_Int HYPRE_StructHybridSetPCGMaxIter(HYPRE_StructSolver solver,
                                          HYPRE_Int          pre_max_its);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
HYPRE_Int HYPRE_StructHybridSetTwoNorm(HYPRE_StructSolver solver,
                                       HYPRE_Int          two_norm);

HYPRE_Int HYPRE_StructHybridSetStopCrit(HYPRE_StructSolver solver,
                                        HYPRE_Int          stop_crit);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int HYPRE_StructHybridSetRelChange(HYPRE_StructSolver solver,
                                         HYPRE_Int          rel_change);

/**
 * (Optional) Set the type of Krylov solver to use.
 *
 * Current krylov methods set by \e solver_type are:
 *
 *    - 0 : PCG (default)
 *    - 1 : GMRES
 *    - 2 : BiCGSTAB
 **/
HYPRE_Int HYPRE_StructHybridSetSolverType(HYPRE_StructSolver solver,
                                          HYPRE_Int          solver_type);

/**
 * (Optional) Set recompute residual (don't rely on 3-term recurrence).
 **/
HYPRE_Int
HYPRE_StructHybridSetRecomputeResidual( HYPRE_StructSolver  solver,
                                        HYPRE_Int           recompute_residual );

/**
 * (Optional) Get recompute residual option.
 **/
HYPRE_Int
HYPRE_StructHybridGetRecomputeResidual( HYPRE_StructSolver  solver,
                                        HYPRE_Int          *recompute_residual );

/**
 * (Optional) Set recompute residual period (don't rely on 3-term recurrence).
 *
 * Recomputes residual after every specified number of iterations.
 **/
HYPRE_Int
HYPRE_StructHybridSetRecomputeResidualP( HYPRE_StructSolver  solver,
                                         HYPRE_Int           recompute_residual_p );

/**
 * (Optional) Get recompute residual period option.
 **/
HYPRE_Int
HYPRE_StructHybridGetRecomputeResidualP( HYPRE_StructSolver  solver,
                                         HYPRE_Int          *recompute_residual_p );

/**
 * (Optional) Set the maximum size of the Krylov space when using GMRES.
 **/
HYPRE_Int HYPRE_StructHybridSetKDim(HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);

/**
 * (Optional) Set the preconditioner to use.
 **/
HYPRE_Int HYPRE_StructHybridSetPrecond(HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_StructHybridSetLogging(HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_StructHybridSetPrintLevel(HYPRE_StructSolver solver,
                                          HYPRE_Int          print_level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_StructHybridGetNumIterations(HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_its);

/**
 * Return the number of diagonal scaling iterations taken.
 **/
HYPRE_Int HYPRE_StructHybridGetDSCGNumIterations(HYPRE_StructSolver  solver,
                                                 HYPRE_Int          *ds_num_its);

/**
 * Return the number of general preconditioning iterations taken.
 **/
HYPRE_Int HYPRE_StructHybridGetPCGNumIterations(HYPRE_StructSolver  solver,
                                                HYPRE_Int          *pre_num_its);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_StructHybridGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                         HYPRE_Real         *norm);

HYPRE_Int HYPRE_StructHybridSetPCGAbsoluteTolFactor(HYPRE_StructSolver solver,
                                                    HYPRE_Real pcg_atolf );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name Struct SparseMSG Solver
 **/

HYPRE_Int HYPRE_StructSparseMSGCreate(MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);

HYPRE_Int HYPRE_StructSparseMSGDestroy(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructSparseMSGSetup(HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);

HYPRE_Int HYPRE_StructSparseMSGSolve(HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);

HYPRE_Int HYPRE_StructSparseMSGSetTol(HYPRE_StructSolver solver,
                                      HYPRE_Real         tol);

HYPRE_Int HYPRE_StructSparseMSGSetMaxIter(HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);

HYPRE_Int HYPRE_StructSparseMSGSetJump(HYPRE_StructSolver solver,
                                       HYPRE_Int          jump);

HYPRE_Int HYPRE_StructSparseMSGSetRelChange(HYPRE_StructSolver solver,
                                            HYPRE_Int          rel_change);

HYPRE_Int HYPRE_StructSparseMSGSetZeroGuess(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructSparseMSGSetNonZeroGuess(HYPRE_StructSolver solver);

HYPRE_Int HYPRE_StructSparseMSGSetRelaxType(HYPRE_StructSolver solver,
                                            HYPRE_Int          relax_type);

HYPRE_Int HYPRE_StructSparseMSGSetJacobiWeight(HYPRE_StructSolver solver,
                                               HYPRE_Real         weight);

HYPRE_Int HYPRE_StructSparseMSGSetNumPreRelax(HYPRE_StructSolver solver,
                                              HYPRE_Int          num_pre_relax);

HYPRE_Int HYPRE_StructSparseMSGSetNumPostRelax(HYPRE_StructSolver solver,
                                               HYPRE_Int          num_post_relax);

HYPRE_Int HYPRE_StructSparseMSGSetNumFineRelax(HYPRE_StructSolver solver,
                                               HYPRE_Int          num_fine_relax);

HYPRE_Int HYPRE_StructSparseMSGSetLogging(HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);

HYPRE_Int HYPRE_StructSparseMSGSetPrintLevel(HYPRE_StructSolver solver,
                                             HYPRE_Int   print_level);


HYPRE_Int HYPRE_StructSparseMSGGetNumIterations(HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                            HYPRE_Real         *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref Eigensolvers.
 *
 * @{
 **/

/**
 * Load interface interpreter. Vector part loaded with hypre_StructKrylov
 * functions and multivector part loaded with mv_TempMultiVector functions.
 **/
HYPRE_Int
HYPRE_StructSetupInterpreter(mv_InterfaceInterpreter *i);

/**
 * Load Matvec interpreter with hypre_StructKrylov functions.
 **/
HYPRE_Int
HYPRE_StructSetupMatvec(HYPRE_MatvecFunctions *mv);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#endif

