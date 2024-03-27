/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_KRYLOV_HEADER
#define HYPRE_KRYLOV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup KrylovSolvers Krylov Solvers
 *
 * A basic interface for Krylov solvers. These solvers support many of the
 * matrix/vector storage schemes in hypre.  They should be used in conjunction
 * with the storage-specific interfaces, particularly the specific Create() and
 * Destroy() functions.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Krylov Solvers
 *
 * @{
 **/

#ifndef HYPRE_MODIFYPC
#define HYPRE_MODIFYPC
typedef HYPRE_Int (*HYPRE_PtrToModifyPCFcn)(HYPRE_Solver,
                                            HYPRE_Int,
                                            HYPRE_Real);

#endif
/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name PCG Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_PCGSetup(HYPRE_Solver solver,
                         HYPRE_Matrix A,
                         HYPRE_Vector b,
                         HYPRE_Vector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_PCGSolve(HYPRE_Solver solver,
                         HYPRE_Matrix A,
                         HYPRE_Vector b,
                         HYPRE_Vector x);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
HYPRE_Int HYPRE_PCGSetTol(HYPRE_Solver solver,
                          HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is
 * 0). If one desires the convergence test to check the absolute
 * convergence tolerance \e only, then set the relative convergence
 * tolerance to 0.0.  (The default convergence test is \f$ <C*r,r> \leq\f$
 * max(relative\f$\_\f$tolerance\f$^{2} \ast <C*b, b>\f$, absolute\f$\_\f$tolerance\f$^2\f$).)
 **/
HYPRE_Int HYPRE_PCGSetAbsoluteTol(HYPRE_Solver solver,
                                  HYPRE_Real   a_tol);

/**
 * (Optional) Set a residual-based convergence tolerance which checks if
 * \f$\|r_{old}-r_{new}\| < rtol \|b\|\f$. This is useful when trying to converge to
 * very low relative and/or absolute tolerances, in order to bail-out before
 * roundoff errors affect the approximation.
 **/
HYPRE_Int HYPRE_PCGSetResidualTol(HYPRE_Solver solver,
                                  HYPRE_Real   rtol);
/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_PCGSetAbsoluteTolFactor(HYPRE_Solver solver,
                                        HYPRE_Real abstolf);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_PCGSetConvergenceFactorTol(HYPRE_Solver solver,
                                           HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_PCGSetStopCrit(HYPRE_Solver solver,
                               HYPRE_Int stop_crit);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_PCGSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int    max_iter);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
HYPRE_Int HYPRE_PCGSetTwoNorm(HYPRE_Solver solver,
                              HYPRE_Int    two_norm);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int HYPRE_PCGSetRelChange(HYPRE_Solver solver,
                                HYPRE_Int    rel_change);

/**
 * (Optional) Recompute the residual at the end to double-check convergence.
 **/
HYPRE_Int HYPRE_PCGSetRecomputeResidual(HYPRE_Solver solver,
                                        HYPRE_Int    recompute_residual);

/**
 * (Optional) Periodically recompute the residual while iterating.
 **/
HYPRE_Int HYPRE_PCGSetRecomputeResidualP(HYPRE_Solver solver,
                                         HYPRE_Int    recompute_residual_p);

/**
 * (Optional) Setting this to 1 allows use of Polak-Ribiere Method (flexible)
 * this incrceases robustness, but adds an additional dot product per iteration
 **/
HYPRE_Int HYPRE_PCGSetFlex(HYPRE_Solver solver,
                           HYPRE_Int    flex);

/**
 * (Optional) Skips subnormal alpha, gamma and iprod values in CG.
 *  If set to 0 (default): will break if values are below HYPRE_REAL_MIN
 *  If set to 1: will break if values are below HYPRE_REAL_TRUE_MIN
 *  (requires C11 minimal or will check to HYPRE_REAL_MIN)
 *  If set to 2: will break if values are <= 0.
 *  If set to 3 or larger: will not break at all
 **/
HYPRE_Int HYPRE_PCGSetSkipBreak(HYPRE_Solver solver,
                                HYPRE_Int    skip_break);

/**
 * (Optional) Set the preconditioner to use.
 **/
HYPRE_Int HYPRE_PCGSetPrecond(HYPRE_Solver         solver,
                              HYPRE_PtrToSolverFcn precond,
                              HYPRE_PtrToSolverFcn precond_setup,
                              HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the preconditioner to use in a generic fashion.
 * This function does not require explicit input of the setup and solve pointers
 * of the preconditioner object. Instead, it automatically extracts this information
 * from the aforementioned object.
 **/
HYPRE_Int HYPRE_PCGSetPreconditioner(HYPRE_Solver  solver,
                                     HYPRE_Solver  precond);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_PCGSetLogging(HYPRE_Solver solver,
                              HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_PCGSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_PCGGetNumIterations(HYPRE_Solver  solver,
                                    HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_PCGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
HYPRE_Int HYPRE_PCGGetResidual(HYPRE_Solver  solver,
                               void         *residual);

/**
 **/
HYPRE_Int HYPRE_PCGGetTol(HYPRE_Solver  solver,
                          HYPRE_Real   *tol);

/**
 **/
HYPRE_Int HYPRE_PCGGetResidualTol(HYPRE_Solver  solver,
                                  HYPRE_Real   *rtol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_PCGGetAbsoluteTolFactor(HYPRE_Solver solver,
                                        HYPRE_Real  *abstolf);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_PCGGetConvergenceFactorTol(HYPRE_Solver solver,
                                           HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_PCGGetStopCrit(HYPRE_Solver solver,
                               HYPRE_Int   *stop_crit);

/**
 **/
HYPRE_Int HYPRE_PCGGetMaxIter(HYPRE_Solver  solver,
                              HYPRE_Int    *max_iter);

/**
 **/
HYPRE_Int HYPRE_PCGGetTwoNorm(HYPRE_Solver  solver,
                              HYPRE_Int    *two_norm);

/**
 **/
HYPRE_Int HYPRE_PCGGetRelChange(HYPRE_Solver  solver,
                                HYPRE_Int    *rel_change);

/**
 **/
HYPRE_Int HYPRE_PCGGetSkipBreak(HYPRE_Solver solver,
                                HYPRE_Int   *skip_break);

/**
 **/
HYPRE_Int HYPRE_PCGGetFlex(HYPRE_Solver solver,
                           HYPRE_Int   *flex);

/**
 **/
HYPRE_Int HYPRE_PCGGetPrecond(HYPRE_Solver  solver,
                              HYPRE_Solver *precond_data_ptr);

/**
 **/
HYPRE_Int HYPRE_PCGGetLogging(HYPRE_Solver  solver,
                              HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_PCGGetPrintLevel(HYPRE_Solver  solver,
                                 HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_PCGGetConverged(HYPRE_Solver  solver,
                                HYPRE_Int          *converged);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name GMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_GMRESSetup(HYPRE_Solver solver,
                           HYPRE_Matrix A,
                           HYPRE_Vector b,
                           HYPRE_Vector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_GMRESSolve(HYPRE_Solver solver,
                           HYPRE_Matrix A,
                           HYPRE_Vector b,
                           HYPRE_Vector x);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
HYPRE_Int HYPRE_GMRESSetTol(HYPRE_Solver solver,
                            HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
HYPRE_Int HYPRE_GMRESSetAbsoluteTol(HYPRE_Solver solver,
                                    HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_GMRESSetConvergenceFactorTol(HYPRE_Solver solver,
                                             HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_GMRESSetStopCrit(HYPRE_Solver solver,
                                 HYPRE_Int stop_crit);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_GMRESSetMinIter(HYPRE_Solver solver,
                                HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_GMRESSetMaxIter(HYPRE_Solver solver,
                                HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
HYPRE_Int HYPRE_GMRESSetKDim(HYPRE_Solver solver,
                             HYPRE_Int    k_dim);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int HYPRE_GMRESSetRelChange(HYPRE_Solver solver,
                                  HYPRE_Int    rel_change);

/**
 * (Optional) By default, hypre checks for convergence by evaluating the actual
 * residual before returnig from GMRES (with restart if the true residual does
 * not indicate convergence). This option allows users to skip the evaluation
 * and the check of the actual residual for badly conditioned problems where
 * restart is not expected to be beneficial.
 **/
HYPRE_Int HYPRE_GMRESSetSkipRealResidualCheck(HYPRE_Solver solver,
                                              HYPRE_Int    skip_real_r_check);

/**
 * (Optional) Set the preconditioner to use.
 **/
HYPRE_Int HYPRE_GMRESSetPrecond(HYPRE_Solver         solver,
                                HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup,
                                HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_GMRESSetLogging(HYPRE_Solver solver,
                                HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_GMRESSetPrintLevel(HYPRE_Solver solver,
                                   HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_GMRESGetNumIterations(HYPRE_Solver  solver,
                                      HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_GMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                  HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
HYPRE_Int HYPRE_GMRESGetResidual(HYPRE_Solver   solver,
                                 void          *residual);

/**
 **/
HYPRE_Int HYPRE_GMRESGetSkipRealResidualCheck(HYPRE_Solver solver,
                                              HYPRE_Int   *skip_real_r_check);

/**
 **/
HYPRE_Int HYPRE_GMRESGetTol(HYPRE_Solver  solver,
                            HYPRE_Real   *tol);

/**
 **/
HYPRE_Int HYPRE_GMRESGetAbsoluteTol(HYPRE_Solver  solver,
                                    HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_GMRESGetConvergenceFactorTol(HYPRE_Solver solver,
                                             HYPRE_Real  *cf_tol);

/*
 * OBSOLETE
 **/
HYPRE_Int HYPRE_GMRESGetStopCrit(HYPRE_Solver solver,
                                 HYPRE_Int   *stop_crit);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_GMRESGetMinIter(HYPRE_Solver solver,
                                HYPRE_Int   *min_iter);

/**
 **/
HYPRE_Int HYPRE_GMRESGetMaxIter(HYPRE_Solver  solver,
                                HYPRE_Int    *max_iter);

/**
 **/
HYPRE_Int HYPRE_GMRESGetKDim(HYPRE_Solver  solver,
                             HYPRE_Int    *k_dim);

/**
 **/
HYPRE_Int HYPRE_GMRESGetRelChange(HYPRE_Solver  solver,
                                  HYPRE_Int    *rel_change);

/**
 **/
HYPRE_Int HYPRE_GMRESGetPrecond(HYPRE_Solver  solver,
                                HYPRE_Solver *precond_data_ptr);

/**
 **/
HYPRE_Int HYPRE_GMRESGetLogging(HYPRE_Solver  solver,
                                HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_GMRESGetPrintLevel(HYPRE_Solver  solver,
                                   HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_GMRESGetConverged(HYPRE_Solver  solver,
                                  HYPRE_Int    *converged);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FlexGMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_FlexGMRESSetup(HYPRE_Solver solver,
                               HYPRE_Matrix A,
                               HYPRE_Vector b,
                               HYPRE_Vector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_FlexGMRESSolve(HYPRE_Solver solver,
                               HYPRE_Matrix A,
                               HYPRE_Vector b,
                               HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_FlexGMRESSetTol(HYPRE_Solver solver,
                                HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
HYPRE_Int HYPRE_FlexGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                        HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_FlexGMRESSetConvergenceFactorTol(HYPRE_Solver solver, HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_FlexGMRESSetMinIter(HYPRE_Solver solver, HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_FlexGMRESSetMaxIter(HYPRE_Solver solver,
                                    HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
HYPRE_Int HYPRE_FlexGMRESSetKDim(HYPRE_Solver solver,
                                 HYPRE_Int    k_dim);

/**
 * (Optional) Set the preconditioner to use.
 **/
HYPRE_Int HYPRE_FlexGMRESSetPrecond(HYPRE_Solver         solver,
                                    HYPRE_PtrToSolverFcn precond,
                                    HYPRE_PtrToSolverFcn precond_setup,
                                    HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_FlexGMRESSetLogging(HYPRE_Solver solver,
                                    HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_FlexGMRESSetPrintLevel(HYPRE_Solver solver,
                                       HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_FlexGMRESGetNumIterations(HYPRE_Solver  solver,
                                          HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_FlexGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                      HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
HYPRE_Int HYPRE_FlexGMRESGetResidual(HYPRE_Solver   solver,
                                     void          *residual);

/**
 **/
HYPRE_Int HYPRE_FlexGMRESGetTol(HYPRE_Solver  solver,
                                HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_FlexGMRESGetConvergenceFactorTol(HYPRE_Solver solver,
                                                 HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_FlexGMRESGetStopCrit(HYPRE_Solver solver,
                                     HYPRE_Int   *stop_crit);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_FlexGMRESGetMinIter(HYPRE_Solver solver,
                                    HYPRE_Int   *min_iter);

/**
 **/
HYPRE_Int HYPRE_FlexGMRESGetMaxIter(HYPRE_Solver  solver,
                                    HYPRE_Int    *max_iter);

/**
 **/
HYPRE_Int HYPRE_FlexGMRESGetKDim(HYPRE_Solver  solver,
                                 HYPRE_Int    *k_dim);

/**
 **/
HYPRE_Int HYPRE_FlexGMRESGetPrecond(HYPRE_Solver  solver,
                                    HYPRE_Solver *precond_data_ptr);

/**
 **/
HYPRE_Int HYPRE_FlexGMRESGetLogging(HYPRE_Solver  solver,
                                    HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_FlexGMRESGetPrintLevel(HYPRE_Solver  solver,
                                       HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_FlexGMRESGetConverged(HYPRE_Solver  solver,
                                      HYPRE_Int    *converged);

/**
 * (Optional) Set a user-defined function to modify solve-time preconditioner
 * attributes.
 **/
HYPRE_Int HYPRE_FlexGMRESSetModifyPC(HYPRE_Solver           solver,
                                     HYPRE_PtrToModifyPCFcn modify_pc);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name LGMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_LGMRESSetup(HYPRE_Solver solver,
                            HYPRE_Matrix A,
                            HYPRE_Vector b,
                            HYPRE_Vector x);

/**
 * Solve the system. Details on LGMRES may be found in A. H. Baker,
 * E.R. Jessup, and T.A. Manteuffel, "A technique for accelerating the
 * convergence of restarted GMRES." SIAM Journal on Matrix Analysis and
 * Applications, 26 (2005), pp. 962-984. LGMRES(m,k) in the paper
 * corresponds to LGMRES(Kdim+AugDim, AugDim).
 **/
HYPRE_Int HYPRE_LGMRESSolve(HYPRE_Solver solver,
                            HYPRE_Matrix A,
                            HYPRE_Vector b,
                            HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_LGMRESSetTol(HYPRE_Solver solver,
                             HYPRE_Real   tol);
/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
HYPRE_Int HYPRE_LGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                     HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_LGMRESSetConvergenceFactorTol(HYPRE_Solver solver,
                                              HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_LGMRESSetMinIter(HYPRE_Solver solver,
                                 HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int
HYPRE_LGMRESSetMaxIter(HYPRE_Solver solver,
                       HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the approximation space
 * (includes the augmentation vectors).
 **/
HYPRE_Int HYPRE_LGMRESSetKDim(HYPRE_Solver solver,
                              HYPRE_Int    k_dim);

/**
 * (Optional) Set the number of augmentation vectors  (default: 2).
 **/
HYPRE_Int HYPRE_LGMRESSetAugDim(HYPRE_Solver solver,
                                HYPRE_Int    aug_dim);

/**
 * (Optional) Set the preconditioner to use.
 **/
HYPRE_Int
HYPRE_LGMRESSetPrecond(HYPRE_Solver         solver,
                       HYPRE_PtrToSolverFcn precond,
                       HYPRE_PtrToSolverFcn precond_setup,
                       HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_LGMRESSetLogging(HYPRE_Solver solver,
                                 HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_LGMRESSetPrintLevel(HYPRE_Solver solver,
                                    HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_LGMRESGetNumIterations(HYPRE_Solver  solver,
                                       HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_LGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                   HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
HYPRE_Int HYPRE_LGMRESGetResidual(HYPRE_Solver   solver,
                                  void          *residual);

/**
 **/
HYPRE_Int HYPRE_LGMRESGetTol(HYPRE_Solver  solver,
                             HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_LGMRESGetConvergenceFactorTol(HYPRE_Solver solver,
                                              HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_LGMRESGetStopCrit(HYPRE_Solver solver,
                                  HYPRE_Int   *stop_crit);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_LGMRESGetMinIter(HYPRE_Solver solver,
                                 HYPRE_Int   *min_iter);

/**
 **/
HYPRE_Int HYPRE_LGMRESGetMaxIter(HYPRE_Solver  solver,
                                 HYPRE_Int    *max_iter);

/**
 **/
HYPRE_Int HYPRE_LGMRESGetKDim(HYPRE_Solver  solver,
                              HYPRE_Int    *k_dim);
/**
 **/
HYPRE_Int HYPRE_LGMRESGetAugDim(HYPRE_Solver  solver,
                                HYPRE_Int    *k_dim);

/**
 **/
HYPRE_Int HYPRE_LGMRESGetPrecond(HYPRE_Solver  solver,
                                 HYPRE_Solver *precond_data_ptr);

/**
 **/
HYPRE_Int HYPRE_LGMRESGetLogging(HYPRE_Solver  solver,
                                 HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_LGMRESGetPrintLevel(HYPRE_Solver  solver,
                                    HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_LGMRESGetConverged(HYPRE_Solver  solver,
                                   HYPRE_Int    *converged);

/**** added by KS ****** */
/**
 * @name COGMRES Solver
 *
 * @{
 **/

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_COGMRESSetup(HYPRE_Solver solver,
                             HYPRE_Matrix A,
                             HYPRE_Vector b,
                             HYPRE_Vector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_COGMRESSolve(HYPRE_Solver solver,
                             HYPRE_Matrix A,
                             HYPRE_Vector b,
                             HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_COGMRESSetTol(HYPRE_Solver solver,
                              HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance\f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
HYPRE_Int HYPRE_COGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                      HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_COGMRESSetConvergenceFactorTol(HYPRE_Solver solver,
                                               HYPRE_Real cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_COGMRESSetMinIter(HYPRE_Solver solver,
                                  HYPRE_Int min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_COGMRESSetMaxIter(HYPRE_Solver solver,
                                  HYPRE_Int    max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
HYPRE_Int HYPRE_COGMRESSetKDim(HYPRE_Solver solver,
                               HYPRE_Int    k_dim);

/**
 * (Optional) Set number of unrolling in mass funcyions in COGMRES
 * Can be 4 or 8. Default: no unrolling.
 **/
HYPRE_Int HYPRE_COGMRESSetUnroll(HYPRE_Solver solver,
                                 HYPRE_Int    unroll);

/**
 * (Optional) Set the number of orthogonalizations in COGMRES (at most 2).
 **/
HYPRE_Int HYPRE_COGMRESSetCGS(HYPRE_Solver solver,
                              HYPRE_Int    cgs);

/**
 * (Optional) Set the preconditioner to use.
 **/
HYPRE_Int HYPRE_COGMRESSetPrecond(HYPRE_Solver         solver,
                                  HYPRE_PtrToSolverFcn precond,
                                  HYPRE_PtrToSolverFcn precond_setup,
                                  HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_COGMRESSetLogging(HYPRE_Solver solver,
                                  HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_COGMRESSetPrintLevel(HYPRE_Solver solver,
                                     HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_COGMRESGetNumIterations(HYPRE_Solver  solver,
                                        HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_COGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                    HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
HYPRE_Int HYPRE_COGMRESGetResidual(HYPRE_Solver   solver,
                                   void          *residual);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetTol(HYPRE_Solver  solver,
                              HYPRE_Real   *tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_COGMRESGetConvergenceFactorTol(HYPRE_Solver solver,
                                               HYPRE_Real  *cf_tol);

/*
 * RE-VISIT
 **/
//HYPRE_Int HYPRE_COGMRESGetStopCrit(HYPRE_Solver solver, HYPRE_Int *stop_crit);
//HYPRE_Int HYPRE_COGMRESSetStopCrit(HYPRE_Solver solver, HYPRE_Int *stop_crit);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_COGMRESGetMinIter(HYPRE_Solver solver,
                                  HYPRE_Int   *min_iter);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetMaxIter(HYPRE_Solver  solver,
                                  HYPRE_Int    *max_iter);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetKDim(HYPRE_Solver  solver,
                               HYPRE_Int    *k_dim);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetUnroll(HYPRE_Solver  solver,
                                 HYPRE_Int    *unroll);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetCGS(HYPRE_Solver  solver,
                              HYPRE_Int    *cgs);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetPrecond(HYPRE_Solver  solver,
                                  HYPRE_Solver *precond_data_ptr);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetLogging(HYPRE_Solver  solver,
                                  HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetPrintLevel(HYPRE_Solver  solver,
                                     HYPRE_Int    *level);

/**
 **/
HYPRE_Int HYPRE_COGMRESGetConverged(HYPRE_Solver  solver,
                                    HYPRE_Int    *converged);

/**
 * (Optional) Set a user-defined function to modify solve-time preconditioner
 * attributes.
 **/
HYPRE_Int HYPRE_COGMRESSetModifyPC(HYPRE_Solver           solver,
                                   HYPRE_PtrToModifyPCFcn modify_pc);

/****** KS code ends here **************************************************/

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name BiCGSTAB Solver
 *
 * @{
 **/

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_BiCGSTABDestroy(HYPRE_Solver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_BiCGSTABSetup(HYPRE_Solver solver,
                              HYPRE_Matrix A,
                              HYPRE_Vector b,
                              HYPRE_Vector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_BiCGSTABSolve(HYPRE_Solver solver,
                              HYPRE_Matrix A,
                              HYPRE_Vector b,
                              HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_BiCGSTABSetTol(HYPRE_Solver solver,
                               HYPRE_Real   tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is 0).
 * If one desires
 * the convergence test to check the absolute convergence tolerance \e only, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is
 * \f$\|r\| \leq\f$ max(relative\f$\_\f$tolerance \f$\ast \|b\|\f$, absolute\f$\_\f$tolerance).)
 *
 **/
HYPRE_Int HYPRE_BiCGSTABSetAbsoluteTol(HYPRE_Solver solver,
                                       HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_BiCGSTABSetConvergenceFactorTol(HYPRE_Solver solver,
                                                HYPRE_Real   cf_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_BiCGSTABSetStopCrit(HYPRE_Solver solver,
                                    HYPRE_Int    stop_crit);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_BiCGSTABSetMinIter(HYPRE_Solver solver,
                                   HYPRE_Int    min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_BiCGSTABSetMaxIter(HYPRE_Solver solver,
                                   HYPRE_Int    max_iter);

/**
 * (Optional) Set the preconditioner to use.
 **/
HYPRE_Int HYPRE_BiCGSTABSetPrecond(HYPRE_Solver         solver,
                                   HYPRE_PtrToSolverFcn precond,
                                   HYPRE_PtrToSolverFcn precond_setup,
                                   HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_BiCGSTABSetLogging(HYPRE_Solver solver,
                                   HYPRE_Int    logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_BiCGSTABSetPrintLevel(HYPRE_Solver solver,
                                      HYPRE_Int    level);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_BiCGSTABGetNumIterations(HYPRE_Solver  solver,
                                         HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_BiCGSTABGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                     HYPRE_Real   *norm);

/**
 * Return the residual.
 **/
HYPRE_Int HYPRE_BiCGSTABGetResidual(HYPRE_Solver   solver,
                                    void          *residual);

/**
 **/
HYPRE_Int HYPRE_BiCGSTABGetPrecond(HYPRE_Solver  solver,
                                   HYPRE_Solver *precond_data_ptr);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name CGNR Solver
 *
 * @{
 **/

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_CGNRDestroy(HYPRE_Solver solver);

/**
 * Prepare to solve the system.  The coefficient data in \e b and \e x is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int HYPRE_CGNRSetup(HYPRE_Solver solver,
                          HYPRE_Matrix A,
                          HYPRE_Vector b,
                          HYPRE_Vector x);

/**
 * Solve the system.
 **/
HYPRE_Int HYPRE_CGNRSolve(HYPRE_Solver solver,
                          HYPRE_Matrix A,
                          HYPRE_Vector b,
                          HYPRE_Vector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int HYPRE_CGNRSetTol(HYPRE_Solver solver,
                           HYPRE_Real   tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_CGNRSetStopCrit(HYPRE_Solver solver,
                                HYPRE_Int    stop_crit);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_CGNRSetMinIter(HYPRE_Solver solver,
                               HYPRE_Int    min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_CGNRSetMaxIter(HYPRE_Solver solver,
                               HYPRE_Int    max_iter);

/**
 * (Optional) Set the preconditioner to use.
 * Note that the only preconditioner available in hypre for use with
 * CGNR is currently BoomerAMG. It requires to use Jacobi as
 * a smoother without CF smoothing, i.e. relax_type needs to be set to 0
 * or 7 and relax_order needs to be set to 0 by the user, since these
 * are not default values. It can be used with a relaxation weight for
 * Jacobi, which can significantly improve convergence.
 **/
HYPRE_Int HYPRE_CGNRSetPrecond(HYPRE_Solver         solver,
                               HYPRE_PtrToSolverFcn precond,
                               HYPRE_PtrToSolverFcn precondT,
                               HYPRE_PtrToSolverFcn precond_setup,
                               HYPRE_Solver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int HYPRE_CGNRSetLogging(HYPRE_Solver solver,
                               HYPRE_Int    logging);

#if 0 /* need to add */
/*
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int HYPRE_CGNRSetPrintLevel(HYPRE_Solver solver,
                                  HYPRE_Int    level);
#endif

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int HYPRE_CGNRGetNumIterations(HYPRE_Solver  solver,
                                     HYPRE_Int    *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_CGNRGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                 HYPRE_Real   *norm);

#if 0 /* need to add */
/*
 * Return the residual.
 **/
HYPRE_Int HYPRE_CGNRGetResidual(HYPRE_Solver   solver,
                                void         **residual);
#endif

/**
 **/
HYPRE_Int HYPRE_CGNRGetPrecond(HYPRE_Solver  solver,
                               HYPRE_Solver *precond_data_ptr);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
