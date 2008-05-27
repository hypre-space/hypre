/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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
 * $Revision$
 ***********************************************************************EHEADER*/





#ifndef HYPRE_SSTRUCT_LS_HEADER
#define HYPRE_SSTRUCT_LS_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "HYPRE.h"
#include "HYPRE_sstruct_mv.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_parcsr_ls.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Solvers
 *
 * These solvers use matrix/vector storage schemes that are taylored
 * to semi-structured grid problems.
 *
 * @memo Linear solvers for semi-structured grids
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Solvers
 **/
/*@{*/

struct hypre_SStructSolver_struct;
/**
 * The solver object.
 **/
typedef struct hypre_SStructSolver_struct *HYPRE_SStructSolver;

typedef int (*HYPRE_PtrToSStructSolverFcn)(HYPRE_SStructSolver,
                                           HYPRE_SStructMatrix,
                                           HYPRE_SStructVector,
                                           HYPRE_SStructVector);


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
 * @name SStruct PCG Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructPCGCreate(MPI_Comm             comm,
                           HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructPCGDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructPCGSetup(HYPRE_SStructSolver solver,
                          HYPRE_SStructMatrix A,
                          HYPRE_SStructVector b,
                          HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
int HYPRE_SStructPCGSolve(HYPRE_SStructSolver solver,
                          HYPRE_SStructMatrix A,
                          HYPRE_SStructVector b,
                          HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_SStructPCGSetTol(HYPRE_SStructSolver solver,
                           double              tol);

/**
 * (Optional) Set the absolute convergence tolerance (default is
 * 0). If one desires the convergence test to check the absolute
 * convergence tolerance {\it only}, then set the relative convergence
 * tolerance to 0.0.  (The default convergence test is $ <C*r,r> \leq$
 * max(relative$\_$tolerance$^{2} \ast <C*b, b>$, absolute$\_$tolerance$^2$).)
 **/
int HYPRE_SStructPCGSetAbsoluteTol(HYPRE_SStructSolver solver,
                                   double              tol);


/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructPCGSetMaxIter(HYPRE_SStructSolver solver,
                               int                 max_iter);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
int
HYPRE_SStructPCGSetTwoNorm( HYPRE_SStructSolver solver,
                            int                 two_norm );

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int
HYPRE_SStructPCGSetRelChange( HYPRE_SStructSolver solver,
                              int                 rel_change );

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_SStructPCGSetPrecond(HYPRE_SStructSolver          solver,
                               HYPRE_PtrToSStructSolverFcn  precond,
                               HYPRE_PtrToSStructSolverFcn  precond_setup,
                               void                        *precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructPCGSetLogging(HYPRE_SStructSolver solver,
                               int                 logging);


/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_SStructPCGSetPrintLevel(HYPRE_SStructSolver solver,
                                  int                 level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructPCGGetNumIterations(HYPRE_SStructSolver  solver,
                                     int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructPCGGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                 double              *norm);

/**
 * Return the residual.
 **/
int HYPRE_SStructPCGGetResidual(HYPRE_SStructSolver  solver,
                                void  **residual);

/**
 * Setup routine for diagonal preconditioning.
 **/
int HYPRE_SStructDiagScaleSetup( HYPRE_SStructSolver solver,
                                 HYPRE_SStructMatrix A,
                                 HYPRE_SStructVector y,
                                 HYPRE_SStructVector x      );

/**
 * Solve routine for diagonal preconditioning.
 **/
int HYPRE_SStructDiagScale( HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector y,
                            HYPRE_SStructVector x      );

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct GMRES Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructGMRESCreate(MPI_Comm             comm,
                             HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructGMRESDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructGMRESSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
int HYPRE_SStructGMRESSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
int HYPRE_SStructGMRESSetTol(HYPRE_SStructSolver solver,
                             double              tol);
/**
 * (Optional) Set the absolute convergence tolerance  (default: 0).
 *  If one desires
 * the convergence test to check the absolute convergence tolerance {\it only}, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is 
 * $\|r\| \leq$ max(relative$\_$tolerance$\ast \|b\|$, absolute$\_$tolerance).)
 **/
int HYPRE_SStructGMRESSetAbsoluteTol(HYPRE_SStructSolver solver,
                             double              tol);

/*
 * RE-VISIT
 **/
int HYPRE_SStructGMRESSetMinIter(HYPRE_SStructSolver solver,
                                 int                 min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructGMRESSetMaxIter(HYPRE_SStructSolver solver,
                                 int                 max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
int HYPRE_SStructGMRESSetKDim(HYPRE_SStructSolver solver,
                              int                 k_dim);

/*
 * RE-VISIT
 **/
int HYPRE_SStructGMRESSetStopCrit(HYPRE_SStructSolver solver,
                                  int                 stop_crit);

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_SStructGMRESSetPrecond(HYPRE_SStructSolver          solver,
                                 HYPRE_PtrToSStructSolverFcn  precond,
                                 HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void                        *precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructGMRESSetLogging(HYPRE_SStructSolver solver,
                                 int                 logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_SStructGMRESSetPrintLevel(HYPRE_SStructSolver solver,
                                    int                 print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructGMRESGetNumIterations(HYPRE_SStructSolver  solver,
                                       int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructGMRESGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

/**
 * Return the residual.
 **/
int HYPRE_SStructGMRESGetResidual(HYPRE_SStructSolver  solver,
                                  void   **residual);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct FlexGMRES Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructFlexGMRESCreate(MPI_Comm             comm,
                             HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructFlexGMRESDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructFlexGMRESSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
int HYPRE_SStructFlexGMRESSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
int HYPRE_SStructFlexGMRESSetTol(HYPRE_SStructSolver solver,
                             double              tol);

/**
 * (Optional) Set the absolute convergence tolerance (default: 0).
 *  If one desires
 * the convergence test to check the absolute convergence tolerance {\it only}, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is 
 * $\|r\| \leq$ max(relative$\_$tolerance$\ast \|b\|$, absolute$\_$tolerance).)
 **/
int HYPRE_SStructFlexGMRESSetAbsoluteTol(HYPRE_SStructSolver solver,
                             double              tol);



/*
 * RE-VISIT
 **/
int HYPRE_SStructFlexGMRESSetMinIter(HYPRE_SStructSolver solver,
                                 int                 min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructFlexGMRESSetMaxIter(HYPRE_SStructSolver solver,
                                 int                 max_iter);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
int HYPRE_SStructFlexGMRESSetKDim(HYPRE_SStructSolver solver,
                              int                 k_dim);



/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_SStructFlexGMRESSetPrecond(HYPRE_SStructSolver          solver,
                                 HYPRE_PtrToSStructSolverFcn  precond,
                                 HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void                        *precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructFlexGMRESSetLogging(HYPRE_SStructSolver solver,
                                 int                 logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_SStructFlexGMRESSetPrintLevel(HYPRE_SStructSolver solver,
                                    int                 print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructFlexGMRESGetNumIterations(HYPRE_SStructSolver  solver,
                                       int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

/**
 * Return the residual.
 **/
int HYPRE_SStructFlexGMRESGetResidual(HYPRE_SStructSolver  solver,
                                  void   **residual);


/**
 * Set a user-defined function to modify solve-time preconditioner attributes.
 **/

int HYPRE_SStructFlexGMRESSetModifyPC( HYPRE_SStructSolver  solver,
                                      HYPRE_PtrToModifyPCFcn modify_pc);



/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct LGMRES Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructLGMRESCreate(MPI_Comm             comm,
                             HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructLGMRESDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructLGMRESSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * Solve the system.Details on LGMRES may be found in A. H. Baker,
 * E.R. Jessup, and T.A. Manteuffel. A technique for accelerating the
 * convergence of restarted GMRES. SIAM Journal on Matrix Analysis and
 * Applications, 26 (2005), pp. 962-984. LGMRES(m,k) in the paper
 * corresponds to LGMRES(Kdim+AugDim, AugDim).
 **/
int HYPRE_SStructLGMRESSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * (Optional) Set the relative convergence tolerance.
 **/
int HYPRE_SStructLGMRESSetTol(HYPRE_SStructSolver solver,
                             double              tol);


/**
 * (Optional) Set the absolute convergence tolerance  (default: 0).
 *  If one desires
 * the convergence test to check the absolute convergence tolerance {\it only}, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is 
 * $\|r\| \leq$ max(relative$\_$tolerance$\ast \|b\|$, absolute$\_$tolerance).)
 **/
int HYPRE_SStructLGMRESSetAbsoluteTol(HYPRE_SStructSolver solver,
                             double              tol);


/*
 * RE-VISIT
 **/
int HYPRE_SStructLGMRESSetMinIter(HYPRE_SStructSolver solver,
                                 int                 min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructLGMRESSetMaxIter(HYPRE_SStructSolver solver,
                                 int                 max_iter);

/**
 * (Optional) Set the maximum size of the approximation space.
 **/
int HYPRE_SStructLGMRESSetKDim(HYPRE_SStructSolver solver,
                              int                 k_dim);
/**
 * (Optional) Set the number of augmentation vectors(default: 2).
 **/
int HYPRE_SStructLGMRESSetAugDim(HYPRE_SStructSolver solver,
                              int                 aug_dim);


/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_SStructLGMRESSetPrecond(HYPRE_SStructSolver          solver,
                                 HYPRE_PtrToSStructSolverFcn  precond,
                                 HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void                        *precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructLGMRESSetLogging(HYPRE_SStructSolver solver,
                                 int                 logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_SStructLGMRESSetPrintLevel(HYPRE_SStructSolver solver,
                                    int                 print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructLGMRESGetNumIterations(HYPRE_SStructSolver  solver,
                                       int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructLGMRESGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

/**
 * Return the residual.
 **/
int HYPRE_SStructLGMRESGetResidual(HYPRE_SStructSolver  solver,
                                  void   **residual);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct BiCGSTAB Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructBiCGSTABCreate(MPI_Comm             comm,
                             HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructBiCGSTABDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructBiCGSTABSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
int HYPRE_SStructBiCGSTABSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_SStructBiCGSTABSetTol(HYPRE_SStructSolver solver,
                             double              tol);
/**
 * (Optional) Set the absolute convergence tolerance (default is 0). 
 * If one desires
 * the convergence test to check the absolute convergence tolerance {\it only}, then
 * set the relative convergence tolerance to 0.0.  (The convergence test is 
 * $\|r\| \leq$ max(relative$\_$tolerance $\ast \|b\|$, absolute$\_$tolerance).)
 *
 **/
int HYPRE_SStructBiCGSTABSetAbsoluteTol(HYPRE_SStructSolver solver,
                                        double              tol);
/*
 * RE-VISIT
 **/
int HYPRE_SStructBiCGSTABSetMinIter(HYPRE_SStructSolver solver,
                                 int                 min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructBiCGSTABSetMaxIter(HYPRE_SStructSolver solver,
                                 int                 max_iter);

/*
 * RE-VISIT
 **/
int HYPRE_SStructBiCGSTABSetStopCrit(HYPRE_SStructSolver solver,
                                  int                 stop_crit);

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_SStructBiCGSTABSetPrecond(HYPRE_SStructSolver          solver,
                                 HYPRE_PtrToSStructSolverFcn  precond,
                                 HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void                        *precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructBiCGSTABSetLogging(HYPRE_SStructSolver solver,
                                 int                 logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_SStructBiCGSTABSetPrintLevel(HYPRE_SStructSolver solver,
                                 int                 level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructBiCGSTABGetNumIterations(HYPRE_SStructSolver  solver,
                                       int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

/**
 * Return the residual.
 **/
int HYPRE_SStructBiCGSTABGetResidual(HYPRE_SStructSolver  solver,
                                    void   **residual);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct SysPFMG Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructSysPFMGCreate( MPI_Comm             comm,
                                HYPRE_SStructSolver *solver );

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructSysPFMGDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructSysPFMGSetup(HYPRE_SStructSolver solver,
                              HYPRE_SStructMatrix A,
                              HYPRE_SStructVector b,
                              HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
int HYPRE_SStructSysPFMGSolve(HYPRE_SStructSolver solver,
                              HYPRE_SStructMatrix A,
                              HYPRE_SStructVector b,
                              HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_SStructSysPFMGSetTol(HYPRE_SStructSolver solver,
                               double              tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructSysPFMGSetMaxIter(HYPRE_SStructSolver solver,
                                   int                 max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_SStructSysPFMGSetRelChange(HYPRE_SStructSolver solver,
                                     int                 rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
int HYPRE_SStructSysPFMGSetZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
int HYPRE_SStructSysPFMGSetNonZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Set relaxation type.
 *
 * Current relaxation methods set by {\tt relax\_type} are:
 *
 * \begin{tabular}{l@{ -- }l}
 * 0 & Jacobi \\
 * 1 & Weighted Jacobi (default) \\
 * 2 & Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation) \\
 * \end{tabular}
 **/
int HYPRE_SStructSysPFMGSetRelaxType(HYPRE_SStructSolver solver,
                                     int                 relax_type);

/**
 * (Optional) Set Jacobi Weight.
 **/
int HYPRE_SStructSysPFMGSetJacobiWeight(HYPRE_SStructSolver solver,
                                        double              weight);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
int HYPRE_SStructSysPFMGSetNumPreRelax(HYPRE_SStructSolver solver,
                                       int                 num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
int HYPRE_SStructSysPFMGSetNumPostRelax(HYPRE_SStructSolver solver,
                                        int                 num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.  This can
 * greatly improve efficiency by eliminating unnecessary relaxations when the
 * underlying problem is isotropic.
 **/
int HYPRE_SStructSysPFMGSetSkipRelax(HYPRE_SStructSolver solver,
                                     int                 skip_relax);

/*
 * RE-VISIT
 **/
int HYPRE_SStructSysPFMGSetDxyz(HYPRE_SStructSolver  solver,
                                double              *dxyz);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructSysPFMGSetLogging(HYPRE_SStructSolver solver,
                                   int                 logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
int HYPRE_SStructSysPFMGSetPrintLevel(HYPRE_SStructSolver solver,
                                   int          print_level);


/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructSysPFMGGetNumIterations(HYPRE_SStructSolver  solver,
                                         int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(
                                          HYPRE_SStructSolver solver,
                                          double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Split Solver
 **/
/*@{*/

#define HYPRE_PFMG   10
#define HYPRE_SMG    11
#define HYPRE_Jacobi 17

/**
 * Create a solver object.
 **/
int HYPRE_SStructSplitCreate(MPI_Comm             comm,
                             HYPRE_SStructSolver *solver);

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructSplitDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructSplitSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
int HYPRE_SStructSplitSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_SStructSplitSetTol(HYPRE_SStructSolver solver,
                             double              tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructSplitSetMaxIter(HYPRE_SStructSolver solver,
                                 int                 max_iter);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
int HYPRE_SStructSplitSetZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
int HYPRE_SStructSplitSetNonZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Set up the type of diagonal struct solver.  Either {\tt ssolver} is
 * set to {\tt HYPRE\_SMG} or {\tt HYPRE\_PFMG}.
 **/
int HYPRE_SStructSplitSetStructSolver(HYPRE_SStructSolver solver,
                                      int                 ssolver );

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructSplitGetNumIterations(HYPRE_SStructSolver  solver,
                                       int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructSplitGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct FAC Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructFACCreate( MPI_Comm             comm,
                            HYPRE_SStructSolver *solver );

/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructFACDestroy2( HYPRE_SStructSolver solver );

/**
 * Re-distribute the composite matrix so that the amr hierachy is approximately
 * nested. Coarse underlying operators are also formed.
 **/
int HYPRE_SStructFACAMR_RAP( HYPRE_SStructMatrix  A,
                             int                (*rfactors)[3],
                             HYPRE_SStructMatrix *fac_A );

/**
 * Set up the FAC solver structure .
 **/
int HYPRE_SStructFACSetup2(HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
int HYPRE_SStructFACSolve3(HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x);

/**
 * Set up amr structure
 **/
int HYPRE_SStructFACSetPLevels(HYPRE_SStructSolver solver,
                               int                 nparts,
                               int                *plevels);
/**
 * Set up amr refinement factors
 **/
int HYPRE_SStructFACSetPRefinements(HYPRE_SStructSolver  solver,
                                    int                  nparts,
                                    int                (*rfactors)[3] );

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 * Zero off the coarse level stencils reaching into a fine level grid.
 **/
int HYPRE_SStructFACZeroCFSten(HYPRE_SStructMatrix  A,
                               HYPRE_SStructGrid    grid,
                               int                  part,
                               int                  rfactors[3]);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 * Zero off the fine level stencils reaching into a coarse level grid.
 **/
int HYPRE_SStructFACZeroFCSten(HYPRE_SStructMatrix  A,
                               HYPRE_SStructGrid    grid,
                               int                  part);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 *  Places the identity in the coarse grid matrix underlying the fine patches.
 *  Required between each pair of amr levels.
 **/
int HYPRE_SStructFACZeroAMRMatrixData(HYPRE_SStructMatrix  A,
                                      int                  part_crse,
                                      int                  rfactors[3]);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 *  Places zeros in the coarse grid vector underlying the fine patches.
 *  Required between each pair of amr levels.
 **/
int HYPRE_SStructFACZeroAMRVectorData(HYPRE_SStructVector  b,
                                      int                 *plevels,
                                      int                (*rfactors)[3] );

/**
 * (Optional) Set maximum number of FAC levels.
 **/
int HYPRE_SStructFACSetMaxLevels( HYPRE_SStructSolver solver , 
                                  int                 max_levels );
/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_SStructFACSetTol(HYPRE_SStructSolver solver,
                           double              tol);
/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructFACSetMaxIter(HYPRE_SStructSolver solver,
                               int                 max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_SStructFACSetRelChange(HYPRE_SStructSolver solver,
                                 int                 rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
int HYPRE_SStructFACSetZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
int HYPRE_SStructFACSetNonZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Set relaxation type.  See \Ref{HYPRE_SStructSysPFMGSetRelaxType}
 * for appropriate values of {\tt relax\_type}.
 **/
int HYPRE_SStructFACSetRelaxType(HYPRE_SStructSolver solver,
                                 int                 relax_type);
/**
 * (Optional) Set Jacobi weight if weighted Jacobi is used.
 **/
int HYPRE_SStructFACSetJacobiWeight(HYPRE_SStructSolver solver,
                                    double              weight);
/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
int HYPRE_SStructFACSetNumPreRelax(HYPRE_SStructSolver solver,
                                   int                 num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
int HYPRE_SStructFACSetNumPostRelax(HYPRE_SStructSolver solver,
                                    int                 num_post_relax);
/**
 * (Optional) Set coarsest solver type.
 *
 * Current solver types set by {\tt csolver\_type} are:
 *
 * \begin{tabular}{l@{ -- }l}
 * 1 & SysPFMG-PCG (default) \\
 * 2 & SysPFMG \\
 * \end{tabular}
 **/
int HYPRE_SStructFACSetCoarseSolverType(HYPRE_SStructSolver solver,
                                        int                 csolver_type);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructFACSetLogging(HYPRE_SStructSolver solver,
                               int                 logging);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructFACGetNumIterations(HYPRE_SStructSolver  solver,
                                     int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructFACGetFinalRelativeResidualNorm(HYPRE_SStructSolver solver,
                                                 double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/**
 * @name SStruct Maxwell Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructMaxwellCreate( MPI_Comm             comm,
                                HYPRE_SStructSolver *solver );
/**
 * Destroy a solver object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_SStructMaxwellDestroy( HYPRE_SStructSolver solver );

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
int HYPRE_SStructMaxwellSetup(HYPRE_SStructSolver solver,
                              HYPRE_SStructMatrix A,
                              HYPRE_SStructVector b,
                              HYPRE_SStructVector x);

/**
 * Solve the system. Full coupling of the augmented system used
 * throughout the multigrid hierarchy.
 **/
int HYPRE_SStructMaxwellSolve(HYPRE_SStructSolver solver,
                              HYPRE_SStructMatrix A,
                              HYPRE_SStructVector b,
                              HYPRE_SStructVector x);

/**
 * Solve the system. Full coupling of the augmented system used
 * only on the finest level, i.e., the node and edge multigrid
 * cycles are coupled only on the finest level.
 **/
int HYPRE_SStructMaxwellSolve2(HYPRE_SStructSolver solver,
                               HYPRE_SStructMatrix A,
                               HYPRE_SStructVector b,
                               HYPRE_SStructVector x);

/**
 * Sets the gradient operator in the Maxwell solver.
 **/
int HYPRE_SStructMaxwellSetGrad(HYPRE_SStructSolver solver,
                                HYPRE_ParCSRMatrix  T);

/**
 * Sets the coarsening factor.
 **/
int HYPRE_SStructMaxwellSetRfactors(HYPRE_SStructSolver solver,
                                    int                 rfactors[3]);

/**
 * Finds the physical boundary row ranks on all levels.
 **/
int HYPRE_SStructMaxwellPhysBdy(HYPRE_SStructGrid  *grid_l,
                                int                 num_levels,
                                int                 rfactors[3],
                                int              ***BdryRanks_ptr,
                                int               **BdryRanksCnt_ptr );

/**
 * Eliminates the rows and cols corresponding to the physical boundary in
 * a parcsr matrix.
 **/
int HYPRE_SStructMaxwellEliminateRowsCols(HYPRE_ParCSRMatrix  parA,
                                          int                 nrows,
                                          int                *rows );

/**
 * Zeros the rows corresponding to the physical boundary in
 * a par vector.
 **/
int HYPRE_SStructMaxwellZeroVector(HYPRE_ParVector  b,
                                   int             *rows,
                                   int              nrows );

/**
 * (Optional) Set the constant coefficient flag- Nedelec interpolation
 * used.
 **/
int HYPRE_SStructMaxwellSetSetConstantCoef(HYPRE_SStructSolver solver,
                                           int                 flag);

/**
 * (Optional) Creates a gradient matrix from the grid. This presupposes
 * a particular orientation of the edge elements.
 **/
int HYPRE_SStructMaxwellGrad(HYPRE_SStructGrid    grid,
                             HYPRE_ParCSRMatrix  *T);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_SStructMaxwellSetTol(HYPRE_SStructSolver solver,
                               double              tol);
/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructMaxwellSetMaxIter(HYPRE_SStructSolver solver,
                                   int                 max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_SStructMaxwellSetRelChange(HYPRE_SStructSolver solver,
                                     int                 rel_change);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
int HYPRE_SStructMaxwellSetNumPreRelax(HYPRE_SStructSolver solver,
                                       int                 num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
int HYPRE_SStructMaxwellSetNumPostRelax(HYPRE_SStructSolver solver,
                                    int                 num_post_relax);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_SStructMaxwellSetLogging(HYPRE_SStructSolver solver,
                                   int                 logging);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_SStructMaxwellGetNumIterations(HYPRE_SStructSolver  solver,
                                         int                 *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(HYPRE_SStructSolver solver,
                                                     double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/*@}*/


#ifdef __cplusplus
}
#endif

#endif

