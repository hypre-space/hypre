/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for HYPRE_ls library
 *
 *****************************************************************************/

#ifndef HYPRE_STRUCT_LS_HEADER
#define HYPRE_STRUCT_LS_HEADER

#include "HYPRE_config.h"
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
 * (Optional) Use a zero initial guess.
 **/
int HYPRE_StructJacobiSetZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.
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
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_StructPFMGSetRelChange(HYPRE_StructSolver solver,
                                 int                rel_change);

/**
 * (Optional) Use a zero initial guess.
 **/
int HYPRE_StructPFMGSetZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.
 **/
int HYPRE_StructPFMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Set relaxation type.
 **/
int HYPRE_StructPFMGSetRelaxType(HYPRE_StructSolver solver,
                                 int                relax_type);

/**
 * (Optional) Set number of pre-relaxation sweeps.
 **/
int HYPRE_StructPFMGSetNumPreRelax(HYPRE_StructSolver solver,
                                   int                num_pre_relax);

/**
 * (Optional) Set number of post-relaxation sweeps.
 **/
int HYPRE_StructPFMGSetNumPostRelax(HYPRE_StructSolver solver,
                                    int                num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.
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
 * (Optional) Use a zero initial guess.
 **/
int HYPRE_StructSMGSetZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.
 **/
int HYPRE_StructSMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * (Optional) Set number of pre-relaxation sweeps.
 **/
int HYPRE_StructSMGSetNumPreRelax(HYPRE_StructSolver solver,
                                  int                num_pre_relax);

/**
 * (Optional) Set number of post-relaxation sweeps.
 **/
int HYPRE_StructSMGSetNumPostRelax(HYPRE_StructSolver solver,
                                   int                num_post_relax);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_StructSMGSetLogging(HYPRE_StructSolver solver,
                              int                logging);

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

/**
 **/
int HYPRE_StructPCGSetup(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/**
 * Solve the system.
 **/
int HYPRE_StructPCGSolve(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_StructPCGSetTol(HYPRE_StructSolver solver,
                          double             tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_StructPCGSetMaxIter(HYPRE_StructSolver solver,
                              int                max_iter);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
int HYPRE_StructPCGSetTwoNorm(HYPRE_StructSolver solver,
                              int                two_norm);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_StructPCGSetRelChange(HYPRE_StructSolver solver,
                                int                rel_change);

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_StructPCGSetPrecond(HYPRE_StructSolver         solver,
                              HYPRE_PtrToStructSolverFcn precond,
                              HYPRE_PtrToStructSolverFcn precond_setup,
                              HYPRE_StructSolver         precond_solver);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_StructPCGSetLogging(HYPRE_StructSolver solver,
                              int                logging);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_StructPCGGetNumIterations(HYPRE_StructSolver  solver,
                                    int                *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_StructPCGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                double             *norm);

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

/*
 * @name Struct GMRES Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int
HYPRE_StructGMRESCreate( MPI_Comm comm, HYPRE_StructSolver *solver );


/**
 * Destroy a solver object.
 **/
int 
HYPRE_StructGMRESDestroy( HYPRE_StructSolver solver );


/**
 * set up
 **/
int 
HYPRE_StructGMRESSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                        HYPRE_StructVector x      );


/**
 * Solve the system.
 **/
int 
HYPRE_StructGMRESSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                        HYPRE_StructVector x      );


/**
 * (Optional) Set the convergence tolerance.
 **/
int
HYPRE_StructGMRESSetTol( HYPRE_StructSolver solver,
                         double             tol    );

/**
 * (Optional) Set maximum number of iterations.
 **/
int
HYPRE_StructGMRESSetMaxIter( HYPRE_StructSolver solver,
                             int                max_iter );


/**
 * (Optional) Set the preconditioner to use.
 **/
int
HYPRE_StructGMRESSetPrecond( HYPRE_StructSolver         solver,
                           HYPRE_PtrToStructSolverFcn precond,
                           HYPRE_PtrToStructSolverFcn precond_setup,
                             HYPRE_StructSolver         precond_solver );

/**
 * (Optional) Set the amount of logging to do.
 **/
int
HYPRE_StructGMRESSetLogging( HYPRE_StructSolver solver,
                             int                logging );

/**
 * Return the number of iterations taken.
 **/
int
HYPRE_StructGMRESGetNumIterations( HYPRE_StructSolver  solver,
                                   int                *num_iterations );

/**
 * Return the norm of the final relative residual.
 **/
int
HYPRE_StructGMRESGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                               double             *norm   );

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

int HYPRE_StructSparseMSGSetNumPreRelax(HYPRE_StructSolver solver,
                                        int                num_pre_relax);

int HYPRE_StructSparseMSGSetNumPostRelax(HYPRE_StructSolver solver,
                                         int                num_post_relax);

int HYPRE_StructSparseMSGSetNumFineRelax(HYPRE_StructSolver solver,
                                         int                num_fine_relax);

int HYPRE_StructSparseMSGSetLogging(HYPRE_StructSolver solver,
                                    int                logging);

int HYPRE_StructSparseMSGGetNumIterations(HYPRE_StructSolver  solver,
                                          int                *num_iterations);

int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                      double             *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name Struct Hybrid Solver
 **/

int HYPRE_StructHybridCreate(MPI_Comm            comm,
                             HYPRE_StructSolver *solver);

int HYPRE_StructHybridDestroy(HYPRE_StructSolver solver);

int HYPRE_StructHybridSetup(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

int HYPRE_StructHybridSolve(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

int HYPRE_StructHybridSetTol(HYPRE_StructSolver solver,
                             double             tol);

int HYPRE_StructHybridSetConvergenceTol(HYPRE_StructSolver solver,
                                        double             cf_tol);

int HYPRE_StructHybridSetDSCGMaxIter(HYPRE_StructSolver solver,
                                     int                dscg_max_its);

int HYPRE_StructHybridSetPCGMaxIter(HYPRE_StructSolver solver,
                                    int                pcg_max_its);

int HYPRE_StructHybridSetTwoNorm(HYPRE_StructSolver solver,
                                 int                two_norm);

int HYPRE_StructHybridSetRelChange(HYPRE_StructSolver solver,
                                   int                rel_change);

int HYPRE_StructHybridSetPrecond(HYPRE_StructSolver         solver,
                                 HYPRE_PtrToStructSolverFcn precond,
                                 HYPRE_PtrToStructSolverFcn precond_setup,
                                 HYPRE_StructSolver         precond_solver);

int HYPRE_StructHybridSetLogging(HYPRE_StructSolver solver,
                                 int                logging);

int HYPRE_StructHybridGetNumIterations(HYPRE_StructSolver  solver,
                                       int                *num_its);

int HYPRE_StructHybridGetDSCGNumIterations(HYPRE_StructSolver  solver,
                                           int                *dscg_num_its);

int HYPRE_StructHybridGetPCGNumIterations(HYPRE_StructSolver  solver,
                                          int                *pcg_num_its);

int HYPRE_StructHybridGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                   double             *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*@}*/

#ifdef __cplusplus
}
#endif

#endif

