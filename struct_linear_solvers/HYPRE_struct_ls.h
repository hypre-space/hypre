/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice,
contact person,
and disclaimer.
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
 * @name Struct Linear Solvers Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A linear solver interface for structured grids
 * @version 1.0
 * @author Robert D. Falgout
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Solvers
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_StructSolver} object ...
 **/
struct hypre_StructSolver_struct;
typedef struct hypre_StructSolver_struct *HYPRE_StructSolver;

typedef int (*HYPRE_PtrToStructSolverFcn)(HYPRE_StructSolver,
                                          HYPRE_StructMatrix,
                                          HYPRE_StructVector,
                                          HYPRE_StructVector);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PCG Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGCreate(MPI_Comm            comm,
                          HYPRE_StructSolver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGDestroy(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSetup(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSolve(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSetTol(HYPRE_StructSolver solver,
                          double             tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSetMaxIter(HYPRE_StructSolver solver,
                              int                max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSetTwoNorm(HYPRE_StructSolver solver,
                              int                two_norm);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSetRelChange(HYPRE_StructSolver solver,
                                int                rel_change);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSetPrecond(HYPRE_StructSolver         solver,
                              HYPRE_PtrToStructSolverFcn precond,
                              HYPRE_PtrToStructSolverFcn precond_setup,
                              HYPRE_StructSolver         precond_solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGSetLogging(HYPRE_StructSolver solver,
                              int                logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGGetNumIterations(HYPRE_StructSolver  solver,
                                    int                *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPCGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                double             *norm);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructDiagScaleSetup(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector y,
                               HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructDiagScale(HYPRE_StructSolver solver,
                          HYPRE_StructMatrix HA,
                          HYPRE_StructVector Hy,
                          HYPRE_StructVector Hx);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Hybrid Solver
 *
 * Description...
 **/
/*@{*/

/* HYPRE_struct_hybrid.c */

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridCreate(MPI_Comm            comm,
                             HYPRE_StructSolver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridDestroy(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetup(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSolve(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetTol(HYPRE_StructSolver solver,
                             double             tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetConvergenceTol(HYPRE_StructSolver solver,
                                        double             cf_tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetDSCGMaxIter(HYPRE_StructSolver solver,
                                     int                dscg_max_its);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetPCGMaxIter(HYPRE_StructSolver solver,
                                    int                pcg_max_its);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetTwoNorm(HYPRE_StructSolver solver,
                                 int                two_norm);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetRelChange(HYPRE_StructSolver solver,
                                   int                rel_change);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetPrecond(HYPRE_StructSolver         solver,
                                 HYPRE_PtrToStructSolverFcn precond,
                                 HYPRE_PtrToStructSolverFcn precond_setup,
                                 HYPRE_StructSolver         precond_solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridSetLogging(HYPRE_StructSolver solver,
                                 int                logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridGetNumIterations(HYPRE_StructSolver  solver,
                                       int                *num_its);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridGetDSCGNumIterations(HYPRE_StructSolver  solver,
                                           int                *dscg_num_its);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridGetPCGNumIterations(HYPRE_StructSolver  solver,
                                          int                *pcg_num_its);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructHybridGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                   double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct Jacobi Solver
 *
 * Description...
 **/
/*@{*/

/* HYPRE_struct_jacobi.c */

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiCreate(MPI_Comm            comm,
                             HYPRE_StructSolver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiDestroy(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiSetup(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiSolve(HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector b,
                            HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiSetTol(HYPRE_StructSolver solver,
                             double             tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiSetMaxIter(HYPRE_StructSolver solver,
                                 int                max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiSetZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiGetNumIterations(HYPRE_StructSolver  solver,
                                       int                *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructJacobiGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                   double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct PFMG Solver
 *
 * Description...
 **/
/*@{*/

/* HYPRE_struct_pfmg.c */

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGCreate(MPI_Comm            comm,
                           HYPRE_StructSolver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGDestroy(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetup(HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSolve(HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetTol(HYPRE_StructSolver solver,
                           double             tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetMaxIter(HYPRE_StructSolver solver,
                               int                max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetRelChange(HYPRE_StructSolver solver,
                                 int                rel_change);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetRelaxType(HYPRE_StructSolver solver,
                                 int                relax_type);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetNumPreRelax(HYPRE_StructSolver solver,
                                   int                num_pre_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetNumPostRelax(HYPRE_StructSolver solver,
                                    int                num_post_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetSkipRelax(HYPRE_StructSolver solver,
                                 int                skip_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetDxyz(HYPRE_StructSolver  solver,
                            double             *dxyz);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGSetLogging(HYPRE_StructSolver solver,
                               int                logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGGetNumIterations(HYPRE_StructSolver  solver,
                                     int                *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructPFMGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                 double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct SMG Solver
 *
 * Description...
 **/
/*@{*/

/* HYPRE_struct_smg.c */

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGCreate(MPI_Comm            comm,
                          HYPRE_StructSolver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGDestroy(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetup(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSolve(HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetMemoryUse(HYPRE_StructSolver solver,
                                int                memory_use);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetTol(HYPRE_StructSolver solver,
                          double             tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetMaxIter(HYPRE_StructSolver solver,
                              int                max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetRelChange(HYPRE_StructSolver solver,
                                int                rel_change);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetNumPreRelax(HYPRE_StructSolver solver,
                                  int                num_pre_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetNumPostRelax(HYPRE_StructSolver solver,
                                   int                num_post_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGSetLogging(HYPRE_StructSolver solver,
                              int                logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGGetNumIterations(HYPRE_StructSolver  solver,
                                    int                *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSMGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Struct SparseMSGSolver
 *
 * Description...
 **/
/*@{*/

/* HYPRE_struct_sparse_msg.c */

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGCreate(MPI_Comm            comm,
                                HYPRE_StructSolver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGDestroy(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetup(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSolve(HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetTol(HYPRE_StructSolver solver,
                                double             tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetMaxIter(HYPRE_StructSolver solver,
                                    int                max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetJump(HYPRE_StructSolver solver,
                                 int                jump);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetRelChange(HYPRE_StructSolver solver,
                                      int                rel_change);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetNonZeroGuess(HYPRE_StructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetRelaxType(HYPRE_StructSolver solver,
                                      int                relax_type);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetNumPreRelax(HYPRE_StructSolver solver,
                                        int                num_pre_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetNumPostRelax(HYPRE_StructSolver solver,
                                         int                num_post_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetNumFineRelax(HYPRE_StructSolver solver,
                                         int                num_fine_relax);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGSetLogging(HYPRE_StructSolver solver,
                                    int                logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGGetNumIterations(HYPRE_StructSolver  solver,
                                          int                *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(HYPRE_StructSolver  solver,
                                                      double             *norm);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif

