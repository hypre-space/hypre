/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef HYPRE_SSTRUCT_LS_HEADER
#define HYPRE_SSTRUCT_LS_HEADER

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "HYPRE_sstruct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Linear Solvers Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A linear solver interface for semi-structured grids
 * @version 0.2
 * @author Robert D. Falgout
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct Solvers
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_SStructSolver} object ...
 **/
struct hypre_SStructSolver_struct;
typedef struct hypre_SStructSolver_struct *HYPRE_SStructSolver;

typedef int (*HYPRE_PtrToStructSolverFcn)(HYPRE_SStructSolver,
                                          HYPRE_SStructMatrix,
                                          HYPRE_SStructVector,
                                          HYPRE_SStructVector);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct GMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESCreate(MPI_Comm             comm,
                             HYPRE_SStructSolver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESDestroy(HYPRE_SStructSolver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetKDim(HYPRE_SStructSolver solver,
                              int                 k_dim);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetTol(HYPRE_SStructSolver solver,
                             double              tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetMinIter(HYPRE_SStructSolver solver,
                                 int                 min_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetMaxIter(HYPRE_SStructSolver solver,
                                 int                 max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetStopCrit(HYPRE_SStructSolver solver,
                                  int                 stop_crit);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetPrecond(HYPRE_SStructSolver         solver,
                                 HYPRE_PtrToStructSolverFcn  precond,
                                 HYPRE_PtrToStructSolverFcn  precond_setup,
                                 void                       *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESSetLogging(HYPRE_SStructSolver solver,
                                 int                 logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESGetNumIterations(HYPRE_SStructSolver  solver,
                                       int                 *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_SStructGMRESGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif

