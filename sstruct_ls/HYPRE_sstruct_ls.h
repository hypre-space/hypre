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
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_SStructPCGSetMaxIter(HYPRE_SStructSolver solver,
                               int                 max_iter);

/**
 * (Optional) Set type of norm to use in stopping criteria.
 **/
int
HYPRE_SStructPCGSetTwoNorm( HYPRE_SStructSolver solver,
                            int                 two_norm );

/**
 * (Optional) Set to use additional relative-change convergence test.
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
 * (Optional) Set the print level.
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

int HYPRE_SStructDiagScaleSetup( HYPRE_SStructSolver solver,
                                 HYPRE_SStructMatrix A,
                                 HYPRE_SStructVector y,
                                 HYPRE_SStructVector x      );

int HYPRE_SStructDiagScale( HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector y,
                            HYPRE_SStructVector x      );



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
 * (Optional) Set the print level
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
 * (Optional) Set the maximum size of the Krylov space.
 **/
int HYPRE_SStructGMRESSetKDim(HYPRE_SStructSolver solver,
                              int                 k_dim);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_SStructGMRESSetTol(HYPRE_SStructSolver solver,
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
 * (Optional) Set the print level.
 **/
int HYPRE_SStructGMRESSetPrintLevel(HYPRE_SStructSolver solver,
                                    int               print_level);

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
 * @name SStruct SysPFMG Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_SStructSysPFMGCreate( MPI_Comm             comm,
                                HYPRE_SStructSolver *solver );

/**
 * Destroy a solver object.
 **/
int HYPRE_SStructSysPFMGDestroy(HYPRE_SStructSolver solver);

/**
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
 * (Optional) Use a zero initial guess.
 **/
int HYPRE_SStructSysPFMGSetZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.
 **/
int HYPRE_SStructSysPFMGSetNonZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Set relaxation type.
 **/
int HYPRE_SStructSysPFMGSetRelaxType(HYPRE_SStructSolver solver,
                                  int                relax_type);

/**
 * (Optional) Set number of pre-relaxation sweeps.
 **/
int HYPRE_SStructSysPFMGSetNumPreRelax(HYPRE_SStructSolver solver,
                                       int                 num_pre_relax);

/**
 * (Optional) Set number of post-relaxation sweeps.
 **/
int HYPRE_SStructSysPFMGSetNumPostRelax(HYPRE_SStructSolver solver,
                                        int                 num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.
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
 * (Optional) Set the print level.
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

/*
 * @name SStruct Split Solver
 **/

#define HYPRE_SMG  0
#define HYPRE_PFMG 1

int HYPRE_SStructSplitCreate(MPI_Comm             comm,
                             HYPRE_SStructSolver *solver);

int HYPRE_SStructSplitDestroy(HYPRE_SStructSolver solver);

int HYPRE_SStructSplitSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

int HYPRE_SStructSplitSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

int HYPRE_SStructSplitSetTol(HYPRE_SStructSolver solver,
                             double              tol);

int HYPRE_SStructSplitSetMaxIter(HYPRE_SStructSolver solver,
                                 int                 max_iter);

int HYPRE_SStructSplitSetZeroGuess(HYPRE_SStructSolver solver);

int HYPRE_SStructSplitSetNonZeroGuess(HYPRE_SStructSolver solver);

int HYPRE_SStructSplitSetStructSolver(HYPRE_SStructSolver solver,
                                      int                 ssolver);

int HYPRE_SStructSplitGetNumIterations(HYPRE_SStructSolver  solver,
                                       int                 *num_iterations);

int HYPRE_SStructSplitGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct FAC Solver
 **/
                                                                                                            
/**
 * Create a FAC solver object.
 **/
int HYPRE_SStructFACCreate( MPI_Comm             comm,
                             HYPRE_SStructSolver *solver );
/**
 * Destroy a FAC solver object.
 **/
int HYPRE_SStructFACDestroy2( HYPRE_SStructSolver solver );
                                                                                                            
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
 * (Optional) Set max FAC levels.
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
 * (Optional) Use a zero initial guess.
 **/
int HYPRE_SStructFACSetZeroGuess(HYPRE_SStructSolver solver);
                                                                                                            
/**
 * (Optional) Use a nonzero initial guess.
 **/
int HYPRE_SStructFACSetNonZeroGuess(HYPRE_SStructSolver solver);
                                                                                                            
/**
 * (Optional) Set relaxation type.
 **/
int HYPRE_SStructFACSetRelaxType(HYPRE_SStructSolver solver,
                                 int                 relax_type);
/**
 * (Optional) Set number of pre-relaxation sweeps.
 **/
int HYPRE_SStructFACSetNumPreRelax(HYPRE_SStructSolver solver,
                                   int                 num_pre_relax);
                                                                                                            
/**
 * (Optional) Set number of post-relaxation sweeps.
 **/
int HYPRE_SStructFACSetNumPostRelax(HYPRE_SStructSolver solver,
                                    int                 num_post_relax);
/**
 * (Optional) Set coarsest solver type.
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
int HYPRE_SStructFACGetFinalRelativeResidualNorm(
                                          HYPRE_SStructSolver solver,
                                          double             *norm);
                                                                                                            

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*@}*/

#ifdef __cplusplus
}
#endif

#endif

