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

#ifndef HYPRE_PARCSR_LS_HEADER
#define HYPRE_PARCSR_LS_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_seq_mv.h"
#include "HYPRE_parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Linear Solvers Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A linear solver interface for unstructured grids
 * @version 1.0
 * @author Robert D. Falgout
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Solvers and Preconditioners Type
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt HYPRE\_Solver} object.
 **/
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR ParaSails Preconditioner
 *
 * Parallel sparse approximate inverse preconditioner for the
 * ParCSR matrix format.
 **/
/*@{*/

/**
 * Create a ParaSails preconditioner.
 *
 * @return Error code.
 * @param comm [IN] MPI Communicator.
 * @param solver [OUT] solver object.
 **/
int HYPRE_ParCSRParaSailsCreate(MPI_Comm      comm,
                                HYPRE_Solver *solver);

/**
 * Destroy a ParaSails preconditioner.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object to be destroyed.
 **/
int HYPRE_ParCSRParaSailsDestroy(HYPRE_Solver solver);

/**
 * Set up the ParaSails preconditioner.  This function should be passed
 * to the iterative solver SetPrecond function.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
int HYPRE_ParCSRParaSailsSetup(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

/**
 * Apply the ParaSails preconditioner.  This function should be passed
 * to the iterative solver SetPrecond function.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object to apply.
 * @param A Ignored by this function.
 * @param b [IN] Vector to precondition.
 * @param x [OUT] Preconditioned vector.
 **/
int HYPRE_ParCSRParaSailsSolve(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

/**
 * Set the threshold and levels parameter for the ParaSails
 * preconditioner.  The accuracy and cost of ParaSails are
 * parameterized by these two parameters.  Lower values of the
 * threshold parameter and higher values of levels parameter 
 * lead to more accurate, but more expensive preconditioners.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object for which to set parameters.
 * @param thresh [IN] Value of threshold parameter, $0 \le$ thresh $\le 1$.
 *                    The default value is 0.1.
 * @param nlevels [IN] Value of levels parameter, $0 \le$ nlevels.  
 *                     The default value is 1.
 **/
int HYPRE_ParCSRParaSailsSetParams(HYPRE_Solver solver,
                                   double       thresh,
                                   int          nlevels);

/**
 * Set the filter parameter for the 
 * ParaSails preconditioner.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object for which to set filter parameter.
 * @param filter [IN] Value of filter parameter.  The filter parameter is
 *                    used to drop small nonzeros in the preconditioner,
 *                    to reduce the cost of applying the preconditioner.
 *                    Values from 0.05 to 0.1 are recommended.
 *                    The default value is 0.1.
 **/
int HYPRE_ParCSRParaSailsSetFilter(HYPRE_Solver solver,
                                   double       filter);

/**
 * Set the symmetry parameter for the
 * ParaSails preconditioner.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object for which to set symmetry parameter.
 * @param sym [IN] Value of the symmetry parameter:
 * \begin{tabular}{|c|l|} \hline
 * value & meaning \\ \hline
 * 0 & nonsymmetric and/or indefinite problem, and nonsymmetric preconditioner\\
 * 1 & SPD problem, and SPD (factored) preconditioner \\
 * 2 & nonsymmetric, definite problem, and SPD (factored) preconditioner \\
 * \hline
 * \end{tabular}
 **/
int HYPRE_ParCSRParaSailsSetSym(HYPRE_Solver solver,
                                int          sym);

/**
 * Set the load balance parameter for the
 * ParaSails preconditioner.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object for which to set the load balance
 *                    parameter.
 * @param loadbal [IN] Value of the load balance parameter, 
 *                     $0 \le$ loadbal $\le 1$.  A zero value indicates that
 *                     no load balance is attempted; a value of unity indicates
 *                     that perfect load balance will be attempted.  The 
 *                     recommended value is 0.9 to balance the overhead of
 *                     data exchanges for load balancing.  No load balancing
 *                     is needed if the preconditioner is very sparse and
 *                     fast to construct.  The default value when this 
 *                     parameter is not set is 0.
 **/
int HYPRE_ParCSRParaSailsSetLoadbal(HYPRE_Solver solver,
                                    double       loadbal);

/**
 * Set the pattern reuse parameter for the
 * ParaSails preconditioner.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object for which to set the pattern reuse 
 *                    parameter.
 * @param reuse [IN] Value of the pattern reuse parameter.  A nonzero value
 *                   indicates that the pattern of the preconditioner should
 *                   be reused for subsequent constructions of the 
 *                   preconditioner.  A zero value indicates that the 
 *                   preconditioner should be constructed from scratch.
 *                   The default value when this parameter is not set is 0.
 **/
int HYPRE_ParCSRParaSailsSetReuse(HYPRE_Solver solver,
                                  int          reuse);

/**
 * Set the logging parameter for the
 * ParaSails preconditioner.
 *
 * @return Error code.
 * @param solver [IN] Preconditioner object for which to set the logging
 *                    parameter.
 * @param logging [IN] Value of the logging parameter.  A nonzero value
 *                     sends statistics of the setup procedure to stdout.
 *                     The default value when this parameter is not set is 0.
 **/
int HYPRE_ParCSRParaSailsSetLogging(HYPRE_Solver solver,
                                    int          logging);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BoomerAMG Preconditioner
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGCreate(HYPRE_Solver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGDestroy(HYPRE_Solver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSolveT(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetRestriction(HYPRE_Solver solver,
                                  int          restr_par);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetMaxLevels(HYPRE_Solver solver,
                                int          max_levels);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetStrongThreshold(HYPRE_Solver solver,
                                      double       strong_threshold);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetMaxRowSum(HYPRE_Solver solver,
                                double        max_row_sum);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetTruncFactor(HYPRE_Solver solver,
                                  double       trunc_factor);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetInterpType(HYPRE_Solver solver,
                                 int          interp_type);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetMinIter(HYPRE_Solver solver,
                              int          min_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetMaxIter(HYPRE_Solver solver,
                              int          max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetCoarsenType(HYPRE_Solver solver,
                                  int          coarsen_type);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetMeasureType(HYPRE_Solver solver,
                                  int          measure_type);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetCycleType(HYPRE_Solver solver,
                                int          cycle_type);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetTol(HYPRE_Solver solver,
                          double       tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetNumGridSweeps(HYPRE_Solver  solver,
                                    int          *num_grid_sweeps);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGInitGridRelaxation(int    **num_grid_sweeps_ptr,
                                      int    **grid_relax_type_ptr,
                                      int   ***grid_relax_points_ptr,
                                      int      coarsen_type,
                                      double **relax_weights_ptr,
                                      int      max_levels);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetGridRelaxType(HYPRE_Solver  solver,
                                    int          *grid_relax_type);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetGridRelaxPoints(HYPRE_Solver   solver,
                                      int          **grid_relax_points);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetRelaxWeight(HYPRE_Solver  solver,
                                  double       *relax_weight);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetIOutDat(HYPRE_Solver solver,
                              int          ioutdat);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetLogFileName(HYPRE_Solver  solver,
                                  char         *log_file_name);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetLogging(HYPRE_Solver  solver,
                              int           ioutdat,
                              char         *log_file_name);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGSetDebugFlag(HYPRE_Solver solver,
                                int          debug_flag);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGGetNumIterations(HYPRE_Solver  solver,
                                    int          *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_BoomerAMGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                double       *rel_resid_norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BiCGSTAB Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABCreate(MPI_Comm      comm,
                               HYPRE_Solver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABDestroy(HYPRE_Solver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSetup(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSolve(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSetTol(HYPRE_Solver solver,
                               double       tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSetMinIter(HYPRE_Solver solver,
                                   int          min_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSetMaxIter(HYPRE_Solver solver,
                                   int          max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSetStopCrit(HYPRE_Solver solver,
                                    int          stop_crit);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSetPrecond(HYPRE_Solver solver,
                                   int (*precond)(HYPRE_Solver       sol,
                                                  HYPRE_ParCSRMatrix matrix,
                                                  HYPRE_ParVector    b,
                                                  HYPRE_ParVector    x),
                                   int (*precond_setup)(HYPRE_Solver     sol,
                                                      HYPRE_ParCSRMatrix matrix,
                                                      HYPRE_ParVector    b,
                                                      HYPRE_ParVector    x),
                                   void *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABGetPrecond(HYPRE_Solver  solver,
                                   HYPRE_Solver *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABSetLogging(HYPRE_Solver solver,
                                   int          logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABGetNumIterations(HYPRE_Solver  solver,
                                         int          *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                 		     double       *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR CGN Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRCreate(MPI_Comm      comm,
                           HYPRE_Solver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRDestroy(HYPRE_Solver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSetup(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSolve(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSetTol(HYPRE_Solver solver,
                           double       tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSetMinIter(HYPRE_Solver solver,
                               int          min_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSetMaxIter(HYPRE_Solver solver,
                               int          max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSetStopCrit(HYPRE_Solver solver,
                                int          stop_crit);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSetPrecond(HYPRE_Solver solver,
                               int (*precond)(HYPRE_Solver       sol,
                                              HYPRE_ParCSRMatrix matrix,
                                              HYPRE_ParVector    b,
                                              HYPRE_ParVector    x),
                               int (*precondT)(HYPRE_Solver       sol,
                                               HYPRE_ParCSRMatrix matrix,
                                               HYPRE_ParVector     b,
                                               HYPRE_ParVector     x),
                               int (*precond_setup)(HYPRE_Solver       sol,
                                                    HYPRE_ParCSRMatrix matrix,
                                                    HYPRE_ParVector    b,
                                                    HYPRE_ParVector    x),
                               void *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRGetPrecond(HYPRE_Solver  solver,
                               HYPRE_Solver *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRSetLogging(HYPRE_Solver solver,
                               int          logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRGetNumIterations(HYPRE_Solver  solver,
                                     int          *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                 double       *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR GMRES Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESCreate(MPI_Comm      comm,
                            HYPRE_Solver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESDestroy(HYPRE_Solver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetup(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSolve(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetKDim(HYPRE_Solver solver,
                             int          k_dim);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetTol(HYPRE_Solver solver,
                            double       tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetMinIter(HYPRE_Solver solver,
                                int          min_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetMaxIter(HYPRE_Solver solver,
                                int          max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetStopCrit(HYPRE_Solver solver,
                                 int          stop_crit);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetPrecond(HYPRE_Solver solver,
                                int (*precond)(HYPRE_Solver sol,
                                               HYPRE_ParCSRMatrix matrix,
                                               HYPRE_ParVector b,
                                               HYPRE_ParVector x),
                                int (*precond_setup)(HYPRE_Solver sol,
                                		     HYPRE_ParCSRMatrix matrix,
                                		     HYPRE_ParVector b,
                                		     HYPRE_ParVector x),
                                void *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESGetPrecond(HYPRE_Solver  solver,
                                HYPRE_Solver *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESSetLogging(HYPRE_Solver solver,
                                int          logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESGetNumIterations(HYPRE_Solver  solver,
                                      int          *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                  double       *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR PCG Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGCreate(MPI_Comm      comm,
                          HYPRE_Solver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGDestroy(HYPRE_Solver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetTol(HYPRE_Solver solver,
                          double       tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetMaxIter(HYPRE_Solver solver,
                              int          max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetStopCrit(HYPRE_Solver solver,
                               int          stop_crit);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetTwoNorm(HYPRE_Solver solver,
                              int          two_norm);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetRelChange(HYPRE_Solver solver,
                                int          rel_change);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetPrecond(HYPRE_Solver solver,
                              int (*precond)(HYPRE_Solver       sol,
                                             HYPRE_ParCSRMatrix matrix,
                                             HYPRE_ParVector    b,
                                             HYPRE_ParVector    x),
                              int (*precond_setup)(HYPRE_Solver       sol,
                                                   HYPRE_ParCSRMatrix matrix,
                                                   HYPRE_ParVector    b,
                                                   HYPRE_ParVector    x),
                              void *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGGetPrecond(HYPRE_Solver  solver,
                              HYPRE_Solver *precond_data);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGSetLogging(HYPRE_Solver solver,
                              int          logging);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGGetNumIterations(HYPRE_Solver  solver,
                                    int          *num_iterations);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                double       *norm);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRDiagScaleSetup(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    y,
                               HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRDiagScale(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix HA,
                          HYPRE_ParVector    Hy,
                          HYPRE_ParVector    Hx);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Pilut Preconditioner
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPilutCreate(MPI_Comm      comm,
                            HYPRE_Solver *solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPilutDestroy(HYPRE_Solver solver);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPilutSetup(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPilutSolve(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPilutSetMaxIter(HYPRE_Solver solver,
                                int          max_iter);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPilutSetDropTolerance(HYPRE_Solver solver,
                                      double       tol);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int HYPRE_ParCSRPilutSetFactorRowSize(HYPRE_Solver solver,
                                      int          size);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Utility Functions
 *
 * Description...
 **/
/*@{*/

HYPRE_ParCSRMatrix GenerateLaplacian(MPI_Comm comm,
                                     int      nx,
                                     int      ny,
                                     int      nz,
                                     int      P,
                                     int      Q,
                                     int      R,
                                     int      p,
                                     int      q,
                                     int      r,
                                     double  *value);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int map(int ix,
        int iy,
        int iz,
        int p,
        int q,
        int r,
        int P,
        int Q,
        int R,
        int *nx_part,
        int *ny_part,
        int *nz_part,
        int *global_part);

HYPRE_ParCSRMatrix GenerateLaplacian27pt(MPI_Comm comm,
                                         int      nx,
                                         int      ny,
                                         int      nz,
                                         int      P,
                                         int      Q,
                                         int      R,
                                         int      p,
                                         int      q,
                                         int      r,
                                         double  *value);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int map3(int ix,
         int iy,
         int iz,
         int p,
         int q,
         int r,
         int P,
         int Q,
         int R,
         int *nx_part,
         int *ny_part,
         int *nz_part,
         int *global_part);

HYPRE_ParCSRMatrix GenerateLaplacian9pt(MPI_Comm comm,
                                        int      nx,
                                        int      ny,
                                        int      P,
                                        int      Q,
                                        int      p,
                                        int      q,
                                        double  *value);

/**
 * Description...
 *
 * @param param [IN] ...
 **/
int map2(int ix,
         int iy,
         int p,
         int q,
         int P,
         int Q,
         int *nx_part,
         int *ny_part,
         int *global_part);

HYPRE_ParCSRMatrix GenerateDifConv(MPI_Comm comm,
                                   int      nx,
                                   int      ny,
                                   int      nz,
                                   int      P,
                                   int      Q,
                                   int      R,
                                   int      p,
                                   int      q,
                                   int      r,
                                   double  *value);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif
