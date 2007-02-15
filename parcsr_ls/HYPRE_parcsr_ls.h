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
 * $Revision$
 ***********************************************************************EHEADER*/



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
#include "HYPRE_IJ_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Solvers
 *
 * These solvers use matrix/vector storage schemes that are taylored
 * for general sparse matrix systems.
 *
 * @memo Linear solvers for sparse matrix systems
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Solvers
 **/
/*@{*/

struct hypre_Solver_struct;
/**
 * The solver object.
 **/

#ifndef HYPRE_SOLVER_STRUCT
#define HYPRE_SOLVER_STRUCT
struct hypre_Solver_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;
#endif

typedef int (*HYPRE_PtrToParSolverFcn)(HYPRE_Solver,
                                       HYPRE_ParCSRMatrix,
                                       HYPRE_ParVector,
                                       HYPRE_ParVector);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BoomerAMG Solver and Preconditioner
 * 
 * Parallel unstructured algebraic multigrid solver and preconditioner
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_BoomerAMGCreate(HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_BoomerAMGDestroy(HYPRE_Solver solver);

/**
 * Set up the BoomerAMG solver or preconditioner.  
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
int HYPRE_BoomerAMGSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Solve the system or apply AMG as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
int HYPRE_BoomerAMGSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Solve the transpose system $A^T x = b$ or apply AMG as a preconditioner
 * to the transpose system .
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix 
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
int HYPRE_BoomerAMGSolveT(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

/**
 * (Optional) Set the convergence tolerance, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, this function has
 * no effect. The default is 1.e-7.
 **/
int HYPRE_BoomerAMGSetTol(HYPRE_Solver solver,
                          double       tol);

/**
 * (Optional) Sets maximum number of iterations, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, this function has
 * no effect. The default is 20.
 **/
int HYPRE_BoomerAMGSetMaxIter(HYPRE_Solver solver,
                              int          max_iter);

/**
 * (Optional) Sets maximum number of multigrid levels.
 * The default is 25.
 **/
int HYPRE_BoomerAMGSetMaxLevels(HYPRE_Solver solver,
                                int          max_levels);

/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For 2d Laplace operators, 0.25 is a good value, for 3d Laplace
 * operators, 0.5 or 0.6 is a better value. For elasticity problems,
 * a large strength threshold, such as 0.9, is often better.
 **/
int HYPRE_BoomerAMGSetStrongThreshold(HYPRE_Solver solver,
                                      double       strong_threshold);

/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If max\_row\_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
int HYPRE_BoomerAMGSetMaxRowSum(HYPRE_Solver solver,
                                double        max_row_sum);

/**
 * (Optional) Defines which parallel coarsening algorithm is used.
 * There are the following options for coarsen\_type: 
* 
* \begin{tabular}{|c|l|} \hline
 * 0 &	CLJP-coarsening (a parallel coarsening algorithm using independent sets. \\
 * 1 &	classical Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!) \\
 * 3 &	classical Ruge-Stueben coarsening on each processor, followed by a third pass, which adds coarse \\
 * & points on the boundaries \\
 * 6 &   Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points \\
 * & generated by 1 as its first independent set) \\
 * 7 &	CLJP-coarsening (using a fixed random vector, for debugging purposes only) \\
 * 8 &	PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating \\
 * & lower complexities than CLJP, might also lead to slower convergence) \\
 * 9 &	PMIS-coarsening (using a fixed random vector, for debugging purposes only) \\
 * 10 &	HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed \\
 * & by PMIS using the interior C-points generated as its first independent set) \\
 * 11 &	one-pass Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!) \\
 * \hline
 * \end{tabular}
 * 
 * The default is 6. 
 **/
int HYPRE_BoomerAMGSetCoarsenType(HYPRE_Solver solver,
                                  int          coarsen_type);

/**
 * (Optional) Defines whether local or global measures are used.
 **/
int HYPRE_BoomerAMGSetMeasureType(HYPRE_Solver solver,
                                  int          measure_type);

/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set cycle\_type to 1, for a W-cycle
 *  set cycle\_type to 2. The default is 1.
 **/
int HYPRE_BoomerAMGSetCycleType(HYPRE_Solver solver,
                                int          cycle_type);

/**
 * (Optional) Defines the number of sweeps for the fine and coarse grid, 
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetNumSweeps or HYPRE\_BoomerAMGSetCycleNumSweeps instead.
 **/
int HYPRE_BoomerAMGSetNumGridSweeps(HYPRE_Solver  solver,
                                    int          *num_grid_sweeps);

/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and 
 * the down cycle the number of sweeps are set to num\_sweeps and on the 
 * coarsest level to 1. The default is 1.
 **/
int HYPRE_BoomerAMGSetNumSweeps(HYPRE_Solver  solver,
                                int           num_sweeps);

/**
 * (Optional) Sets the number of sweeps at a specified cycle.
 * There are the following options for k:
 *
 * \begin{tabular}{|l|l|} \hline
 * the finest level &	if k=0 \\
 * the down cycle &	if k=1 \\
 * the up cycle	&	if k=2 \\
 * the coarsest level &  if k=3.\\
 * \hline
 * \end{tabular}
 **/
int HYPRE_BoomerAMGSetCycleNumSweeps(HYPRE_Solver  solver,
                                     int           num_sweeps,
                                     int           k);

/**
 * (Optional) Defines which smoother is used on the fine and coarse grid, 
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetRelaxType or HYPRE\_BoomerAMGSetCycleRelaxType instead.
 **/
int HYPRE_BoomerAMGSetGridRelaxType(HYPRE_Solver  solver,
                                    int          *grid_relax_type);

/**
 * (Optional) Defines the smoother to be used. It uses the given
 * smoother on the fine grid, the up and 
 * the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is Gauss-Seidel (3).
 *
 * There are the following options for relax\_type:
 *
 * \begin{tabular}{|c|l|} \hline
 * 0 &	Jacobi \\
 * 1 &	Gauss-Seidel, sequential (very slow!) \\
 * 2 &	Gauss-Seidel, interior points in parallel, boundary sequential (slow!) \\
 * 3 &	hybrid Gauss-Seidel or SOR, forward solve \\
 * 4 &	hybrid Gauss-Seidel or SOR, backward solve \\
 * 5 &	hybrid chaotic Gauss-Seidel (works only with OpenMP) \\
 * 6 &	hybrid symmetric Gauss-Seidel or SSOR \\
 * 9 &	Gaussian elimination (only on coarsest level) \\
 * \hline
 * \end{tabular}
 **/
int HYPRE_BoomerAMGSetRelaxType(HYPRE_Solver  solver,
                                int           relax_type);

/**
 * (Optional) Defines the smoother at a given cycle.
 * For options of relax\_type see
 * description of HYPRE\_BoomerAMGSetRelaxType). Options for k are
 *
 * \begin{tabular}{|l|l|} \hline
 * the finest level &	if k=0 \\
 * the down cycle &	if k=1 \\
 * the up cycle	&	if k=2 \\
 * the coarsest level &  if k=3. \\
 * \hline
 * \end{tabular}
 **/
int HYPRE_BoomerAMGSetCycleRelaxType(HYPRE_Solver  solver,
                                     int           relax_type,
                                     int           k);

/**
 * (Optional) Defines in which order the points are relaxed. There are
 * the following options for
 * relax\_order: 
 *
 * \begin{tabular}{|c|l|} \hline
 * 0 & the points are relaxed in natural or lexicographic
 *                   order on each processor \\
 * 1 &  CF-relaxation is used, i.e on the fine grid and the down
 *                   cycle the coarse points are relaxed first, \\
 * & followed by the fine points; on the up cycle the F-points are relaxed
 * first, followed by the C-points. \\
 * & On the coarsest level, if an iterative scheme is used, 
 * the points are relaxed in lexicographic order. \\
 * \hline
 * \end{tabular}
 *
 * The default is 1 (CF-relaxation).
 **/
int HYPRE_BoomerAMGSetRelaxOrder(HYPRE_Solver  solver,
                                 int           relax_order);

/**
 * (Optional) Defines in which order the points are relaxed. 
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetRelaxOrder instead.
 **/
int HYPRE_BoomerAMGSetGridRelaxPoints(HYPRE_Solver   solver,
                                      int          **grid_relax_points);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetRelaxWt or HYPRE\_BoomerAMGSetLevelRelaxWt instead.
 **/
int HYPRE_BoomerAMGSetRelaxWeight(HYPRE_Solver  solver,
                                  double       *relax_weight); 
/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR 
 * on all levels. 
 * 
 * \begin{tabular}{|l|l|} \hline
 * relax\_weight > 0 & this assigns the given relaxation weight on all levels \\
 * relax\_weight = 0 &  the weight is determined on each level
 *                       with the estimate $3 \over {4\|D^{-1/2}AD^{-1/2}\|}$,\\
 * & where $D$ is the diagonal matrix of $A$ (this should only be used with Jacobi) \\
 * relax\_weight = -k & the relaxation weight is determined with at most k CG steps
 *                       on each level \\
 * & this should only be used for symmetric positive definite problems) \\
 * \hline
 * \end{tabular} 
 * 
 * The default is 1.
 **/
int HYPRE_BoomerAMGSetRelaxWt(HYPRE_Solver  solver,
                              double        relax_weight);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive relax\_weight, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetRelaxWt. 
 * The default is 1.
 **/
int HYPRE_BoomerAMGSetLevelRelaxWt(HYPRE_Solver  solver,
                                   double        relax_weight,
                                   int		 level);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR.
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetOuterWt or HYPRE\_BoomerAMGSetLevelOuterWt instead.
 **/
int HYPRE_BoomerAMGSetOmega(HYPRE_Solver  solver,
                            double       *omega);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR and SSOR
 * on all levels.
 * 
 * \begin{tabular}{|l|l|} \hline 
 * omega > 0 & this assigns the same outer relaxation weight omega on each level\\
 * omega = -k & an outer relaxation weight is determined with at most k CG
 *                steps on each level \\
 * & (this only makes sense for symmetric
 *                positive definite problems and smoothers, e.g. SSOR) \\
 * \hline
 * \end{tabular} 
 * 
 * The default is 1.
 **/
int HYPRE_BoomerAMGSetOuterWt(HYPRE_Solver  solver,
                              double        omega);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetOuterWt. 
 * The default is 1.
 **/
int HYPRE_BoomerAMGSetLevelOuterWt(HYPRE_Solver  solver,
                                   double        omega,
                                   int           level);

/**
 * (Optional)
 **/
int HYPRE_BoomerAMGSetDebugFlag(HYPRE_Solver solver,
                                int          debug_flag);

/**
 * Returns the residual.
 **/
int HYPRE_BoomerAMGGetResidual(HYPRE_Solver  solver,
                               HYPRE_ParVector * residual);

/**
 * Returns the number of iterations taken.
 **/
int HYPRE_BoomerAMGGetNumIterations(HYPRE_Solver  solver,
                                    int          *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
int HYPRE_BoomerAMGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                double       *rel_resid_norm);

/*
 * (Optional)
 **/
int HYPRE_BoomerAMGSetRestriction(HYPRE_Solver solver,
                                  int          restr_par);

/**
 * (Optional) Defines a truncation factor for the interpolation.
 * The default is 0.
 **/
int HYPRE_BoomerAMGSetTruncFactor(HYPRE_Solver solver,
                                  double       trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the interpolation.
 * The default is 0.
 **/
int HYPRE_BoomerAMGSetPMaxElmts(HYPRE_Solver solver,
                                 int       P_max_elmts);

/**
 * (Optional) Defines the largest strength threshold for which 
 * the strength matrix S uses the communication package of the operator A.
 * If the strength threshold is larger than this values,
 * a communication package is generated for S. This can save
 * memory and decrease the amount of data that needs to be communicated,
 * if S is substantially sparser than A.
 * The default is 1.0.
 **/
int HYPRE_BoomerAMGSetSCommPkgSwitch(HYPRE_Solver solver,
                                  double       S_commpkg_switch);

/**
 * (Optional) Defines which parallel interpolation operator is used.
 * There are the following options for interp\_type: 
* 
* \begin{tabular}{|c|l|} \hline
 * 0 &	classical modified interpolation \\
 * 1 &	LS interpolation (for use with GSMG) \\
 * 2 &	classical modified interpolation for hyperbolic PDEs \\
 * 3 &	direct interpolation (with separation of weights) \\
 * 4 &	multipass interpolation \\
 * 5 &	multipass interpolation (with separation of weights) \\
 * 6 &  extended classical modified interpolation \\
 * 7 &  extended (if no common C neighbor) classical modified interpolation \\
 * 8 &	standard interpolation \\
 * 9 &	standard interpolation (with separation of weights) \\
 * 10 &	classical block interpolation (for use with nodal systems version only) \\
 * 11 &	classical block interpolation (for use with nodal systems version only) \\
 *   &	with diagonalized diagonal blocks \\
 * 12 &	FF interpolation \\
 * 13 &	FF1 interpolation \\
 * \hline
 * \end{tabular}
 * 
 * The default is 0. 
 **/
int HYPRE_BoomerAMGSetInterpType(HYPRE_Solver solver,
                                 int          interp_type);

/**
 * (Optional)
 **/
int HYPRE_BoomerAMGSetMinIter(HYPRE_Solver solver,
                              int          min_iter);

/**
 * (Optional) This routine will be eliminated in the future.
 **/
int HYPRE_BoomerAMGInitGridRelaxation(int    **num_grid_sweeps_ptr,
                                      int    **grid_relax_type_ptr,
                                      int   ***grid_relax_points_ptr,
                                      int      coarsen_type,
                                      double **relax_weights_ptr,
                                      int      max_levels);

/**
 * (Optional) Enables the use of more complex smoothers.
 * The following options exist for smooth\_type:
 *
 * \begin{tabular}{|c|l|l|} \hline
 * value & smoother & routines needed to set smoother parameters \\
 * 6 &	Schwarz smoothers & HYPRE\_BoomerAMGSetDomainType, HYPRE\_BoomerAMGSetOverlap, \\
 *  &  & HYPRE\_BoomerAMGSetVariant, HYPRE\_BoomerAMGSetSchwarzRlxWeight \\
 * 7 &	Pilut & HYPRE\_BoomerAMGSetDropTol, HYPRE\_BoomerAMGSetMaxNzPerRow \\
 * 8 &	ParaSails & HYPRE\_BoomerAMGSetSym, HYPRE\_BoomerAMGSetLevel, \\
 * &  &  HYPRE\_BoomerAMGSetFilter, HYPRE\_BoomerAMGSetThreshold \\
 * 9 &	Euclid & HYPRE\_BoomerAMGSetEuclidFile \\
 * \hline
 * \end{tabular}
 *
 * The default is 6. Also, if no smoother parameters are set via the routines mentioned in the table above,
 * default values are used.
 **/
int HYPRE_BoomerAMGSetSmoothType(HYPRE_Solver  solver,
                                 int       smooth_type);

/**
 * (Optional) Sets the number of levels for more complex smoothers.
 * The smoothers, 
 * as defined by HYPRE\_BoomerAMGSetSmoothType, will be used
 * on level 0 (the finest level) through level smooth\_num\_levels-1. 
 * The default is 0, i.e. no complex smoothers are used.
 **/
int HYPRE_BoomerAMGSetSmoothNumLevels(HYPRE_Solver  solver,
                                      int       smooth_num_levels);

/**
 * (Optional) Sets the number of sweeps for more complex smoothers.
 * The default is 1.
 **/
int HYPRE_BoomerAMGSetSmoothNumSweeps(HYPRE_Solver  solver,
                                  int       smooth_num_sweeps);

/*
 * (Optional) Name of file to which BoomerAMG will print;
 * cf HYPRE\_BoomerAMGSetPrintLevel.  (Presently this is ignored).
 **/
int HYPRE_BoomerAMGSetPrintFileName(HYPRE_Solver  solver,
                                  const char   *print_file_name);

/**
 * (Optional) Requests automatic printing of setup and solve information.
 *
 * \begin{tabular}{|c|l|} \hline
 * 0 & no printout (default) \\
 * 1 & print setup information \\
 * 2 & print solve information \\
 * 3 & print both setup and solve information \\
 * \hline
 * \end{tabular}
 *
 * Note, that if one desires to print information and uses BoomerAMG as a 
 * preconditioner, suggested print$\_$level is 1 to avoid excessive output,
 * and use print$\_$level of solver for solve phase information.
 **/
int HYPRE_BoomerAMGSetPrintLevel(HYPRE_Solver  solver,
                              int           print_level);

/**
 * (Optional) Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default to 0 for do nothing.  The latest
 * residual will be available if logging > 1.
 **/
int HYPRE_BoomerAMGSetLogging(HYPRE_Solver  solver,
                              int           logging);

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1.
 **/
int HYPRE_BoomerAMGSetNumFunctions(HYPRE_Solver solver,
                                int          num_functions);

/**
 * (Optional) Sets whether to use the nodal systems version.
 * The default is 0.
 **/
int HYPRE_BoomerAMGSetNodal(HYPRE_Solver solver,
                                int          nodal);
/**
 * (Optional) Sets whether to give spoecial treatment to diagonal elements in 
 * the nodal systems version.
 * The default is 0.
 **/
int HYPRE_BoomerAMGSetNodalDiag(HYPRE_Solver solver,
                                int          nodal_diag);
/**
 * (Optional) Sets the mapping that assigns the function to each variable, 
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
int HYPRE_BoomerAMGSetDofFunc(HYPRE_Solver solver,
                              int         *dof_func);

/**
 * (Optional) Defines the number of levels of aggressive coarsening.
 * The default is 0, i.e. no aggressive coarsening.
 **/
int HYPRE_BoomerAMGSetAggNumLevels(HYPRE_Solver solver,
                                int          agg_num_levels);

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1.
 **/
int HYPRE_BoomerAMGSetNumPaths(HYPRE_Solver solver,
                                int          num_paths);

/**
 * (Optional) Defines which variant of the Schwarz method is used.
 * The following options exist for variant:
 * 
 * \begin{tabular}{|c|l|} \hline
 * 0 & hybrid multiplicative Schwarz method (no overlap across processor 
 *    boundaries) \\
 * 1 & hybrid additive Schwarz method (no overlap across processor 
 *    boundaries) \\
 * 2 & additive Schwarz method \\
 * 3 & hybrid multiplicative Schwarz method (with overlap across processor 
 *    boundaries) \\
 * \hline
 * \end{tabular}
 *
 * The default is 0.
 **/
int HYPRE_BoomerAMGSetVariant(HYPRE_Solver solver,
                                int          variant);

/**
 * (Optional) Defines the overlap for the Schwarz method.
 * The following options exist for overlap:
 *
 * \begin{tabular}{|c|l|} \hline
 * 0  & no overlap \\
 * 1  & minimal overlap (default) \\
 * 2  & overlap generated by including all neighbors of domain boundaries \\
 * \hline
 * \end{tabular}
 **/
int HYPRE_BoomerAMGSetOverlap(HYPRE_Solver solver,
                                int          overlap);

/**
 * (Optional) Defines the type of domain used for the Schwarz method.
 * The following options exist for domain\_type:
 *
 * \begin{tabular}{|c|l|} \hline
 * 0 &  each point is a domain \\
 * 1 &  each node is a domain (only of interest in "systems" AMG) \\
 * 2 &  each domain is generated by agglomeration (default) \\
 * \hline
 * \end{tabular}
 **/
int HYPRE_BoomerAMGSetDomainType(HYPRE_Solver solver,
                                int          domain_type);

/**
 * (Optional) Defines a smoothing parameter for the additive Schwarz method.
 **/
int HYPRE_BoomerAMGSetSchwarzRlxWeight(HYPRE_Solver solver,
                                double    schwarz_rlx_weight);

/**
 * (Optional) Defines symmetry for ParaSAILS. 
 * For further explanation see description of ParaSAILS.
 **/
int HYPRE_BoomerAMGSetSym(HYPRE_Solver solver,
                          int          sym);

/**
 * (Optional) Defines number of levels for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
int HYPRE_BoomerAMGSetLevel(HYPRE_Solver solver,
                            int          level);

/**
 * (Optional) Defines threshold for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
int HYPRE_BoomerAMGSetThreshold(HYPRE_Solver solver,
                                double       threshold);

/**
 * (Optional) Defines filter for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
int HYPRE_BoomerAMGSetFilter(HYPRE_Solver solver,
                             double  	  filter);

/**
 * (Optional) Defines drop tolerance for PILUT.
 * For further explanation see description of PILUT.
 **/
int HYPRE_BoomerAMGSetDropTol(HYPRE_Solver solver,
                              double 	   drop_tol);

/**
 * (Optional) Defines maximal number of nonzeros for PILUT.
 * For further explanation see description of PILUT.
 **/
int HYPRE_BoomerAMGSetMaxNzPerRow(HYPRE_Solver solver,
                          	  int          max_nz_per_row);

/**
 * (Optional) Defines name of an input file for Euclid parameters.
 * For further explanation see description of Euclid.
 **/
int HYPRE_BoomerAMGSetEuclidFile(HYPRE_Solver solver,
                         	 char        *euclidfile); 

/**
 * (Optional) Specifies the use of GSMG - geometrically smooth 
 * coarsening and interpolation. Currently any nonzero value for
 * gsmg will lead to the use of GSMG.
 * The default is 0, i.e. (GSMG is not used)
 **/
int HYPRE_BoomerAMGSetGSMG(HYPRE_Solver solver,
                                int    gsmg);

/**
 * (Optional) Defines the number of sample vectors used in GSMG
 * or LS interpolation.
 **/
int HYPRE_BoomerAMGSetNumSamples(HYPRE_Solver solver,
                                int    num_samples);

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
 **/
int HYPRE_ParaSailsCreate(MPI_Comm      comm,
                          HYPRE_Solver *solver);

/**
 * Destroy a ParaSails preconditioner.
 **/
int HYPRE_ParaSailsDestroy(HYPRE_Solver solver);

/**
 * Set up the ParaSails preconditioner.  This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
int HYPRE_ParaSailsSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Apply the ParaSails preconditioner.  This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to apply.
 * @param A Ignored by this function.
 * @param b [IN] Vector to precondition.
 * @param x [OUT] Preconditioned vector.
 **/
int HYPRE_ParaSailsSolve(HYPRE_Solver       solver,
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
 * @param solver [IN] Preconditioner object for which to set parameters.
 * @param thresh [IN] Value of threshold parameter, $0 \le$ thresh $\le 1$.
 *                    The default value is 0.1.
 * @param nlevels [IN] Value of levels parameter, $0 \le$ nlevels.  
 *                     The default value is 1.
 **/
int HYPRE_ParaSailsSetParams(HYPRE_Solver solver,
                             double       thresh,
                             int          nlevels);
/**
 * Set the filter parameter for the 
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set filter parameter.
 * @param filter [IN] Value of filter parameter.  The filter parameter is
 *                    used to drop small nonzeros in the preconditioner,
 *                    to reduce the cost of applying the preconditioner.
 *                    Values from 0.05 to 0.1 are recommended.
 *                    The default value is 0.1.
 **/
int HYPRE_ParaSailsSetFilter(HYPRE_Solver solver,
                             double       filter);

/**
 * Set the symmetry parameter for the
 * ParaSails preconditioner.
 *
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
int HYPRE_ParaSailsSetSym(HYPRE_Solver solver,
                          int          sym);

/**
 * Set the load balance parameter for the
 * ParaSails preconditioner.
 *
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
int HYPRE_ParaSailsSetLoadbal(HYPRE_Solver solver,
                              double       loadbal);

/**
 * Set the pattern reuse parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set the pattern reuse 
 *                    parameter.
 * @param reuse [IN] Value of the pattern reuse parameter.  A nonzero value
 *                   indicates that the pattern of the preconditioner should
 *                   be reused for subsequent constructions of the 
 *                   preconditioner.  A zero value indicates that the 
 *                   preconditioner should be constructed from scratch.
 *                   The default value when this parameter is not set is 0.
 **/
int HYPRE_ParaSailsSetReuse(HYPRE_Solver solver,
                            int          reuse);

/**
 * Set the logging parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set the logging
 *                    parameter.
 * @param logging [IN] Value of the logging parameter.  A nonzero value
 *                     sends statistics of the setup procedure to stdout.
 *                     The default value when this parameter is not set is 0.
 **/
int HYPRE_ParaSailsSetLogging(HYPRE_Solver solver,
                              int          logging);

/**
 * Build IJ Matrix of the sparse approximate inverse (factor).
 * This function explicitly creates the IJ Matrix corresponding to 
 * the sparse approximate inverse or the inverse factor.
 * Example:  HYPRE\_IJMatrix ij\_A;
 *           HYPRE\_ParaSailsBuildIJMatrix(solver, \&ij\_A);
 *
 * @param solver [IN] Preconditioner object.
 * @param pij_A [OUT] Pointer to the IJ Matrix.
 **/
int HYPRE_ParaSailsBuildIJMatrix(HYPRE_Solver solver, HYPRE_IJMatrix *pij_A);


/*@}*/

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Euclid Preconditioner 
 *
 * MPI Parallel ILU preconditioner 
 *
 * Options summary:
 * \begin{center}
 * \begin{tabular}{|l|c|l|}
 * \hline
 * Option & Default & Synopsis \\
 * \hline
 * -level    & 1 & ILU($k$) factorization level \\ \hline
 * -bj       & 0 (false) & Use Block Jacobi ILU instead of PILU \\ \hline
 * -eu\_stats & 0 (false) & Print  internal timing and statistics \\ \hline
 * -eu\_mem   & 0 (false) & Print  internal memory usage \\ \hline
 * \end{tabular}
 * \end{center}
 *
 **/
/*@{*/

/**
 * Create a Euclid object.
 **/
int HYPRE_EuclidCreate(MPI_Comm      comm,
                             HYPRE_Solver *solver);

/**
 * Destroy a Euclid object.
 **/
int HYPRE_EuclidDestroy(HYPRE_Solver solver);

/**
 * Set up the Euclid preconditioner.  This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
int HYPRE_EuclidSetup(HYPRE_Solver       solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector    b,
                      HYPRE_ParVector    x);

/**
 * Apply the Euclid preconditioner. This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to apply.
 * @param A Ignored by this function.
 * @param b [IN] Vector to precondition.
 * @param x [OUT] Preconditioned vector.
 **/
int HYPRE_EuclidSolve(HYPRE_Solver       solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector    b,
                      HYPRE_ParVector    x);

/**
 * Insert (name, value) pairs in Euclid's options database
 * by passing Euclid the command line (or an array of strings).
 * All Euclid options (e.g, level, drop-tolerance) are stored in
 * this database.  
 * If a (name, value) pair already exists, this call updates the value.
 * See also: HYPRE\_EuclidSetParamsFromFile.
 *
 * @param argc [IN] Length of argv array
 * @param argv [IN] Array of strings
 **/
int HYPRE_EuclidSetParams(HYPRE_Solver solver,
                          int argc,
                          char *argv[]);

/**
 * Insert (name, value) pairs in Euclid's options database.
 * Each line of the file should either begin with a ``\#,''
 * indicating a comment line, or contain a (name value)
 * pair, e.g: \\
 *
 * >cat optionsFile \\
 * \#sample runtime parameter file \\
 * -blockJacobi 3 \\
 * -matFile     /home/hysom/myfile.euclid \\
 * -doSomething true \\
 * -xx\_coeff -1.0
 *
 * See also: HYPRE\_EuclidSetParams.
 *
 * @param filename[IN] Pathname/filename to read
 **/
int HYPRE_EuclidSetParamsFromFile(HYPRE_Solver solver, char *filename);

/*@}*/
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Pilut Preconditioner
 **/
/*@{*/

/**
 * Create a preconditioner object.
 **/
int HYPRE_ParCSRPilutCreate(MPI_Comm      comm,
                            HYPRE_Solver *solver);

/**
 * Destroy a preconditioner object.
 **/
int HYPRE_ParCSRPilutDestroy(HYPRE_Solver solver);

/**
 **/
int HYPRE_ParCSRPilutSetup(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * Precondition the system.
 **/
int HYPRE_ParCSRPilutSolve(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_ParCSRPilutSetMaxIter(HYPRE_Solver solver,
                                int          max_iter);

/**
 * (Optional)
 **/
int HYPRE_ParCSRPilutSetDropTolerance(HYPRE_Solver solver,
                                      double       tol);

/**
 * (Optional)
 **/
int HYPRE_ParCSRPilutSetFactorRowSize(HYPRE_Solver solver,
                                      int          size);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR AMS Solver and Preconditioner
 *
 * Parallel auxiliary space Maxwell solver and preconditioner
 **/
/*@{*/

/**
 * Create an AMS solver object.
 **/
int HYPRE_AMSCreate(HYPRE_Solver *solver);

/**
 * Destroy an AMS solver object.
 **/
int HYPRE_AMSDestroy(HYPRE_Solver solver);

/**
 * Set up the AMS solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
int HYPRE_AMSSetup(HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x);

/**
 * Solve the system or apply AMS as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
int HYPRE_AMSSolve(HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x);

/**
 * (Optional) Sets the problem dimension (2 or 3). The default is 3.
 **/
int HYPRE_AMSSetDimension(HYPRE_Solver solver, int dim);

/**
 * Sets the discrete gradient matrix $G$.
 * This function should be called before HYPRE\_AMSSetup()!
 **/
int HYPRE_AMSSetDiscreteGradient(HYPRE_Solver solver,
                                 HYPRE_ParCSRMatrix G);

/**
 * Sets the $x$, $y$ and $z$ coordinates of the vertices in the mesh.
 *
 * Either HYPRE\_AMSSetCoordinateVectors() or HYPRE\_AMSSetEdgeConstantVectors()
 * should be called before HYPRE\_AMSSetup()!
 **/
int HYPRE_AMSSetCoordinateVectors(HYPRE_Solver solver,
                                  HYPRE_ParVector x,
                                  HYPRE_ParVector y,
                                  HYPRE_ParVector z);

/**
 * Sets the vectors $Gx$, $Gy$ and $Gz$ which give the representations of
 * the constant vector fields $(1,0,0)$, $(0,1,0)$ and $(0,0,1)$ in the
 * edge element basis.
 *
 * Either HYPRE\_AMSSetCoordinateVectors() or HYPRE\_AMSSetEdgeConstantVectors()
 * should be called before HYPRE\_AMSSetup()!
 **/
int HYPRE_AMSSetEdgeConstantVectors(HYPRE_Solver solver,
                                    HYPRE_ParVector Gx,
                                    HYPRE_ParVector Gy,
                                    HYPRE_ParVector Gz);

/**
 * (Optional) Sets the matrix $A_\alpha$ corresponding to the Poisson
 * problem with coefficient $\alpha$ (the curl-curl term coefficient in
 * the Maxwell problem).
 *
 * If this function is called, the coarse space solver on the range
 * of $\Pi^T$ is a block-diagonal version of $A_\Pi$. If this function is not
 * called, the coarse space solver on the range of $\Pi^T$ is constructed
 * as $\Pi^T A \Pi$ in HYPRE\_AMSSetup(). See the user's manual for more details.
 **/
int HYPRE_AMSSetAlphaPoissonMatrix(HYPRE_Solver solver,
                                   HYPRE_ParCSRMatrix A_alpha);

/**
 * (Optional) Sets the matrix $A_\beta$ corresponding to the Poisson
 * problem with coefficient $\beta$ (the mass term coefficient in the
 * Maxwell problem).
 *
 * If not given, the Poisson matrix will be computed in HYPRE\_AMSSetup().
 * If the given matrix is NULL, we assume that $\beta$ is identically $0$
 * and use two-level (instead of three-level) methods. See the user's manual for more details.
 **/
int HYPRE_AMSSetBetaPoissonMatrix(HYPRE_Solver solver,
                                  HYPRE_ParCSRMatrix A_beta);

/**
 * (Optional) Sets maximum number of iterations, if AMS is used
 * as a solver. To use AMS as a preconditioner, set the maximum
 * number of iterations to $1$. The default is $20$.
 **/
int HYPRE_AMSSetMaxIter(HYPRE_Solver solver, int maxit);

/**
 * (Optional) Set the convergence tolerance, if AMS is used
 * as a solver. When using AMS as a preconditioner, set the tolerance
 * to $0.0$. The default is $10^{-6}$.
 **/
int HYPRE_AMSSetTol(HYPRE_Solver solver, double tol);

/**
 * (Optional) Choose which three-level solver to use. Possible values are:
 *
 * \begin{tabular}{|c|l|}
 * \hline
 *   1 & 3-level multiplicative solver (01210) \\
 *   2 & 3-level additive solver (0+1+2) \\
 *   3 & 3-level multiplicative solver (02120) \\
 *   4 & 3-level additive solver (010+2) \\
 *   5 & 3-level multiplicative solver (0102010) \\
 *   6 & 3-level additive solver (1+020) \\
 *   7 & 3-level multiplicative solver (0201020) \\
 *   8 & 3-level additive solver (0(1+2)0) \\
 *  11 & 5-level multiplicative solver (013454310) \\
 *  12 & 5-level additive solver (0+1+3+4+5) \\
 *  13 & 5-level multiplicative solver (034515430) \\
 *  14 & 5-level additive solver (01(3+4+5)10) \\
 * \hline
 * \end{tabular}
 *
 * The default is $1$. See the user's manual for more details.
 **/
int HYPRE_AMSSetCycleType(HYPRE_Solver solver, int cycle_type);

/**
 * (Optional) Control how much information is printed during the
 * solution iterations.
 * The default is $1$ (print residual norm at each step).
 **/
int HYPRE_AMSSetPrintLevel(HYPRE_Solver solver, int print_level);

/**
 * (Optional) Sets relaxation parameters for $A$.
 * The defaults are $2$, $1$, $1.0$, $1.0$.
 **/
int HYPRE_AMSSetSmoothingOptions(HYPRE_Solver solver,
                                 int relax_type,
                                 int relax_times,
                                 double relax_weight,
                                 double omega);

/**
 * (Optional) Sets AMG parameters for $B_\Pi$.
 * The defaults are $10$, $1$, $3$, $0.25$. See the user's manual for more details.
 **/
int HYPRE_AMSSetAlphaAMGOptions(HYPRE_Solver solver,
                                int alpha_coarsen_type,
                                int alpha_agg_levels,
                                int alpha_relax_type,
                                double alpha_strength_threshold);

/**
 * (Optional) Sets AMG parameters for $B_G$.
 * The defaults are $10$, $1$, $3$, $0.25$. See the user's manual for more details.
 **/
int HYPRE_AMSSetBetaAMGOptions(HYPRE_Solver solver,
                               int beta_coarsen_type,
                               int beta_agg_levels,
                               int beta_relax_type,
                               double beta_strength_threshold);

/**
 * Returns the number of iterations taken.
 **/
int HYPRE_AMSGetNumIterations(HYPRE_Solver solver,
                              int *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
int HYPRE_AMSGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                          double *rel_resid_norm);


/**
 * Construct and return the discrete gradient matrix G using some
 * edge and vertex information. We assume that edge\_vertex lists
 * the edge vertices consecutively, and that the orientation of edge i
 * depends only on the sign of edge\_vertex[2*i+1] - edge\_vertex[2*i].
 **/
int HYPRE_AMSConstructDiscreteGradient(HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector x_coord,
				       int *edge_vertex,
                                       HYPRE_ParCSRMatrix *G);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Hybrid Solver
 **/
/*@{*/

/**
  *  Create solver object
 **/
int HYPRE_ParCSRHybridCreate( HYPRE_Solver *solver);
/**
  *  Destroy solver object
 **/
int HYPRE_ParCSRHybridDestroy(HYPRE_Solver solver);

/**
 *  Setup the hybrid solver
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
int HYPRE_ParCSRHybridSetup(HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x);
                                                                                                              
/**
 *  Solve linear system
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
int HYPRE_ParCSRHybridSolve(HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x);
/**
  *  Set the convergence tolerance for the Krylov solver. The default is 1.e-7.
 **/
int HYPRE_ParCSRHybridSetTol(HYPRE_Solver solver,
                             double             tol);
                                                                                                              
/**
  *  Set the desired convergence factor
 **/
int HYPRE_ParCSRHybridSetConvergenceTol(HYPRE_Solver solver,
                                        double             cf_tol);
                                                                                                              
/**
  *  Set the maximal number of iterations for the diagonally
  *  preconditioned solver
 **/
int HYPRE_ParCSRHybridSetDSCGMaxIter(HYPRE_Solver solver,
                                     int                dscg_max_its);
                                                                                                              
/**
  *  Set the maximal number of iterations for the AMG
  *  preconditioned solver
 **/
int HYPRE_ParCSRHybridSetPCGMaxIter(HYPRE_Solver solver,
                                    int                pcg_max_its);

/*
 *
 **/
int HYPRE_ParCSRHybridSetSetupType(HYPRE_Solver solver,
                                    int                setup_type);
                                                                                                              
/**
  *  Set the desired solver type. There are the following options:
  * \begin{tabular}{l l}
  *     1 & PCG (default) \\
  *     2 & GMRES \\
  *     3 & BiCGSTAB
  * \end{tabular}
 **/
int HYPRE_ParCSRHybridSetSolverType(HYPRE_Solver solver,
                                    int                solver_type);
                                                                                                              
/**
  * Set the Krylov dimension for restarted GMRES.
  * The default is 5.
 **/
int HYPRE_ParCSRHybridSetKDim(HYPRE_Solver solver,
                                    int                k_dim);
                                                                                                              
/**
  * Set the type of norm for PCG.
 **/
int HYPRE_ParCSRHybridSetTwoNorm(HYPRE_Solver solver,
                                 int                two_norm);
                                                                                                              
/**
  * Set the choice of stopping criterion for PCG.
 **/
int HYPRE_ParCSRHybridSetStopCrit(HYPRE_Solver solver,
                                 int                stop_crit);
                                                                                                              
/*
 *
 **/
int HYPRE_ParCSRHybridSetRelChange(HYPRE_Solver solver,
                                   int                rel_change);
                                                                                                              
/**
 * Set preconditioner if wanting to use one that is not set up by
 * the hybrid solver.
 **/
int HYPRE_ParCSRHybridSetPrecond(HYPRE_Solver         solver,
                                 HYPRE_PtrToParSolverFcn precond,
                                 HYPRE_PtrToParSolverFcn precond_setup,
                                 HYPRE_Solver         precond_solver);
                    
/**
 * Set logging parameter (default: 0, no logging).
 **/
int HYPRE_ParCSRHybridSetLogging(HYPRE_Solver solver,
                                 int                logging);
                                                                                                              
/**
 * Set print level (default: 0, no printing).
 **/
int HYPRE_ParCSRHybridSetPrintLevel(HYPRE_Solver solver,
                                 int                print_level);
                                                                                                              
/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For 2d Laplace operators, 0.25 is a good value, for 3d Laplace
 * operators, 0.5 or 0.6 is a better value. For elasticity problems,
 * a large strength threshold, such as 0.9, is often better.
 **/
int
HYPRE_ParCSRHybridSetStrongThreshold( HYPRE_Solver solver,
                              double            strong_threshold    );
                                                                                                              
/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If max\_row\_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
int
HYPRE_ParCSRHybridSetMaxRowSum( HYPRE_Solver solver,
                              double             max_row_sum    );
                                                                                                              
/**
 * (Optional) Defines a truncation factor for the interpolation.
 * The default is 0.
 **/
int
HYPRE_ParCSRHybridSetTruncFactor( HYPRE_Solver solver,
                              double              trunc_factor    );
                                                                                                              
/**
 * (Optional) Defines the maximal number of levels used for AMG.
 * The default is 25.
 **/
int
HYPRE_ParCSRHybridSetMaxLevels( HYPRE_Solver solver,
                              int                max_levels    );
                                                                                                              
/**
 * (Optional) Defines whether local or global measures are used.
 **/
int
HYPRE_ParCSRHybridSetMeasureType( HYPRE_Solver solver,
                              int                measure_type    );
                                                                                                              
/**
 * (Optional) Defines which parallel coarsening algorithm is used.
 * There are the following options for coarsen\_type:
*
* \begin{tabular}{|c|l|} \hline
 * 0 &  CLJP-coarsening (a parallel coarsening algorithm using independent sets). \\
 * 1 &  classical Ruge-Stueben coarsening on each processor, no boundary treatment \\
 * 3 &  classical Ruge-Stueben coarsening on each processor, followed by a third \\
 *   &  pass, which adds coarse points on the boundaries \\
 * 6 &  Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse \\
 * & points generated by 1 as its first independent set) \\
 * 7 &  CLJP-coarsening (using a fixed random vector, for debugging purposes only) \\
 * 8 &  PMIS-coarsening (a parallel coarsening algorithm using independent sets \\
 * & with lower complexities than CLJP, might also lead to slower convergence) \\
 * 9 &  PMIS-coarsening (using a fixed random vector, for debugging purposes only) \\
 * 10 & HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, \\ 
 * & followed by PMIS using the interior C-points as its first independent set) \\
 * 11 & one-pass Ruge-Stueben coarsening on each processor, no boundary treatment \\
 * \hline
 * \end{tabular}
 *
 * The default is 6.
 **/
int
HYPRE_ParCSRHybridSetCoarsenType( HYPRE_Solver solver,
                              int                coarsen_type    );
                                                                                                              
/*
 * (Optional) Specifies which interpolation operator is used
 * The default is modified ''classical" interpolation.
 **/
int
HYPRE_ParCSRHybridSetInterpType( HYPRE_Solver solver,
                              int                interp_type    );
                                                                                                              
/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set cycle\_type to 1, for a W-cycle
 *  set cycle\_type to 2. The default is 1.
 **/
int
HYPRE_ParCSRHybridSetCycleType( HYPRE_Solver solver,
                              int                cycle_type    );
                                                                                                              
/*
 *
 **/
int
HYPRE_ParCSRHybridSetGridRelaxType( HYPRE_Solver solver,
                              int               *grid_relax_type    );
                                                                                                              
                                                                                   
/*
 *
 **/
int
HYPRE_ParCSRHybridSetGridRelaxPoints( HYPRE_Solver solver,
                              int              **grid_relax_points    );
                                                                                                              
/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and
 * the down cycle the number of sweeps are set to num\_sweeps and on the
 * coarsest level to 1. The default is 1.
 **/
int
HYPRE_ParCSRHybridSetNumSweeps( HYPRE_Solver solver,
                                int          num_sweeps    );
                                                                                                              
/**
 * (Optional) Sets the number of sweeps at a specified cycle.
 * There are the following options for k:
 *
 * \begin{tabular}{|l|l|} \hline
 * the down cycle &     if k=1 \\
 * the up cycle &       if k=2 \\
 * the coarsest level &  if k=3.\\
 * \hline
 * \end{tabular}
 **/
int
HYPRE_ParCSRHybridSetCycleNumSweeps( HYPRE_Solver solver,
                                     int          num_sweeps,
                                     int          k    );
                                                                                                              
/**
 * (Optional) Defines the smoother to be used. It uses the given
 * smoother on the fine grid, the up and
 * the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is Gauss-Seidel (3).
 *
 * There are the following options for relax\_type:
 *
 * \begin{tabular}{|c|l|} \hline
 * 0 &  Jacobi \\
 * 1 &  Gauss-Seidel, sequential (very slow!) \\
 * 2 &  Gauss-Seidel, interior points in parallel, boundary sequential (slow!) \\
 * 3 &  hybrid Gauss-Seidel or SOR, forward solve \\
 * 4 &  hybrid Gauss-Seidel or SOR, backward solve \\
 * 5 &  hybrid chaotic Gauss-Seidel (works only with OpenMP) \\
 * 6 &  hybrid symmetric Gauss-Seidel or SSOR \\
 * 9 &  Gaussian elimination (only on coarsest level) \\
 * \hline
 * \end{tabular}
 **/
int
HYPRE_ParCSRHybridSetRelaxType( HYPRE_Solver solver,
                                int          relax_type    );
                                                                                                              
/**
 * (Optional) Defines the smoother at a given cycle.
 * For options of relax\_type see
 * description of HYPRE\_BoomerAMGSetRelaxType). Options for k are
 *
 * \begin{tabular}{|l|l|} \hline
 * the down cycle &     if k=1 \\
 * the up cycle &       if k=2 \\
 * the coarsest level &  if k=3. \\
 * \hline
 * \end{tabular}
 **/
int
HYPRE_ParCSRHybridSetCycleRelaxType( HYPRE_Solver solver,
                                     int          relax_type,
                                     int          k   );
                                                                                                              
/**
 * (Optional) Defines in which order the points are relaxed. There are
 * the following options for
 * relax\_order:
 *
 * \begin{tabular}{|c|l|} \hline
 * 0 & the points are relaxed in natural or lexicographic
 *                   order on each processor \\
 * 1 &  CF-relaxation is used, i.e on the fine grid and the down
 *                   cycle the coarse points are relaxed first, \\
 * & followed by the fine points; on the up cycle the F-points are relaxed
 * first, followed by the C-points. \\
 * & On the coarsest level, if an iterative scheme is used,
 * the points are relaxed in lexicographic order. \\
 * \hline
 * \end{tabular}
 *
 * The default is 1 (CF-relaxation).
 **/
int
HYPRE_ParCSRHybridSetRelaxOrder( HYPRE_Solver solver,
                                 int          relax_order    );
                                                                                                              
/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on all levels.
 *
 * \begin{tabular}{|l|l|} \hline
 * relax\_weight > 0 & this assigns the given relaxation weight on all levels \\
 * relax\_weight = 0 &  the weight is determined on each level
 *                       with the estimate $3 \over {4\|D^{-1/2}AD^{-1/2}\|}$,\\
 * & where $D$ is the diagonal matrix of $A$ (this should only be used with Jacobi) \\
 * relax\_weight = -k & the relaxation weight is determined with at most k CG steps
 *                       on each level \\
 * & this should only be used for symmetric positive definite problems) \\
 * \hline
 * \end{tabular}
 *
 * The default is 1.
 **/
int
HYPRE_ParCSRHybridSetRelaxWt( HYPRE_Solver solver,
                              double       relax_wt    );
                                                                                                              
/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive relax\_weight, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetRelaxWt.
 * The default is 1.
 **/
int
HYPRE_ParCSRHybridSetLevelRelaxWt( HYPRE_Solver solver,
                                   double       relax_wt,
                                   int          level    );
                                                                                                              
/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR and SSOR
 * on all levels.
 *
 * \begin{tabular}{|l|l|} \hline
 * omega > 0 & this assigns the same outer relaxation weight omega on each level\\
 * omega = -k & an outer relaxation weight is determined with at most k CG
 *                steps on each level \\
 * & (this only makes sense for symmetric
 *                positive definite problems and smoothers, e.g. SSOR) \\
 * \hline
 * \end{tabular}
 *
 * The default is 1.
 **/
int
HYPRE_ParCSRHybridSetOuterWt( HYPRE_Solver solver,
                              double       outer_wt    );
                                                                                                              
/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetOuterWt.
 * The default is 1.
 **/
int
HYPRE_ParCSRHybridSetLevelOuterWt( HYPRE_Solver solver,
                                   double       outer_wt,
                                   int          level    );
                                                                                                              
/*
 *
 **/
int
HYPRE_ParCSRHybridSetRelaxWeight( HYPRE_Solver solver,
                              double             *relax_weight    );

/*
 *
 **/
int
HYPRE_ParCSRHybridSetOmega( HYPRE_Solver solver,
                              double             *omega    );

/**
 * (Optional) Defines the number of levels of aggressive coarsening,
 * starting with the finest level.
 * The default is 0, i.e. no aggressive coarsening.
 **/
int
HYPRE_ParCSRHybridSetAggNumLevels( HYPRE_Solver solver,
                              int             agg_num_levels    );

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1, which leads to the most aggressive coarsening.
 * Setting num$\_$paths to 2 will increase complexity somewhat,
 * but can lead to better convergence.**/
int
HYPRE_ParCSRHybridSetNumPaths( HYPRE_Solver solver,
                              int             num_paths    );

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1.
 **/
int
HYPRE_ParCSRHybridSetNumFunctions( HYPRE_Solver solver,
                              int             num_functions);

/**
 * (Optional) Sets the mapping that assigns the function to each variable,
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
int
HYPRE_ParCSRHybridSetDofFunc( HYPRE_Solver solver,
                              int            *dof_func    );
/**
 * (Optional) Sets whether to use the nodal systems version.
 * The default is 0 (the unknown based approach).
 **/
int
HYPRE_ParCSRHybridSetNodal( HYPRE_Solver solver,
                              int             nodal    );
                                                                                                              
/**
 * Retrieves the total number of iterations.
 **/
int HYPRE_ParCSRHybridGetNumIterations(HYPRE_Solver  solver,
                                       int                *num_its);
                                                                                                              
/**
 * Retrieves the number of iterations used by the diagonally scaled solver.
 **/
int HYPRE_ParCSRHybridGetDSCGNumIterations(HYPRE_Solver  solver,
                                           int                *dscg_num_its);
                                                                                                              
/**
 * Retrieves the number of iterations used by the AMG preconditioned solver.
 **/
int HYPRE_ParCSRHybridGetPCGNumIterations(HYPRE_Solver  solver,
                                          int                *pcg_num_its);
                                                                                                              
/**
 * Retrieves the final relative residual norm.
 **/
int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(HYPRE_Solver  solver,              double             *norm);
                                                                                          

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR PCG Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_ParCSRPCGCreate(MPI_Comm      comm,
                          HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_ParCSRPCGDestroy(HYPRE_Solver solver);

/**
 **/
int HYPRE_ParCSRPCGSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Solve the system.
 **/
int HYPRE_ParCSRPCGSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_ParCSRPCGSetTol(HYPRE_Solver solver,
                          double       tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_ParCSRPCGSetMaxIter(HYPRE_Solver solver,
                              int          max_iter);

/*
 * RE-VISIT
 **/
int HYPRE_ParCSRPCGSetStopCrit(HYPRE_Solver solver,
                               int          stop_crit);

/**
 * (Optional) Use the two-norm in stopping criteria.
 **/
int HYPRE_ParCSRPCGSetTwoNorm(HYPRE_Solver solver,
                              int          two_norm);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
int HYPRE_ParCSRPCGSetRelChange(HYPRE_Solver solver,
                                int          rel_change);

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_ParCSRPCGSetPrecond(HYPRE_Solver         solver,
                              HYPRE_PtrToParSolverFcn precond,
                              HYPRE_PtrToParSolverFcn precond_setup,
                              HYPRE_Solver         precond_solver);

/**
 **/
int HYPRE_ParCSRPCGGetPrecond(HYPRE_Solver  solver,
                              HYPRE_Solver *precond_data);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_ParCSRPCGSetLogging(HYPRE_Solver solver,
                              int          logging);

/**
 * (Optional) Set the print level
 **/
int HYPRE_ParCSRPCGSetPrintLevel(HYPRE_Solver solver,
                              int          print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_ParCSRPCGGetNumIterations(HYPRE_Solver  solver,
                                    int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                double       *norm);

/**
 * Setup routine for diagonal preconditioning.
 **/
int HYPRE_ParCSRDiagScaleSetup(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    y,
                               HYPRE_ParVector    x);

/**
 * Solve routine for diagonal preconditioning.
 **/
int HYPRE_ParCSRDiagScale(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix HA,
                          HYPRE_ParVector    Hy,
                          HYPRE_ParVector    Hx);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR GMRES Solver
 **/
/*@{*/

/**
 * Create a solver object.
 **/
int HYPRE_ParCSRGMRESCreate(MPI_Comm      comm,
                            HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_ParCSRGMRESDestroy(HYPRE_Solver solver);

/**
 **/
int HYPRE_ParCSRGMRESSetup(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * Solve the system.
 **/
int HYPRE_ParCSRGMRESSolve(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * (Optional) Set the maximum size of the Krylov space.
 **/
int HYPRE_ParCSRGMRESSetKDim(HYPRE_Solver solver,
                             int          k_dim);

/**
 * (Optional) Set the convergence tolerance.
 **/
int HYPRE_ParCSRGMRESSetTol(HYPRE_Solver solver,
                            double       tol);

/*
 * RE-VISIT
 **/
int HYPRE_ParCSRGMRESSetMinIter(HYPRE_Solver solver,
                                int          min_iter);

/**
 * (Optional) Set maximum number of iterations.
 **/
int HYPRE_ParCSRGMRESSetMaxIter(HYPRE_Solver solver,
                                int          max_iter);

/*
 * RE-VISIT
 **/
int HYPRE_ParCSRGMRESSetStopCrit(HYPRE_Solver solver,
                                 int          stop_crit);

/**
 * (Optional) Set the preconditioner to use.
 **/
int HYPRE_ParCSRGMRESSetPrecond(HYPRE_Solver          solver,
                                HYPRE_PtrToParSolverFcn  precond,
                                HYPRE_PtrToParSolverFcn  precond_setup,
                                HYPRE_Solver          precond_solver);

/**
 **/
int HYPRE_ParCSRGMRESGetPrecond(HYPRE_Solver  solver,
                                HYPRE_Solver *precond_data);

/**
 * (Optional) Set the amount of logging to do.
 **/
int HYPRE_ParCSRGMRESSetLogging(HYPRE_Solver solver,
                                int          logging);

/**
 * (Optional) Set print level.
 **/
int HYPRE_ParCSRGMRESSetPrintLevel(HYPRE_Solver solver,
                                int          print_level);

/**
 * Return the number of iterations taken.
 **/
int HYPRE_ParCSRGMRESGetNumIterations(HYPRE_Solver  solver,
                                      int          *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                  double       *norm);

/*@}*/

 /*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BiCGSTAB Solver
 **/
/*@{*/

/**
 * Create a solver object
 **/
int HYPRE_ParCSRBiCGSTABCreate(MPI_Comm      comm,
                               HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
int HYPRE_ParCSRBiCGSTABDestroy(HYPRE_Solver solver);

/**
 * Set up BiCGSTAB solver.
 **/
int HYPRE_ParCSRBiCGSTABSetup(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

/**
 * Solve the linear system.
 **/
int HYPRE_ParCSRBiCGSTABSolve(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

/**
 * (Optional) Set the convergence tolerance (default is 1.e-6).
 **/
int HYPRE_ParCSRBiCGSTABSetTol(HYPRE_Solver solver,
                               double       tol);

/**
 * (Optional) Set the minimal number of iterations (default: 0).
 **/
int HYPRE_ParCSRBiCGSTABSetMinIter(HYPRE_Solver solver,
                                   int          min_iter);

/**
 * (Optional) Set the maximal number of iterations allowed (default: 1000).
 **/
int HYPRE_ParCSRBiCGSTABSetMaxIter(HYPRE_Solver solver,
                                   int          max_iter);

/**
 * (Optional) If stop$\_$crit = 1, the absolute residual norm is used
 *  for the stopping criterion. The default is the relative residual
 *  norm (stop$\_$crit = 0).
 **/
int HYPRE_ParCSRBiCGSTABSetStopCrit(HYPRE_Solver solver,
                                    int          stop_crit);

/**
 * (Optional) Set the preconditioner. 
 **/
int HYPRE_ParCSRBiCGSTABSetPrecond(HYPRE_Solver         solver,
                                   HYPRE_PtrToParSolverFcn precond,
                                   HYPRE_PtrToParSolverFcn precond_setup,
                                   HYPRE_Solver         precond_solver);

/**
 * Get the preconditioner object.
 **/
int HYPRE_ParCSRBiCGSTABGetPrecond(HYPRE_Solver  solver,
                                   HYPRE_Solver *precond_data);

/**
 * (Optional) Set the amount of logging to be done. The default is 0, i.e.
 * no logging.
 **/
int HYPRE_ParCSRBiCGSTABSetLogging(HYPRE_Solver solver,
                                   int          logging);

/**
 * (Optional) Set the desired print level. The default is 0, i.e. no printing.
 **/
int HYPRE_ParCSRBiCGSTABSetPrintLevel(HYPRE_Solver solver,
                                   int          print_level);

/**
 * Retrieve the number of iterations taken.
 **/
int HYPRE_ParCSRBiCGSTABGetNumIterations(HYPRE_Solver  solver,
                                         int          *num_iterations);

/**
 * Retrieve the final relative residual norm.
 **/
int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                 		     double       *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name Schwarz Solver
 **/

int HYPRE_SchwarzCreate( HYPRE_Solver *solver);

int HYPRE_SchwarzDestroy(HYPRE_Solver solver);

int HYPRE_SchwarzSetup(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

int HYPRE_SchwarzSolve(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

int HYPRE_SchwarzSetVariant(HYPRE_Solver solver, int variant);

int HYPRE_SchwarzSetOverlap(HYPRE_Solver solver, int overlap);

int HYPRE_SchwarzSetDomainType(HYPRE_Solver solver, int domain_type);

int HYPRE_SchwarzSetRelaxWeight(HYPRE_Solver solver, double relax_weight);

int HYPRE_SchwarzSetDomainStructure(HYPRE_Solver solver,
                                   HYPRE_CSRMatrix domain_structure);

int HYPRE_SchwarzSetNumFunctions(HYPRE_Solver solver, int num_functions);

int HYPRE_SchwarzSetDofFunc(HYPRE_Solver solver, int *dof_func);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name ParCSR CGNR Solver
 **/

int HYPRE_ParCSRCGNRCreate(MPI_Comm      comm,
                           HYPRE_Solver *solver);

int HYPRE_ParCSRCGNRDestroy(HYPRE_Solver solver);

int HYPRE_ParCSRCGNRSetup(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

int HYPRE_ParCSRCGNRSolve(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

int HYPRE_ParCSRCGNRSetTol(HYPRE_Solver solver,
                           double       tol);

int HYPRE_ParCSRCGNRSetMinIter(HYPRE_Solver solver,
                               int          min_iter);

int HYPRE_ParCSRCGNRSetMaxIter(HYPRE_Solver solver,
                               int          max_iter);

int HYPRE_ParCSRCGNRSetStopCrit(HYPRE_Solver solver,
                                int          stop_crit);

int HYPRE_ParCSRCGNRSetPrecond(HYPRE_Solver         solver,
                               HYPRE_PtrToParSolverFcn precond,
                               HYPRE_PtrToParSolverFcn precondT,
                               HYPRE_PtrToParSolverFcn precond_setup,
                               HYPRE_Solver         precond_solver);

int HYPRE_ParCSRCGNRGetPrecond(HYPRE_Solver  solver,
                               HYPRE_Solver *precond_data);

int HYPRE_ParCSRCGNRSetLogging(HYPRE_Solver solver,
                               int          logging);

int HYPRE_ParCSRCGNRGetNumIterations(HYPRE_Solver  solver,
                                     int          *num_iterations);

int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                 double       *norm);

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

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

HYPRE_ParCSRMatrix GenerateLaplacian9pt(MPI_Comm comm,
                                        int      nx,
                                        int      ny,
                                        int      P,
                                        int      Q,
                                        int      p,
                                        int      q,
                                        double  *value);

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

HYPRE_ParCSRMatrix
GenerateRotate7pt( MPI_Comm comm,
                      int      nx,
                      int      ny,
                      int      P,
                      int      Q,
                      int      p,
                      int      q,
                      double   alpha,
                      double   eps );
                                                                                
HYPRE_ParCSRMatrix
GenerateVarDifConv( MPI_Comm comm,
                 int      nx,
                 int      ny,
                 int      nz,
                 int      P,
                 int      Q,
                 int      R,
                 int      p,
                 int      q,
                 int      r,
                 double eps,
                 HYPRE_ParVector *rhs_ptr);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*@}*/

/*
 * ParCSR ParaSails Preconditioner
 *
 * Parallel sparse approximate inverse preconditioner for the
 * ParCSR matrix format.
 **/

/*
 * Create a ParaSails preconditioner.
 **/
int HYPRE_ParCSRParaSailsCreate(MPI_Comm      comm,
                                HYPRE_Solver *solver);

/*
 * Destroy a ParaSails preconditioner.
 **/
int HYPRE_ParCSRParaSailsDestroy(HYPRE_Solver solver);

/*
 * Set up the ParaSails preconditioner.  This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
int HYPRE_ParCSRParaSailsSetup(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

/*
 * Apply the ParaSails preconditioner.  This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to apply.
 * @param A Ignored by this function.
 * @param b [IN] Vector to precondition.
 * @param x [OUT] Preconditioned vector.
 **/
int HYPRE_ParCSRParaSailsSolve(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

/*
 * Set the threshold and levels parameter for the ParaSails
 * preconditioner.  The accuracy and cost of ParaSails are
 * parameterized by these two parameters.  Lower values of the
 * threshold parameter and higher values of levels parameter 
 * lead to more accurate, but more expensive preconditioners.
 *
 * @param solver [IN] Preconditioner object for which to set parameters.
 * @param thresh [IN] Value of threshold parameter, $0 \le$ thresh $\le 1$.
 *                    The default value is 0.1.
 * @param nlevels [IN] Value of levels parameter, $0 \le$ nlevels.  
 *                     The default value is 1.
 **/
int HYPRE_ParCSRParaSailsSetParams(HYPRE_Solver solver,
                                   double       thresh,
                                   int          nlevels);

/*
 * Set the filter parameter for the 
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set filter parameter.
 * @param filter [IN] Value of filter parameter.  The filter parameter is
 *                    used to drop small nonzeros in the preconditioner,
 *                    to reduce the cost of applying the preconditioner.
 *                    Values from 0.05 to 0.1 are recommended.
 *                    The default value is 0.1.
 **/
int HYPRE_ParCSRParaSailsSetFilter(HYPRE_Solver solver,
                                   double       filter);

/*
 * Set the symmetry parameter for the
 * ParaSails preconditioner.
 *
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

/*
 * Set the load balance parameter for the
 * ParaSails preconditioner.
 *
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

/*
 * Set the pattern reuse parameter for the
 * ParaSails preconditioner.
 *
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

/*
 * Set the logging parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set the logging
 *                    parameter.
 * @param logging [IN] Value of the logging parameter.  A nonzero value
 *                     sends statistics of the setup procedure to stdout.
 *                     The default value when this parameter is not set is 0.
 **/
int HYPRE_ParCSRParaSailsSetLogging(HYPRE_Solver solver,
                                    int          logging);
/*
 * @name ParCSRHybrid Solver
 **/
 
int HYPRE_ParCSRHybridCreate( HYPRE_Solver *solver);
 
int HYPRE_ParCSRHybridDestroy(HYPRE_Solver solver);
 
int HYPRE_ParCSRHybridSetup(HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x); 
 
int HYPRE_ParCSRHybridSolve(HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x);
 
int HYPRE_ParCSRHybridSetTol(HYPRE_Solver solver,
                             double             tol);
 
int HYPRE_ParCSRHybridSetConvergenceTol(HYPRE_Solver solver,
                                        double             cf_tol);
 
int HYPRE_ParCSRHybridSetDSCGMaxIter(HYPRE_Solver solver,
                                     int                dscg_max_its);
 
int HYPRE_ParCSRHybridSetPCGMaxIter(HYPRE_Solver solver,
                                    int                pcg_max_its);
 
int HYPRE_ParCSRHybridSetSetupType(HYPRE_Solver solver,
                                    int                setup_type);
 
int HYPRE_ParCSRHybridSetSolverType(HYPRE_Solver solver,
                                    int                solver_type);
 
int HYPRE_ParCSRHybridSetKDim(HYPRE_Solver solver,
                                    int                k_dim);
 
int HYPRE_ParCSRHybridSetTwoNorm(HYPRE_Solver solver,
                                 int                two_norm);
 
int HYPRE_ParCSRHybridSetStopCrit(HYPRE_Solver solver,
                                 int                stop_crit);
 
int HYPRE_ParCSRHybridSetRelChange(HYPRE_Solver solver,
                                   int                rel_change); 
 
int HYPRE_ParCSRHybridSetPrecond(HYPRE_Solver         solver,
                                 HYPRE_PtrToParSolverFcn precond,
                                 HYPRE_PtrToParSolverFcn precond_setup,
                                 HYPRE_Solver         precond_solver);
 
int HYPRE_ParCSRHybridSetLogging(HYPRE_Solver solver,
                                 int                logging);

int HYPRE_ParCSRHybridSetPrintLevel(HYPRE_Solver solver,
                                 int                print_level);

int
HYPRE_ParCSRHybridSetPrintLevel( HYPRE_Solver solver,
                              int               print_level    );
 
int
HYPRE_ParCSRHybridSetStrongThreshold( HYPRE_Solver solver,
                              double            strong_threshold    );
 
int
HYPRE_ParCSRHybridSetMaxRowSum( HYPRE_Solver solver,
                              double             max_row_sum    );
 
int
HYPRE_ParCSRHybridSetTruncFactor( HYPRE_Solver solver,
                              double              trunc_factor    );
 
int
HYPRE_ParCSRHybridSetMaxLevels( HYPRE_Solver solver,
                              int                max_levels    );
 
int
HYPRE_ParCSRHybridSetMeasureType( HYPRE_Solver solver,
                              int                measure_type    );
 
int
HYPRE_ParCSRHybridSetCoarsenType( HYPRE_Solver solver,
                              int                coarsen_type    );
 
int
HYPRE_ParCSRHybridSetInterpType( HYPRE_Solver solver,
                              int                interp_type    );
 
int
HYPRE_ParCSRHybridSetCycleType( HYPRE_Solver solver,
                              int                cycle_type    );
 
int
HYPRE_ParCSRHybridSetNumGridSweeps( HYPRE_Solver solver,
                              int               *num_grid_sweeps    );
 
int
HYPRE_ParCSRHybridSetGridRelaxType( HYPRE_Solver solver,
                              int               *grid_relax_type    );
 
int
HYPRE_ParCSRHybridSetGridRelaxPoints( HYPRE_Solver solver,
                              int              **grid_relax_points    );
 
int
HYPRE_ParCSRHybridSetNumSweeps( HYPRE_Solver solver,
                                int          num_sweeps    );
 
int
HYPRE_ParCSRHybridSetCycleNumSweeps( HYPRE_Solver solver,
                                     int          num_sweeps,
                                     int          k    );
 
int
HYPRE_ParCSRHybridSetRelaxType( HYPRE_Solver solver,
                                int          relax_type    );
 
int
HYPRE_ParCSRHybridSetCycleRelaxType( HYPRE_Solver solver,
                                     int          relax_type,
                                     int          k   );
 
int
HYPRE_ParCSRHybridSetRelaxOrder( HYPRE_Solver solver,
                                 int          relax_order    );

int
HYPRE_ParCSRHybridSetRelaxWt( HYPRE_Solver solver,
                              double       relax_wt    );
 
int
HYPRE_ParCSRHybridSetLevelRelaxWt( HYPRE_Solver solver,
                                   double       relax_wt,
                                   int          level    );
 
int
HYPRE_ParCSRHybridSetOuterWt( HYPRE_Solver solver,
                              double       outer_wt    );
 
int
HYPRE_ParCSRHybridSetLevelOuterWt( HYPRE_Solver solver,
                                   double       outer_wt,
                                   int          level    );
 
int
HYPRE_ParCSRHybridSetRelaxWeight( HYPRE_Solver solver,
                              double             *relax_weight    );
int
HYPRE_ParCSRHybridSetOmega( HYPRE_Solver solver,
                              double             *omega    );
int
HYPRE_ParCSRHybridSetAggNumLevels( HYPRE_Solver solver,
                              int             agg_num_levels    );
int
HYPRE_ParCSRHybridSetNumPaths( HYPRE_Solver solver,
                              int             num_paths    );
int
HYPRE_ParCSRHybridSetNumFunctions( HYPRE_Solver solver,
                              int             num_functions    );
int
HYPRE_ParCSRHybridSetDofFunc( HYPRE_Solver solver,
                              int            *dof_func    );
int
HYPRE_ParCSRHybridSetNodal( HYPRE_Solver solver,
                              int             nodal    );
 
int HYPRE_ParCSRHybridGetNumIterations(HYPRE_Solver  solver,
                                       int                *num_its);
 
int HYPRE_ParCSRHybridGetDSCGNumIterations(HYPRE_Solver  solver,
                                           int                *dscg_num_its);
 
int HYPRE_ParCSRHybridGetPCGNumIterations(HYPRE_Solver  solver,
                                          int                *pcg_num_its);
 
int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(HYPRE_Solver  solver,                                                   double             *norm); 

/*
 * (Optional) Switches on use of Jacobi interpolation after computing
 * an original interpolation
 **/
int HYPRE_BoomerAMGSetPostInterpType(HYPRE_Solver solver,
                                int          post_interp_type);

/*
 * (Optional) Sets a truncation threshold for Jacobi interpolation.
 **/
int HYPRE_BoomerAMGSetJacobiTruncThreshold(HYPRE_Solver solver,
                                double          jacobi_trunc_threshold);

/*
 * (Optional) Defines the number of relaxation steps for CR
 * The default is 2.
 **/
int HYPRE_BoomerAMGSetNumCRRelaxSteps(HYPRE_Solver solver,
                                int          num_CR_relax_steps);

/*
 * (Optional) Defines convergence rate for CR
 * The default is 0.7.
 **/
int HYPRE_BoomerAMGSetCRRate(HYPRE_Solver solver,
                             double  CR_rate);

/*
 * (Optional) Defines the Type of independent set algorithm used for CR
 **/
int HYPRE_BoomerAMGSetISType(HYPRE_Solver solver,
                                int          IS_type);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/ 

#ifdef __cplusplus
}
#endif

#endif
