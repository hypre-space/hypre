/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.113 $
 ***********************************************************************EHEADER*/


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

typedef HYPRE_Int (*HYPRE_PtrToParSolverFcn)(HYPRE_Solver,
                                       HYPRE_ParCSRMatrix,
                                       HYPRE_ParVector,
                                       HYPRE_ParVector);

#ifndef HYPRE_MODIFYPC
#define HYPRE_MODIFYPC
typedef HYPRE_Int (*HYPRE_PtrToModifyPCFcn)(HYPRE_Solver,
                                      HYPRE_Int,
                                      double);
#endif

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
HYPRE_Int HYPRE_BoomerAMGCreate(HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_BoomerAMGDestroy(HYPRE_Solver solver);

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
HYPRE_Int HYPRE_BoomerAMGSetup(HYPRE_Solver       solver,
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
HYPRE_Int HYPRE_BoomerAMGSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Solve the transpose system $A^T x = b$ or apply AMG as a preconditioner
 * to the transpose system . Note that this function should only be used
 * when preconditioning CGNR with BoomerAMG. It can only be used with
 * Jacobi smoothing (relax_type 0 or 7) and without CF smoothing,
 * i.e relax_order needs to be set to 0.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix 
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
HYPRE_Int HYPRE_BoomerAMGSolveT(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

/**
 * (Optional) Set the convergence tolerance, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, it should be set to 0.
 * The default is 1.e-7.
 **/
HYPRE_Int HYPRE_BoomerAMGSetTol(HYPRE_Solver solver,
                          double       tol);

/**
 * (Optional) Sets maximum number of iterations, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, it should be set to 1.
 * The default is 20.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int          max_iter);

/**
 * (Optional) Sets maximum number of multigrid levels.
 * The default is 25.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxLevels(HYPRE_Solver solver,
                                HYPRE_Int          max_levels);

/**
 * (Optional) Sets maximum size of coarsest grid.
 * The default is 9.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxCoarseSize(HYPRE_Solver solver,
                                    HYPRE_Int          max_coarse_size);

/**
 * (Optional) Sets maximal size for redundant coarse grid solve. 
 * When the system is smaller than this threshold, sequential AMG is used 
 * on all remaining active processors.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSeqThreshold(HYPRE_Solver solver,
                                    HYPRE_Int          seq_threshold);

/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For 2d Laplace operators, 0.25 is a good value, for 3d Laplace
 * operators, 0.5 or 0.6 is a better value. For elasticity problems,
 * a large strength threshold, such as 0.9, is often better.
 **/
HYPRE_Int HYPRE_BoomerAMGSetStrongThreshold(HYPRE_Solver solver,
                                      double       strong_threshold);

/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If max\_row\_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxRowSum(HYPRE_Solver solver,
                                double        max_row_sum);

/**
 * (Optional) Defines which parallel coarsening algorithm is used.
 * There are the following options for coarsen\_type: 
 * 
 * \begin{tabular}{|c|l|} \hline
 * 0 & CLJP-coarsening (a parallel coarsening algorithm using independent sets. \\
 * 1 & classical Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!) \\
 * 3 & classical Ruge-Stueben coarsening on each processor, followed by a third pass, which adds coarse \\
 * & points on the boundaries \\
 * 6 & Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points \\
 * & generated by 1 as its first independent set) \\
 * 7 & CLJP-coarsening (using a fixed random vector, for debugging purposes only) \\
 * 8 & PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating \\
 * & lower complexities than CLJP, might also lead to slower convergence) \\
 * 9 & PMIS-coarsening (using a fixed random vector, for debugging purposes only) \\
 * 10 & HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed \\
 * & by PMIS using the interior C-points generated as its first independent set) \\
 * 11 & one-pass Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!) \\
 * 21 & CGC coarsening by M. Griebel, B. Metsch and A. Schweitzer \\
 * 22 & CGC-E coarsening by M. Griebel, B. Metsch and A.Schweitzer \\
 * \hline
 * \end{tabular}
 * 
 * The default is 6. 
 **/
HYPRE_Int HYPRE_BoomerAMGSetCoarsenType(HYPRE_Solver solver,
                                  HYPRE_Int          coarsen_type);

/**
 * (Optional) Defines whether local or global measures are used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMeasureType(HYPRE_Solver solver,
                                  HYPRE_Int          measure_type);

/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set cycle\_type to 1, for a W-cycle
 *  set cycle\_type to 2. The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCycleType(HYPRE_Solver solver,
                                HYPRE_Int          cycle_type);

/**
 * (Optional) Defines the number of sweeps for the fine and coarse grid, 
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetNumSweeps or HYPRE\_BoomerAMGSetCycleNumSweeps instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumGridSweeps(HYPRE_Solver  solver,
                                    HYPRE_Int          *num_grid_sweeps);

/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and 
 * the down cycle the number of sweeps are set to num\_sweeps and on the 
 * coarsest level to 1. The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumSweeps(HYPRE_Solver  solver,
                                HYPRE_Int           num_sweeps);

/**
 * (Optional) Sets the number of sweeps at a specified cycle.
 * There are the following options for k:
 *
 * \begin{tabular}{|l|l|} \hline
 * the down cycle     & if k=1 \\
 * the up cycle       & if k=2 \\
 * the coarsest level & if k=3.\\
 * \hline
 * \end{tabular}
 **/
HYPRE_Int HYPRE_BoomerAMGSetCycleNumSweeps(HYPRE_Solver  solver,
                                     HYPRE_Int           num_sweeps,
                                     HYPRE_Int           k);

/**
 * (Optional) Defines which smoother is used on the fine and coarse grid, 
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetRelaxType or HYPRE\_BoomerAMGSetCycleRelaxType instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxType(HYPRE_Solver  solver,
                                    HYPRE_Int          *grid_relax_type);

/**
 * (Optional) Defines the smoother to be used. It uses the given
 * smoother on the fine grid, the up and 
 * the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is Gauss-Seidel (3).
 *
 * There are the following options for relax\_type:
 *
 * \begin{tabular}{|c|l|} \hline
 * 0 & Jacobi \\
 * 1 & Gauss-Seidel, sequential (very slow!) \\
 * 2 & Gauss-Seidel, interior points in parallel, boundary sequential (slow!) \\
 * 3 & hybrid Gauss-Seidel or SOR, forward solve \\
 * 4 & hybrid Gauss-Seidel or SOR, backward solve \\
 * 5 & hybrid chaotic Gauss-Seidel (works only with OpenMP) \\
 * 6 & hybrid symmetric Gauss-Seidel or SSOR \\
 * 8 & $\ell_1$-scaled hybrid symmetric Gauss-Seidel\\
 * 9 & Gaussian elimination (only on coarsest level) \\
 * 15 & CG (warning - not a fixed smoother - may require FGMRES)\\
 * 16 & Chebyshev\\
 * 17 & FCF-Jacobi\\                              
 * 18 & $\ell_1$-scaled jacobi\\
 * \hline
 * \end{tabular}
 **/
HYPRE_Int HYPRE_BoomerAMGSetRelaxType(HYPRE_Solver  solver,
                                HYPRE_Int           relax_type);

/**
 * (Optional) Defines the smoother at a given cycle.
 * For options of relax\_type see
 * description of HYPRE\_BoomerAMGSetRelaxType). Options for k are
 *
 * \begin{tabular}{|l|l|} \hline
 * the down cycle     & if k=1 \\
 * the up cycle       & if k=2 \\
 * the coarsest level & if k=3. \\
 * \hline
 * \end{tabular}
 **/
HYPRE_Int HYPRE_BoomerAMGSetCycleRelaxType(HYPRE_Solver  solver,
                                     HYPRE_Int           relax_type,
                                     HYPRE_Int           k);

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
HYPRE_Int HYPRE_BoomerAMGSetRelaxOrder(HYPRE_Solver  solver,
                                 HYPRE_Int           relax_order);

/**
 * (Optional) Defines in which order the points are relaxed. 
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetRelaxOrder instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxPoints(HYPRE_Solver   solver,
                                      HYPRE_Int          **grid_relax_points);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetRelaxWt or HYPRE\_BoomerAMGSetLevelRelaxWt instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetRelaxWeight(HYPRE_Solver  solver,
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
HYPRE_Int HYPRE_BoomerAMGSetRelaxWt(HYPRE_Solver  solver,
                              double        relax_weight);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive relax\_weight, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetRelaxWt. 
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLevelRelaxWt(HYPRE_Solver  solver,
                                   double        relax_weight,
                                   HYPRE_Int           level);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR.
 * Note: This routine will be phased out!!!!
 * Use HYPRE\_BoomerAMGSetOuterWt or HYPRE\_BoomerAMGSetLevelOuterWt instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetOmega(HYPRE_Solver  solver,
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
HYPRE_Int HYPRE_BoomerAMGSetOuterWt(HYPRE_Solver  solver,
                              double        omega);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetOuterWt. 
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLevelOuterWt(HYPRE_Solver  solver,
                                   double        omega,
                                   HYPRE_Int           level);


/**
 * (Optional) Defines the Order for Chebyshev smoother.
 *  The default is 2 (valid options are 1-4).
 **/
HYPRE_Int HYPRE_BoomerAMGSetChebyOrder(HYPRE_Solver solver,
                                 HYPRE_Int          order);

/**
 * (Optional) Fraction of the spectrum to use for the Chebyshev smoother.
 *  The default is .3 (i.e., damp on upper 30% of the spectrum).
 **/
HYPRE_Int HYPRE_BoomerAMGSetChebyFraction (HYPRE_Solver solver,
                                     double         ratio);




/**
 * (Optional)
 **/
HYPRE_Int HYPRE_BoomerAMGSetDebugFlag(HYPRE_Solver solver,
                                HYPRE_Int          debug_flag);

/**
 * Returns the residual.
 **/
HYPRE_Int HYPRE_BoomerAMGGetResidual(HYPRE_Solver     solver,
                               HYPRE_ParVector *residual);

/**
 * Returns the number of iterations taken.
 **/
HYPRE_Int HYPRE_BoomerAMGGetNumIterations(HYPRE_Solver  solver,
                                    HYPRE_Int          *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_BoomerAMGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                double       *rel_resid_norm);

/*
 * (Optional)
 **/
HYPRE_Int HYPRE_BoomerAMGSetRestriction(HYPRE_Solver solver,
                                  HYPRE_Int          restr_par);

/**
 * (Optional) Defines a truncation factor for the interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetTruncFactor(HYPRE_Solver solver,
                                  double       trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetPMaxElmts(HYPRE_Solver solver,
                                HYPRE_Int          P_max_elmts);

/**
 * (Optional) Defines the largest strength threshold for which 
 * the strength matrix S uses the communication package of the operator A.
 * If the strength threshold is larger than this values,
 * a communication package is generated for S. This can save
 * memory and decrease the amount of data that needs to be communicated,
 * if S is substantially sparser than A.
 * The default is 1.0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSCommPkgSwitch(HYPRE_Solver solver,
                                     double       S_commpkg_switch);

/**
 * (Optional) Defines which parallel interpolation operator is used.
 * There are the following options for interp\_type: 
 * 
 * \begin{tabular}{|c|l|} \hline
 * 0  & classical modified interpolation \\
 * 1  & LS interpolation (for use with GSMG) \\
 * 2  & classical modified interpolation for hyperbolic PDEs \\
 * 3  & direct interpolation (with separation of weights) \\
 * 4  & multipass interpolation \\
 * 5  & multipass interpolation (with separation of weights) \\
 * 6  & extended+i interpolation \\
 * 7  & extended+i (if no common C neighbor) interpolation \\
 * 8  & standard interpolation \\
 * 9  & standard interpolation (with separation of weights) \\
 * 10 & classical block interpolation (for use with nodal systems version only) \\
 * 11 & classical block interpolation (for use with nodal systems version only) \\
 *    & with diagonalized diagonal blocks \\
 * 12 & FF interpolation \\
 * 13 & FF1 interpolation \\
 * 14 & extended interpolation \\
 * \hline
 * \end{tabular}
 * 
 * The default is 0. 
 **/
HYPRE_Int HYPRE_BoomerAMGSetInterpType(HYPRE_Solver solver,
                                 HYPRE_Int          interp_type);

/**
 * (Optional) Defines whether separation of weights is used
 * when defining strength for standard interpolation or
 * multipass interpolation.
 * Default: 0, i.e. no separation of weights used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSepWeight(HYPRE_Solver solver,
                                HYPRE_Int          sep_weight);

/**
 * (Optional)
 **/
HYPRE_Int HYPRE_BoomerAMGSetMinIter(HYPRE_Solver solver,
                              HYPRE_Int          min_iter);

/**
 * (Optional) This routine will be eliminated in the future.
 **/
HYPRE_Int HYPRE_BoomerAMGInitGridRelaxation(HYPRE_Int    **num_grid_sweeps_ptr,
                                      HYPRE_Int    **grid_relax_type_ptr,
                                      HYPRE_Int   ***grid_relax_points_ptr,
                                      HYPRE_Int      coarsen_type,
                                      double **relax_weights_ptr,
                                      HYPRE_Int      max_levels);

/**
 * (Optional) Enables the use of more complex smoothers.
 * The following options exist for smooth\_type:
 *
 * \begin{tabular}{|c|l|l|} \hline
 * value & smoother & routines needed to set smoother parameters \\
 * 6 & Schwarz smoothers & HYPRE\_BoomerAMGSetDomainType, HYPRE\_BoomerAMGSetOverlap, \\
 *   &  & HYPRE\_BoomerAMGSetVariant, HYPRE\_BoomerAMGSetSchwarzRlxWeight \\
 * 7 & Pilut & HYPRE\_BoomerAMGSetDropTol, HYPRE\_BoomerAMGSetMaxNzPerRow \\
 * 8 & ParaSails & HYPRE\_BoomerAMGSetSym, HYPRE\_BoomerAMGSetLevel, \\
 *   &  &  HYPRE\_BoomerAMGSetFilter, HYPRE\_BoomerAMGSetThreshold \\
 * 9 & Euclid & HYPRE\_BoomerAMGSetEuclidFile \\
 * \hline
 * \end{tabular}
 *
 * The default is 6. Also, if no smoother parameters are set via the routines mentioned in the table above,
 * default values are used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSmoothType(HYPRE_Solver solver,
                                 HYPRE_Int          smooth_type);

/**
 * (Optional) Sets the number of levels for more complex smoothers.
 * The smoothers, 
 * as defined by HYPRE\_BoomerAMGSetSmoothType, will be used
 * on level 0 (the finest level) through level smooth\_num\_levels-1. 
 * The default is 0, i.e. no complex smoothers are used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumLevels(HYPRE_Solver solver,
                                      HYPRE_Int          smooth_num_levels);

/**
 * (Optional) Sets the number of sweeps for more complex smoothers.
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumSweeps(HYPRE_Solver solver,
                                      HYPRE_Int          smooth_num_sweeps);

/*
 * (Optional) Name of file to which BoomerAMG will print;
 * cf HYPRE\_BoomerAMGSetPrintLevel.  (Presently this is ignored).
 **/
HYPRE_Int HYPRE_BoomerAMGSetPrintFileName(HYPRE_Solver  solver,
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
HYPRE_Int HYPRE_BoomerAMGSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int          print_level);

/**
 * (Optional) Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default to 0 for do nothing.  The latest
 * residual will be available if logging > 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLogging(HYPRE_Solver solver,
                              HYPRE_Int          logging);

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumFunctions(HYPRE_Solver solver,
                                   HYPRE_Int          num_functions);

/**
 * (Optional) Sets whether to use the nodal systems coarsening.
 * The default is 0 (unknown-based coarsening).
 **/
HYPRE_Int HYPRE_BoomerAMGSetNodal(HYPRE_Solver solver,
                            HYPRE_Int          nodal);
/*  Don't want this in manual...
 * (Optional) Sets whether to give spoecial treatment to diagonal elements in 
 * the nodal systems version.
 * The default is 0.
 */
HYPRE_Int HYPRE_BoomerAMGSetNodalDiag(HYPRE_Solver solver,
                                HYPRE_Int          nodal_diag);
/**
 * (Optional) Sets the mapping that assigns the function to each variable, 
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
HYPRE_Int HYPRE_BoomerAMGSetDofFunc(HYPRE_Solver  solver,
                              HYPRE_Int          *dof_func);

/**
 * (Optional) Defines the number of levels of aggressive coarsening.
 * The default is 0, i.e. no aggressive coarsening.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggNumLevels(HYPRE_Solver solver,
                                   HYPRE_Int          agg_num_levels);

/**
 * (Optional) Defines the interpolation used on levels of aggressive coarsening
 * The default is 4, i.e. multipass interpolation.
 * The following options exist:
 * 
 * \begin{tabular}{|c|l|} \hline
 * 1 & 2-stage extended+i interpolation \\
 * 2 & 2-stage standard interpolation \\
 * 3 & 2-stage extended interpolation \\
 * 4 & multipass interpolation \\
 * \hline
 * \end{tabular}
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggInterpType(HYPRE_Solver solver,
                                    HYPRE_Int          agg_interp_type);

/**
 * (Optional) Defines the truncation factor for the 
 * interpolation used for aggressive coarsening.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggTruncFactor(HYPRE_Solver solver,
                                     double       agg_trunc_factor);

/**
 * (Optional) Defines the truncation factor for the 
 * matrices P1 and P2 which are used to build 2-stage interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggP12TruncFactor(HYPRE_Solver solver,
                                        double       agg_P12_trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the 
 * interpolation used for aggressive coarsening.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggPMaxElmts(HYPRE_Solver solver,
                                   HYPRE_Int          agg_P_max_elmts);

/**
 * (Optional) Defines the maximal number of elements per row for the 
 * matrices P1 and P2 which are used to build 2-stage interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggP12MaxElmts(HYPRE_Solver solver,
                                     HYPRE_Int          agg_P12_max_elmts);

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumPaths(HYPRE_Solver solver,
                               HYPRE_Int          num_paths);

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
HYPRE_Int HYPRE_BoomerAMGSetVariant(HYPRE_Solver solver,
                              HYPRE_Int          variant);

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
HYPRE_Int HYPRE_BoomerAMGSetOverlap(HYPRE_Solver solver,
                              HYPRE_Int          overlap);

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
HYPRE_Int HYPRE_BoomerAMGSetDomainType(HYPRE_Solver solver,
                                 HYPRE_Int          domain_type);

/**
 * (Optional) Defines a smoothing parameter for the additive Schwarz method.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSchwarzRlxWeight(HYPRE_Solver solver,
                                       double       schwarz_rlx_weight);

/**
 *  (Optional) Indicates that the aggregates may not be SPD for the Schwarz method.
 * The following options exist for use\_nonsymm:
 *
 * \begin{tabular}{|c|l|} \hline
 * 0  & assume SPD (default) \\
 * 1  & assume non-symmetric \\
 * \hline
 * \end{tabular}
**/
HYPRE_Int HYPRE_BoomerAMGSetSchwarzUseNonSymm(HYPRE_Solver solver,
                                        HYPRE_Int          use_nonsymm);

/**
 * (Optional) Defines symmetry for ParaSAILS. 
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSym(HYPRE_Solver solver,
                          HYPRE_Int          sym);

/**
 * (Optional) Defines number of levels for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLevel(HYPRE_Solver solver,
                            HYPRE_Int          level);

/**
 * (Optional) Defines threshold for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetThreshold(HYPRE_Solver solver,
                                double       threshold);

/**
 * (Optional) Defines filter for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFilter(HYPRE_Solver solver,
                             double       filter);

/**
 * (Optional) Defines drop tolerance for PILUT.
 * For further explanation see description of PILUT.
 **/
HYPRE_Int HYPRE_BoomerAMGSetDropTol(HYPRE_Solver solver,
                              double       drop_tol);

/**
 * (Optional) Defines maximal number of nonzeros for PILUT.
 * For further explanation see description of PILUT.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxNzPerRow(HYPRE_Solver solver,
                                  HYPRE_Int          max_nz_per_row);

/**
 * (Optional) Defines name of an input file for Euclid parameters.
 * For further explanation see description of Euclid.
 **/
HYPRE_Int HYPRE_BoomerAMGSetEuclidFile(HYPRE_Solver  solver,
                                 char         *euclidfile); 

/**
 * (Optional) Defines number of levels for ILU(k) in Euclid.
 * For further explanation see description of Euclid.
 **/
HYPRE_Int HYPRE_BoomerAMGSetEuLevel(HYPRE_Solver solver,
                              HYPRE_Int          eu_level);

/**
 * (Optional) Defines filter for ILU(k) for Euclid.
 * For further explanation see description of Euclid.
 **/
HYPRE_Int HYPRE_BoomerAMGSetEuSparseA(HYPRE_Solver solver,
                                double       eu_sparse_A);

/**
 * (Optional) Defines use of block jacobi ILUT for Euclid.
 * For further explanation see description of Euclid.
 **/
HYPRE_Int HYPRE_BoomerAMGSetEuBJ(HYPRE_Solver solver,
                           HYPRE_Int          eu_bj);

/**
 * (Optional) Specifies the use of GSMG - geometrically smooth 
 * coarsening and interpolation. Currently any nonzero value for
 * gsmg will lead to the use of GSMG.
 * The default is 0, i.e. (GSMG is not used)
 **/
HYPRE_Int HYPRE_BoomerAMGSetGSMG(HYPRE_Solver solver,
                           HYPRE_Int          gsmg);

/**
 * (Optional) Defines the number of sample vectors used in GSMG
 * or LS interpolation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumSamples(HYPRE_Solver solver,
                                 HYPRE_Int          num_samples);
/**
 * (optional) Defines the number of pathes for CGC-coarsening.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCGCIts (HYPRE_Solver solver,
                              HYPRE_Int          its);

/*
 * HYPRE_BoomerAMGSetPlotGrids
 **/
HYPRE_Int HYPRE_BoomerAMGSetPlotGrids (HYPRE_Solver solver,
                                 HYPRE_Int          plotgrids);

/*
 * HYPRE_BoomerAMGSetPlotFilename
 **/
HYPRE_Int HYPRE_BoomerAMGSetPlotFileName (HYPRE_Solver  solver,
                                    const char   *plotfilename);

/*
 * HYPRE_BoomerAMGSetCoordDim
 **/
HYPRE_Int HYPRE_BoomerAMGSetCoordDim (HYPRE_Solver solver,
                                HYPRE_Int          coorddim);

/*
 * HYPRE_BoomerAMGSetCoordinates
 **/
HYPRE_Int HYPRE_BoomerAMGSetCoordinates (HYPRE_Solver  solver,
                                   float        *coordinates);

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
HYPRE_Int HYPRE_ParaSailsCreate(MPI_Comm      comm,
                          HYPRE_Solver *solver);

/**
 * Destroy a ParaSails preconditioner.
 **/
HYPRE_Int HYPRE_ParaSailsDestroy(HYPRE_Solver solver);

/**
 * Set up the ParaSails preconditioner.  This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
HYPRE_Int HYPRE_ParaSailsSetup(HYPRE_Solver       solver,
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
HYPRE_Int HYPRE_ParaSailsSolve(HYPRE_Solver       solver,
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
HYPRE_Int HYPRE_ParaSailsSetParams(HYPRE_Solver solver,
                             double       thresh,
                             HYPRE_Int          nlevels);
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
HYPRE_Int HYPRE_ParaSailsSetFilter(HYPRE_Solver solver,
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
HYPRE_Int HYPRE_ParaSailsSetSym(HYPRE_Solver solver,
                          HYPRE_Int          sym);

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
HYPRE_Int HYPRE_ParaSailsSetLoadbal(HYPRE_Solver solver,
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
HYPRE_Int HYPRE_ParaSailsSetReuse(HYPRE_Solver solver,
                            HYPRE_Int          reuse);

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
HYPRE_Int HYPRE_ParaSailsSetLogging(HYPRE_Solver solver,
                              HYPRE_Int          logging);

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
HYPRE_Int HYPRE_ParaSailsBuildIJMatrix(HYPRE_Solver    solver,
                                 HYPRE_IJMatrix *pij_A);

/* ParCSRParaSails routines */

HYPRE_Int HYPRE_ParCSRParaSailsCreate(MPI_Comm      comm,
                                HYPRE_Solver *solver);

HYPRE_Int HYPRE_ParCSRParaSailsDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRParaSailsSetup(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRParaSailsSolve(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRParaSailsSetParams(HYPRE_Solver solver,
                                   double       thresh,
                                   HYPRE_Int          nlevels);

HYPRE_Int HYPRE_ParCSRParaSailsSetFilter(HYPRE_Solver solver,
                                   double       filter);

HYPRE_Int HYPRE_ParCSRParaSailsSetSym(HYPRE_Solver solver,
                                HYPRE_Int          sym);

HYPRE_Int HYPRE_ParCSRParaSailsSetLoadbal(HYPRE_Solver solver,
                                    double       loadbal);

HYPRE_Int HYPRE_ParCSRParaSailsSetReuse(HYPRE_Solver solver,
                                  HYPRE_Int          reuse);

HYPRE_Int HYPRE_ParCSRParaSailsSetLogging(HYPRE_Solver solver,
                                    HYPRE_Int          logging);

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
HYPRE_Int HYPRE_EuclidCreate(MPI_Comm      comm,
                       HYPRE_Solver *solver);

/**
 * Destroy a Euclid object.
 **/
HYPRE_Int HYPRE_EuclidDestroy(HYPRE_Solver solver);

/**
 * Set up the Euclid preconditioner.  This function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] Preconditioner object to set up.
 * @param A [IN] ParCSR matrix used to construct the preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
HYPRE_Int HYPRE_EuclidSetup(HYPRE_Solver       solver,
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
HYPRE_Int HYPRE_EuclidSolve(HYPRE_Solver       solver,
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
HYPRE_Int HYPRE_EuclidSetParams(HYPRE_Solver  solver,
                          HYPRE_Int           argc,
                          char         *argv[]);

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
HYPRE_Int HYPRE_EuclidSetParamsFromFile(HYPRE_Solver  solver,
                                  char         *filename);


/**
 * Set level k for ILU(k) factorization, default: 1
 **/
HYPRE_Int HYPRE_EuclidSetLevel(HYPRE_Solver solver,
                         HYPRE_Int          level);

/**
 * Use block Jacobi ILU preconditioning instead of PILU
 **/
HYPRE_Int HYPRE_EuclidSetBJ(HYPRE_Solver solver,
                      HYPRE_Int          bj);

/**
 * If eu\_stats not equal 0, a summary of runtime settings and 
 * timing information is printed to stdout.
 **/
HYPRE_Int HYPRE_EuclidSetStats(HYPRE_Solver solver,
                         HYPRE_Int          eu_stats);

/**
 * If eu\_mem not equal 0, a summary of Euclid's memory usage
 * is printed to stdout.
 **/
HYPRE_Int HYPRE_EuclidSetMem(HYPRE_Solver solver,
                       HYPRE_Int          eu_mem);

/**
 * Defines a drop tolerance for ILU(k). Default: 0
 * Use with HYPRE\_EuclidSetRowScale. 
 * Note that this can destroy symmetry in a matrix.
 **/
HYPRE_Int HYPRE_EuclidSetSparseA(HYPRE_Solver solver,
                           double       sparse_A);

/**
 * If row\_scale not equal 0, values are scaled prior to factorization
 * so that largest value in any row is +1 or -1.
 * Note that this can destroy symmetry in a matrix.
 **/
HYPRE_Int HYPRE_EuclidSetRowScale(HYPRE_Solver solver,
                            HYPRE_Int          row_scale);

/**
 * uses ILUT and defines a drop tolerance relative to the largest
 * absolute value of any entry in the row being factored.
 **/
HYPRE_Int HYPRE_EuclidSetILUT(HYPRE_Solver solver,
                        double       drop_tol);

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
HYPRE_Int HYPRE_ParCSRPilutCreate(MPI_Comm      comm,
                            HYPRE_Solver *solver);

/**
 * Destroy a preconditioner object.
 **/
HYPRE_Int HYPRE_ParCSRPilutDestroy(HYPRE_Solver solver);

/**
 **/
HYPRE_Int HYPRE_ParCSRPilutSetup(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * Precondition the system.
 **/
HYPRE_Int HYPRE_ParCSRPilutSolve(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int HYPRE_ParCSRPilutSetMaxIter(HYPRE_Solver solver,
                                HYPRE_Int          max_iter);

/**
 * (Optional)
 **/
HYPRE_Int HYPRE_ParCSRPilutSetDropTolerance(HYPRE_Solver solver,
                                      double       tol);

/**
 * (Optional)
 **/
HYPRE_Int HYPRE_ParCSRPilutSetFactorRowSize(HYPRE_Solver solver,
                                      HYPRE_Int          size);

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
HYPRE_Int HYPRE_AMSCreate(HYPRE_Solver *solver);

/**
 * Destroy an AMS solver object.
 **/
HYPRE_Int HYPRE_AMSDestroy(HYPRE_Solver solver);

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
HYPRE_Int HYPRE_AMSSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

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
HYPRE_Int HYPRE_AMSSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * (Optional) Sets the problem dimension (2 or 3). The default is 3.
 **/
HYPRE_Int HYPRE_AMSSetDimension(HYPRE_Solver solver,
                                HYPRE_Int          dim);

/**
 * Sets the discrete gradient matrix $G$.
 * This function should be called before HYPRE\_AMSSetup()!
 **/
HYPRE_Int HYPRE_AMSSetDiscreteGradient(HYPRE_Solver       solver,
                                       HYPRE_ParCSRMatrix G);

/**
 * Sets the $x$, $y$ and $z$ coordinates of the vertices in the mesh.
 *
 * Either HYPRE\_AMSSetCoordinateVectors() or HYPRE\_AMSSetEdgeConstantVectors()
 * should be called before HYPRE\_AMSSetup()!
 **/
HYPRE_Int HYPRE_AMSSetCoordinateVectors(HYPRE_Solver    solver,
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
HYPRE_Int HYPRE_AMSSetEdgeConstantVectors(HYPRE_Solver    solver,
                                          HYPRE_ParVector Gx,
                                          HYPRE_ParVector Gy,
                                          HYPRE_ParVector Gz);

/**
 * (Optional) Set the (components of) the Nedelec interpolation matrix
 * $\Pi = [ \Pi^x, \Pi^y, \Pi^z ]$.
 *
 * This function is generally intended to be used only for high-order Nedelec
 * discretizations (in the lowest order case, $\Pi$ is constructed internally in
 * AMS from the discreet gradient matrix and the coordinates of the vertices),
 * though it can also be used in the lowest-order case or for other types of
 * discretizations (e.g. ones based on the second family of Nedelec elements).
 *
 * By definition, $\Pi$ is the matrix representation of the linear operator that
 * interpolates (high-order) vector nodal finite elements into the (high-order)
 * Nedelec space. The component matrices are defined as $\Pi^x \varphi = \Pi
 * (\varphi,0,0)$ and similarly for $\Pi^y$ and $\Pi^z$. Note that all these
 * operators depend on the choice of the basis and degrees of freedom in the
 * high-order spaces.
 *
 * The column numbering of Pi should be node-based, i.e. the $x$/$y$/$z$
 * components of the first node (vertex or high-order dof) should be listed
 * first, followed by the $x$/$y$/$z$ components of the second node and so on
 * (see the documentation of HYPRE\_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before HYPRE\_AMSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * $\{\Pi\}$ and $\{\Pi^x,\Pi^y,\Pi^z\}$ needs to be specified (though it is OK
 * to provide both).  If Pix is NULL, then scalar $\Pi$-based AMS cycles,
 * i.e. those with cycle\_type > 10, will be unavailable. Similarly, AMS cycles
 * based on monolithic $\Pi$ (cycle\_type < 10) require that Pi is not NULL.
 **/
HYPRE_Int HYPRE_AMSSetInterpolations(HYPRE_Solver       solver,
                                     HYPRE_ParCSRMatrix Pi,
                                     HYPRE_ParCSRMatrix Pix,
                                     HYPRE_ParCSRMatrix Piy,
                                     HYPRE_ParCSRMatrix Piz);

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
HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix(HYPRE_Solver       solver,
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
HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix(HYPRE_Solver       solver,
                                        HYPRE_ParCSRMatrix A_beta);

/**
 * (Optional) Set the list of nodes which are interior to a zero-conductivity
 * region. This way, a more robust solver is constructed, that can be iterated
 * to lower tolerance levels. This function should be called before
 * HYPRE\_AMSSetup()!
 **/
HYPRE_Int HYPRE_AMSSetInteriorNodes(HYPRE_Solver    solver,
                                    HYPRE_ParVector interior_nodes);

/**
 * (Optional) Set the frequency at which a projection onto the compatible
 * subspace for problems with zero-conductivity regions is performed. The
 * default value is 5.
 **/
HYPRE_Int HYPRE_AMSSetProjectionFrequency(HYPRE_Solver solver,
                                          HYPRE_Int    projection_frequency);

/**
 * (Optional) Sets maximum number of iterations, if AMS is used
 * as a solver. To use AMS as a preconditioner, set the maximum
 * number of iterations to $1$. The default is $20$.
 **/
HYPRE_Int HYPRE_AMSSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int    maxit);

/**
 * (Optional) Set the convergence tolerance, if AMS is used
 * as a solver. When using AMS as a preconditioner, set the tolerance
 * to $0.0$. The default is $10^{-6}$.
 **/
HYPRE_Int HYPRE_AMSSetTol(HYPRE_Solver solver,
                          double       tol);

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
HYPRE_Int HYPRE_AMSSetCycleType(HYPRE_Solver solver,
                                HYPRE_Int    cycle_type);

/**
 * (Optional) Control how much information is printed during the
 * solution iterations.
 * The default is $1$ (print residual norm at each step).
 **/
HYPRE_Int HYPRE_AMSSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int    print_level);

/**
 * (Optional) Sets relaxation parameters for $A$.
 * The defaults are $2$, $1$, $1.0$, $1.0$.
 *
 * The available options for relax\_type are:
 *
 * \begin{tabular}{|c|l|} \hline
 * 1 & $\ell_1$-scaled Jacobi \\
 * 2 & $\ell_1$-scaled block symmetric Gauss-Seidel/SSOR \\
 * 3 & Kaczmarz \\
 * 4 & truncated version of $\ell_1$-scaled block symmetric Gauss-Seidel/SSOR \\
 * 16 & Chebyshev \\
 * \hline
 * \end{tabular}
 **/
HYPRE_Int HYPRE_AMSSetSmoothingOptions(HYPRE_Solver solver,
                                       HYPRE_Int    relax_type,
                                       HYPRE_Int    relax_times,
                                       double       relax_weight,
                                       double       omega);

/**
 * (Optional) Sets AMG parameters for $B_\Pi$.
 * The defaults are $10$, $1$, $3$, $0.25$, $0$, $0$. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_AMSSetAlphaAMGOptions(HYPRE_Solver solver,
                                      HYPRE_Int    alpha_coarsen_type,
                                      HYPRE_Int    alpha_agg_levels,
                                      HYPRE_Int    alpha_relax_type,
                                      double       alpha_strength_threshold,
                                      HYPRE_Int    alpha_interp_type,
                                      HYPRE_Int    alpha_Pmax);

/**
 * (Optional) Sets AMG parameters for $B_G$.
 * The defaults are $10$, $1$, $3$, $0.25$, $0$, $0$. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_AMSSetBetaAMGOptions(HYPRE_Solver solver,
                                     HYPRE_Int    beta_coarsen_type,
                                     HYPRE_Int    beta_agg_levels,
                                     HYPRE_Int    beta_relax_type,
                                     double       beta_strength_threshold,
                                     HYPRE_Int    beta_interp_type,
                                     HYPRE_Int    beta_Pmax);

/**
 * Returns the number of iterations taken.
 **/
HYPRE_Int HYPRE_AMSGetNumIterations(HYPRE_Solver  solver,
                                    HYPRE_Int    *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                double       *rel_resid_norm);

/**
 * For problems with zero-conductivity regions, project the vector onto the
 * compatible subspace: $x = (I - G_0 (G_0^t G_0)^{-1} G_0^T) x$, where $G_0$ is
 * the discrete gradient restricted to the interior nodes of the regions with
 * zero conductivity. This ensures that x is orthogonal to the gradients in the
 * range of $G_0$.
 *
 * This function is typically called after the solution iteration is complete,
 * in order to facilitate the visualization of the computed field. Without it
 * the values in the zero-conductivity regions contain kernel components.
 **/
HYPRE_Int HYPRE_AMSProjectOutGradients(HYPRE_Solver    solver,
                                       HYPRE_ParVector x);

/**
 * Construct and return the lowest-order discrete gradient matrix G using some
 * edge and vertex information. We assume that edge\_vertex lists the edge
 * vertices consecutively, and that the orientation of all edges is consistent.
 *
 * If edge\_orientation = 1, the edges are already oriented.
 *
 * If edge\_orientation = 2, the orientation of edge i depends only
 * on the sign of edge\_vertex[2*i+1] - edge\_vertex[2*i].
 **/
HYPRE_Int HYPRE_AMSConstructDiscreteGradient(HYPRE_ParCSRMatrix  A,
                                             HYPRE_ParVector     x_coord,
                                             HYPRE_Int          *edge_vertex,
                                             HYPRE_Int           edge_orientation,
                                             HYPRE_ParCSRMatrix *G);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR ADS Solver and Preconditioner
 *
 * Parallel auxiliary space divergence solver and preconditioner
 **/
/*@{*/

/**
 * Create an ADS solver object.
 **/
HYPRE_Int HYPRE_ADSCreate(HYPRE_Solver *solver);

/**
 * Destroy an ADS solver object.
 **/
HYPRE_Int HYPRE_ADSDestroy(HYPRE_Solver solver);

/**
 * Set up the ADS solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
HYPRE_Int HYPRE_ADSSetup(HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x);

/**
 * Solve the system or apply ADS as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver {\tt SetPrecond} function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
HYPRE_Int HYPRE_ADSSolve(HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x);

/**
 * Sets the discrete curl matrix $C$.
 * This function should be called before HYPRE\_ADSSetup()!
 **/
HYPRE_Int HYPRE_ADSSetDiscreteCurl(HYPRE_Solver solver , HYPRE_ParCSRMatrix C);

/**
 * Sets the discrete gradient matrix $G$.
 * This function should be called before HYPRE\_ADSSetup()!
 **/
HYPRE_Int HYPRE_ADSSetDiscreteGradient(HYPRE_Solver solver , HYPRE_ParCSRMatrix G);

/**
 * Sets the $x$, $y$ and $z$ coordinates of the vertices in the mesh.
 * This function should be called before HYPRE\_ADSSetup()!
 **/
HYPRE_Int HYPRE_ADSSetCoordinateVectors(HYPRE_Solver solver , HYPRE_ParVector x , HYPRE_ParVector y ,
                                        HYPRE_ParVector z);

/**
 * (Optional) Set the (components of) the Raviart-Thomas ($\Pi_{RT}$) and the Nedelec
 * ($\Pi_{ND}$) interpolation matrices.
 *
 * This function is generally intended to be used only for high-order $H(div)$
 * discretizations (in the lowest order case, these matrices are constructed
 * internally in ADS from the discreet gradient and curl matrices and the
 * coordinates of the vertices), though it can also be used in the lowest-order
 * case or for other types of discretizations.
 *
 * By definition, RT\_Pi and ND\_Pi are the matrix representations of the linear
 * operators $\Pi_{RT}$ and $\Pi_{ND}$ that interpolate (high-order) vector
 * nodal finite elements into the (high-order) Raviart-Thomas and Nedelec
 * spaces. The component matrices are defined in both cases as $\Pi^x \varphi =
 * \Pi (\varphi,0,0)$ and similarly for $\Pi^y$ and $\Pi^z$. Note that all these
 * operators depend on the choice of the basis and degrees of freedom in the
 * high-order spaces.
 *
 * The column numbering of RT\_Pi and ND\_Pi should be node-based, i.e. the
 * $x$/$y$/$z$ components of the first node (vertex or high-order dof) should be
 * listed first, followed by the $x$/$y$/$z$ components of the second node and
 * so on (see the documentation of HYPRE\_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before hypre\_ADSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * $\{\Pi_{RT}\}$ and $\{\Pi_{RT}^x,\Pi_{RT}^y,\Pi_{RT}^z\}$ needs to be
 * specified (though it is OK to provide both).  If RT\_Pix is NULL, then scalar
 * $\Pi$-based ADS cycles, i.e. those with cycle\_type > 10, will be
 * unavailable. Similarly, ADS cycles based on monolithic $\Pi$ (cycle\_type <
 * 10) require that RT\_Pi is not NULL. The same restrictions hold for the sets
 * $\{\Pi_{ND}\}$ and $\{\Pi_{ND}^x,\Pi_{ND}^y,\Pi_{ND}^z\}$ -- only one of them
 * needs to be specified, and the availability of each enables different AMS
 * cycle type options.
 **/
HYPRE_Int HYPRE_ADSSetInterpolations(HYPRE_Solver solver,
                                     HYPRE_ParCSRMatrix RT_Pi,
                                     HYPRE_ParCSRMatrix RT_Pix,
                                     HYPRE_ParCSRMatrix RT_Piy,
                                     HYPRE_ParCSRMatrix RT_Piz,
                                     HYPRE_ParCSRMatrix ND_Pi,
                                     HYPRE_ParCSRMatrix ND_Pix,
                                     HYPRE_ParCSRMatrix ND_Piy,
                                     HYPRE_ParCSRMatrix ND_Piz);
/**
 * (Optional) Sets maximum number of iterations, if ADS is used
 * as a solver. To use ADS as a preconditioner, set the maximum
 * number of iterations to $1$. The default is $20$.
 **/
HYPRE_Int HYPRE_ADSSetMaxIter(HYPRE_Solver solver , HYPRE_Int maxit);

/**
 * (Optional) Set the convergence tolerance, if ADS is used
 * as a solver. When using ADS as a preconditioner, set the tolerance
 * to $0.0$. The default is $10^{-6}$.
 **/
HYPRE_Int HYPRE_ADSSetTol(HYPRE_Solver solver , double tol);

/**
 * (Optional) Choose which auxiliary-space solver to use. Possible values are:
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
HYPRE_Int HYPRE_ADSSetCycleType(HYPRE_Solver solver , HYPRE_Int cycle_type);

/**
 * (Optional) Control how much information is printed during the
 * solution iterations.
 * The default is $1$ (print residual norm at each step).
 **/
HYPRE_Int HYPRE_ADSSetPrintLevel(HYPRE_Solver solver , HYPRE_Int print_level);

/**
 * (Optional) Sets relaxation parameters for $A$.
 * The defaults are $2$, $1$, $1.0$, $1.0$.
 *
 * The available options for relax\_type are:
 *
 * \begin{tabular}{|c|l|} \hline
 * 1 & $\ell_1$-scaled Jacobi \\
 * 2 & $\ell_1$-scaled block symmetric Gauss-Seidel/SSOR \\
 * 3 & Kaczmarz \\
 * 4 & truncated version of $\ell_1$-scaled block symmetric Gauss-Seidel/SSOR \\
 * 16 & Chebyshev \\
 * \hline
 * \end{tabular}
 **/
HYPRE_Int HYPRE_ADSSetSmoothingOptions(HYPRE_Solver solver , HYPRE_Int relax_type , HYPRE_Int relax_times , double relax_weight , double omega);

/**
 * (Optional) Sets parameters for Chebyshev relaxation.
 * The defaults are $2$, $0.3$.
 **/
HYPRE_Int HYPRE_ADSSetChebySmoothingOptions(HYPRE_Solver solver , HYPRE_Int cheby_order , HYPRE_Int cheby_fraction);

/**
 * (Optional) Sets AMS parameters for $B_C$.
 * The defaults are $11$, $10$, $1$, $3$, $0.25$, $0$, $0$.
 * Note that cycle\_type should be greater than 10, unless the high-order
 * interface of HYPRE\_ADSSetInterpolations is being used!
 * See the user's manual for more details.
 **/
HYPRE_Int HYPRE_ADSSetAMSOptions(HYPRE_Solver solver , HYPRE_Int cycle_type , HYPRE_Int coarsen_type , HYPRE_Int agg_levels , HYPRE_Int relax_type , double strength_threshold , HYPRE_Int interp_type , HYPRE_Int Pmax);

/**
 * (Optional) Sets AMG parameters for $B_\Pi$.
 * The defaults are $10$, $1$, $3$, $0.25$, $0$, $0$. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_ADSSetAMGOptions(HYPRE_Solver solver , HYPRE_Int coarsen_type , HYPRE_Int agg_levels , HYPRE_Int relax_type , double strength_threshold , HYPRE_Int interp_type , HYPRE_Int Pmax);

/**
 * Returns the number of iterations taken.
 **/
HYPRE_Int HYPRE_ADSGetNumIterations(HYPRE_Solver solver , HYPRE_Int *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm(HYPRE_Solver solver , double *rel_resid_norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR PCG Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{PCG Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_ParCSRPCGCreate(MPI_Comm      comm,
                          HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_ParCSRPCGDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRPCGSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRPCGSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRPCGSetTol(HYPRE_Solver solver,
                          double       tol);

HYPRE_Int HYPRE_ParCSRPCGSetAbsoluteTol(HYPRE_Solver solver,
                                  double       tol);

HYPRE_Int HYPRE_ParCSRPCGSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int          max_iter);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRPCGSetStopCrit(HYPRE_Solver solver,
                               HYPRE_Int          stop_crit);

HYPRE_Int HYPRE_ParCSRPCGSetTwoNorm(HYPRE_Solver solver,
                              HYPRE_Int          two_norm);

HYPRE_Int HYPRE_ParCSRPCGSetRelChange(HYPRE_Solver solver,
                                HYPRE_Int          rel_change);

HYPRE_Int HYPRE_ParCSRPCGSetPrecond(HYPRE_Solver            solver,
                              HYPRE_PtrToParSolverFcn precond,
                              HYPRE_PtrToParSolverFcn precond_setup,
                              HYPRE_Solver            precond_solver);

HYPRE_Int HYPRE_ParCSRPCGGetPrecond(HYPRE_Solver  solver,
                              HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRPCGSetLogging(HYPRE_Solver solver,
                              HYPRE_Int          logging);

HYPRE_Int HYPRE_ParCSRPCGSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int          print_level);

HYPRE_Int HYPRE_ParCSRPCGGetNumIterations(HYPRE_Solver  solver,
                                    HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                double       *norm);

/**
 * Setup routine for diagonal preconditioning.
 **/
HYPRE_Int HYPRE_ParCSRDiagScaleSetup(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    y,
                               HYPRE_ParVector    x);

/**
 * Solve routine for diagonal preconditioning.
 **/
HYPRE_Int HYPRE_ParCSRDiagScale(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix HA,
                          HYPRE_ParVector    Hy,
                          HYPRE_ParVector    Hx);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{GMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_ParCSRGMRESCreate(MPI_Comm      comm,
                            HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_ParCSRGMRESDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRGMRESSetup(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRGMRESSolve(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRGMRESSetKDim(HYPRE_Solver solver,
                             HYPRE_Int          k_dim);

HYPRE_Int HYPRE_ParCSRGMRESSetTol(HYPRE_Solver solver,
                            double       tol);

HYPRE_Int HYPRE_ParCSRGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                    double       a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRGMRESSetMinIter(HYPRE_Solver solver,
                                HYPRE_Int          min_iter);

HYPRE_Int HYPRE_ParCSRGMRESSetMaxIter(HYPRE_Solver solver,
                                HYPRE_Int          max_iter);

/*
 * Obsolete
 **/
HYPRE_Int HYPRE_ParCSRGMRESSetStopCrit(HYPRE_Solver solver,
                                 HYPRE_Int          stop_crit);

HYPRE_Int HYPRE_ParCSRGMRESSetPrecond(HYPRE_Solver             solver,
                                HYPRE_PtrToParSolverFcn  precond,
                                HYPRE_PtrToParSolverFcn  precond_setup,
                                HYPRE_Solver             precond_solver);

HYPRE_Int HYPRE_ParCSRGMRESGetPrecond(HYPRE_Solver  solver,
                                HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRGMRESSetLogging(HYPRE_Solver solver,
                                HYPRE_Int          logging);

HYPRE_Int HYPRE_ParCSRGMRESSetPrintLevel(HYPRE_Solver solver,
                                   HYPRE_Int          print_level);

HYPRE_Int HYPRE_ParCSRGMRESGetNumIterations(HYPRE_Solver  solver,
                                      HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                  double       *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{FlexGMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_ParCSRFlexGMRESCreate(MPI_Comm      comm,
                                HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_ParCSRFlexGMRESDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetup(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRFlexGMRESSolve(HYPRE_Solver       solver,
                               HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector    b,
                               HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetKDim(HYPRE_Solver solver,
                                 HYPRE_Int          k_dim);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetTol(HYPRE_Solver solver,
                                double       tol);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                        double       a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMinIter(HYPRE_Solver solver,
                                    HYPRE_Int          min_iter);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetMaxIter(HYPRE_Solver solver,
                                    HYPRE_Int          max_iter);


HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrecond(HYPRE_Solver             solver,
                                    HYPRE_PtrToParSolverFcn  precond,
                                    HYPRE_PtrToParSolverFcn  precond_setup,
                                    HYPRE_Solver             precond_solver);

HYPRE_Int HYPRE_ParCSRFlexGMRESGetPrecond(HYPRE_Solver  solver,
                                    HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetLogging(HYPRE_Solver solver,
                                    HYPRE_Int          logging);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrintLevel(HYPRE_Solver solver,
                                       HYPRE_Int          print_level);

HYPRE_Int HYPRE_ParCSRFlexGMRESGetNumIterations(HYPRE_Solver  solver,
                                          HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                      double       *norm);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC( HYPRE_Solver           solver,
                                      HYPRE_PtrToModifyPCFcn modify_pc);
   
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{LGMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_ParCSRLGMRESCreate(MPI_Comm      comm,
                             HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_ParCSRLGMRESDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRLGMRESSetup(HYPRE_Solver       solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector    b,
                            HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRLGMRESSolve(HYPRE_Solver       solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector    b,
                            HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRLGMRESSetKDim(HYPRE_Solver solver,
                              HYPRE_Int          k_dim);

HYPRE_Int HYPRE_ParCSRLGMRESSetAugDim(HYPRE_Solver solver,
                                HYPRE_Int          aug_dim);

HYPRE_Int HYPRE_ParCSRLGMRESSetTol(HYPRE_Solver solver,
                             double       tol);
HYPRE_Int HYPRE_ParCSRLGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                     double       a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRLGMRESSetMinIter(HYPRE_Solver solver,
                                HYPRE_Int          min_iter);

HYPRE_Int HYPRE_ParCSRLGMRESSetMaxIter(HYPRE_Solver solver,
                                 HYPRE_Int          max_iter);

HYPRE_Int HYPRE_ParCSRLGMRESSetPrecond(HYPRE_Solver             solver,
                                 HYPRE_PtrToParSolverFcn  precond,
                                 HYPRE_PtrToParSolverFcn  precond_setup,
                                 HYPRE_Solver             precond_solver);

HYPRE_Int HYPRE_ParCSRLGMRESGetPrecond(HYPRE_Solver  solver,
                                 HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRLGMRESSetLogging(HYPRE_Solver solver,
                                 HYPRE_Int          logging);

HYPRE_Int HYPRE_ParCSRLGMRESSetPrintLevel(HYPRE_Solver solver,
                                    HYPRE_Int          print_level);

HYPRE_Int HYPRE_ParCSRLGMRESGetNumIterations(HYPRE_Solver  solver,
                                       HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                   double       *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{BiCGSTAB Solver}.
 **/
/*@{*/

/**
 * Create a solver object
 **/
HYPRE_Int HYPRE_ParCSRBiCGSTABCreate(MPI_Comm      comm,
                               HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_ParCSRBiCGSTABDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetup(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRBiCGSTABSolve(HYPRE_Solver       solver,
                              HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector    b,
                              HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetTol(HYPRE_Solver solver,
                               double       tol);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetAbsoluteTol(HYPRE_Solver solver,
                                       double       a_tol);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetMinIter(HYPRE_Solver solver,
                                   HYPRE_Int          min_iter);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetMaxIter(HYPRE_Solver solver,
                                   HYPRE_Int          max_iter);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetStopCrit(HYPRE_Solver solver,
                                    HYPRE_Int          stop_crit);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrecond(HYPRE_Solver            solver,
                                   HYPRE_PtrToParSolverFcn precond,
                                   HYPRE_PtrToParSolverFcn precond_setup,
                                   HYPRE_Solver            precond_solver);

HYPRE_Int HYPRE_ParCSRBiCGSTABGetPrecond(HYPRE_Solver  solver,
                                   HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetLogging(HYPRE_Solver solver,
                                   HYPRE_Int          logging);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrintLevel(HYPRE_Solver solver,
                                      HYPRE_Int          print_level);

HYPRE_Int HYPRE_ParCSRBiCGSTABGetNumIterations(HYPRE_Solver  solver,
                                         HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                     double       *norm);

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
HYPRE_Int HYPRE_ParCSRHybridCreate(HYPRE_Solver *solver);
/**
 *  Destroy solver object
 **/
HYPRE_Int HYPRE_ParCSRHybridDestroy(HYPRE_Solver solver);

/**
 *  Setup the hybrid solver
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetup(HYPRE_Solver       solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector    b,
                            HYPRE_ParVector    x);

/**
 *  Solve linear system
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
HYPRE_Int HYPRE_ParCSRHybridSolve(HYPRE_Solver       solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector    b,
                            HYPRE_ParVector    x);
/**
 *  Set the convergence tolerance for the Krylov solver. The default is 1.e-7.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetTol(HYPRE_Solver solver,
                             double       tol);
/**
 *  Set the absolute convergence tolerance for the Krylov solver. The default is 0.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetAbsoluteTol(HYPRE_Solver solver,
                                     double       tol);

/**
 *  Set the desired convergence factor
 **/
HYPRE_Int HYPRE_ParCSRHybridSetConvergenceTol(HYPRE_Solver solver,
                                        double       cf_tol);

/**
 *  Set the maximal number of iterations for the diagonally
 *  preconditioned solver
 **/
HYPRE_Int HYPRE_ParCSRHybridSetDSCGMaxIter(HYPRE_Solver solver,
                                     HYPRE_Int          dscg_max_its);

/**
 *  Set the maximal number of iterations for the AMG
 *  preconditioned solver
 **/
HYPRE_Int HYPRE_ParCSRHybridSetPCGMaxIter(HYPRE_Solver solver,
                                    HYPRE_Int          pcg_max_its);

/*
 *
 **/
HYPRE_Int HYPRE_ParCSRHybridSetSetupType(HYPRE_Solver solver,
                                   HYPRE_Int          setup_type);

/**
 *  Set the desired solver type. There are the following options:
 * \begin{tabular}{l l}
 *     1 & PCG (default) \\
 *     2 & GMRES \\
 *     3 & BiCGSTAB
 * \end{tabular}
 **/
HYPRE_Int HYPRE_ParCSRHybridSetSolverType(HYPRE_Solver solver,
                                    HYPRE_Int          solver_type);

/**
 * Set the Krylov dimension for restarted GMRES.
 * The default is 5.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetKDim(HYPRE_Solver solver,
                              HYPRE_Int          k_dim);

/**
 * Set the type of norm for PCG.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetTwoNorm(HYPRE_Solver solver,
                                 HYPRE_Int          two_norm);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRHybridSetStopCrit(HYPRE_Solver solver,
                                  HYPRE_Int          stop_crit);

/*
 *
 **/
HYPRE_Int HYPRE_ParCSRHybridSetRelChange(HYPRE_Solver solver,
                                   HYPRE_Int          rel_change);

/**
 * Set preconditioner if wanting to use one that is not set up by
 * the hybrid solver.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetPrecond(HYPRE_Solver            solver,
                                 HYPRE_PtrToParSolverFcn precond,
                                 HYPRE_PtrToParSolverFcn precond_setup,
                                 HYPRE_Solver            precond_solver);
                    
/**
 * Set logging parameter (default: 0, no logging).
 **/
HYPRE_Int HYPRE_ParCSRHybridSetLogging(HYPRE_Solver solver,
                                 HYPRE_Int          logging);

/**
 * Set print level (default: 0, no printing).
 **/
HYPRE_Int HYPRE_ParCSRHybridSetPrintLevel(HYPRE_Solver solver,
                                    HYPRE_Int          print_level);

/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For 2d Laplace operators, 0.25 is a good value, for 3d Laplace
 * operators, 0.5 or 0.6 is a better value. For elasticity problems,
 * a large strength threshold, such as 0.9, is often better.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetStrongThreshold(HYPRE_Solver solver,
                                     double       strong_threshold);

/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If max\_row\_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMaxRowSum(HYPRE_Solver solver,
                               double       max_row_sum);

/**
 * (Optional) Defines a truncation factor for the interpolation.
 * The default is 0.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetTruncFactor(HYPRE_Solver solver,
                                 double       trunc_factor);


/**
 * (Optional) Defines the maximal number of elements per row for the interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetPMaxElmts(HYPRE_Solver solver,
                                   HYPRE_Int          P_max_elmts);

/**
 * (Optional) Defines the maximal number of levels used for AMG.
 * The default is 25.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMaxLevels(HYPRE_Solver solver,
                               HYPRE_Int          max_levels);

/**
 * (Optional) Defines whether local or global measures are used.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMeasureType(HYPRE_Solver solver,
                                 HYPRE_Int          measure_type);

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
HYPRE_Int
HYPRE_ParCSRHybridSetCoarsenType(HYPRE_Solver solver,
                                 HYPRE_Int          coarsen_type);

/*
 * (Optional) Specifies which interpolation operator is used
 * The default is modified ''classical" interpolation.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetInterpType(HYPRE_Solver solver,
                                HYPRE_Int          interp_type);

/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set cycle\_type to 1, for a W-cycle
 *  set cycle\_type to 2. The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetCycleType(HYPRE_Solver solver,
                               HYPRE_Int          cycle_type);

/*
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetGridRelaxType(HYPRE_Solver  solver,
                                   HYPRE_Int          *grid_relax_type);

/*
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetGridRelaxPoints(HYPRE_Solver   solver,
                                     HYPRE_Int          **grid_relax_points);

/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and
 * the down cycle the number of sweeps are set to num\_sweeps and on the
 * coarsest level to 1. The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetNumSweeps(HYPRE_Solver solver,
                               HYPRE_Int          num_sweeps);

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
HYPRE_Int
HYPRE_ParCSRHybridSetCycleNumSweeps(HYPRE_Solver solver,
                                    HYPRE_Int          num_sweeps,
                                    HYPRE_Int          k);

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
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxType(HYPRE_Solver solver,
                               HYPRE_Int          relax_type);

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
HYPRE_Int
HYPRE_ParCSRHybridSetCycleRelaxType(HYPRE_Solver solver,
                                    HYPRE_Int          relax_type,
                                    HYPRE_Int          k);

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
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxOrder(HYPRE_Solver solver,
                                HYPRE_Int          relax_order);

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
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxWt(HYPRE_Solver solver,
                             double       relax_wt);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive relax\_weight, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetRelaxWt.
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetLevelRelaxWt(HYPRE_Solver solver,
                                  double       relax_wt,
                                  HYPRE_Int          level);

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
HYPRE_Int
HYPRE_ParCSRHybridSetOuterWt(HYPRE_Solver solver,
                             double       outer_wt);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for HYPRE\_BoomerAMGSetOuterWt.
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetLevelOuterWt(HYPRE_Solver solver,
                                  double       outer_wt,
                                  HYPRE_Int    level);

/**
 * (Optional) Defines the maximal coarse grid size.
 * The default is 9.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMaxCoarseSize(HYPRE_Solver solver,
                               HYPRE_Int        max_coarse_size);

/**
 * (Optional) enables redundant coarse grid size. If the system size becomes
 * smaller than seq_threshold, sequential AMG is used on all remaining processors.
 * The default is 0.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetSeqThreshold(HYPRE_Solver solver,
                               HYPRE_Int       seq_threshold);

/*
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxWeight(HYPRE_Solver  solver,
                                 double       *relax_weight);

/*
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetOmega(HYPRE_Solver  solver,
                           double       *omega);

/**
 * (Optional) Defines the number of levels of aggressive coarsening,
 * starting with the finest level.
 * The default is 0, i.e. no aggressive coarsening.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetAggNumLevels(HYPRE_Solver solver,
                                  HYPRE_Int          agg_num_levels);

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1, which leads to the most aggressive coarsening.
 * Setting num$\_$paths to 2 will increase complexity somewhat,
 * but can lead to better convergence.**/
HYPRE_Int
HYPRE_ParCSRHybridSetNumPaths(HYPRE_Solver solver,
                              HYPRE_Int          num_paths);

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetNumFunctions(HYPRE_Solver solver,
                                  HYPRE_Int          num_functions);

/**
 * (Optional) Sets the mapping that assigns the function to each variable,
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetDofFunc(HYPRE_Solver  solver,
                             HYPRE_Int          *dof_func);
/**
 * (Optional) Sets whether to use the nodal systems version.
 * The default is 0 (the unknown based approach).
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetNodal(HYPRE_Solver solver,
                           HYPRE_Int          nodal);

/**
 * Retrieves the total number of iterations.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetNumIterations(HYPRE_Solver  solver,
                                       HYPRE_Int          *num_its);

/**
 * Retrieves the number of iterations used by the diagonally scaled solver.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetDSCGNumIterations(HYPRE_Solver  solver,
                                           HYPRE_Int          *dscg_num_its);

/**
 * Retrieves the number of iterations used by the AMG preconditioned solver.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetPCGNumIterations(HYPRE_Solver  solver,
                                          HYPRE_Int          *pcg_num_its);

/**
 * Retrieves the final relative residual norm.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                   double       *norm);

/* Is this a retired function? (RDF) */
HYPRE_Int
HYPRE_ParCSRHybridSetNumGridSweeps(HYPRE_Solver  solver,
                                   HYPRE_Int          *num_grid_sweeps);


/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name Schwarz Solver
 **/

HYPRE_Int HYPRE_SchwarzCreate(HYPRE_Solver *solver);

HYPRE_Int HYPRE_SchwarzDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_SchwarzSetup(HYPRE_Solver       solver,
                       HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector    b,
                       HYPRE_ParVector    x);

HYPRE_Int HYPRE_SchwarzSolve(HYPRE_Solver       solver,
                       HYPRE_ParCSRMatrix A,
                       HYPRE_ParVector    b,
                       HYPRE_ParVector    x);

HYPRE_Int HYPRE_SchwarzSetVariant(HYPRE_Solver solver,
                            HYPRE_Int          variant);

HYPRE_Int HYPRE_SchwarzSetOverlap(HYPRE_Solver solver,
                            HYPRE_Int          overlap);

HYPRE_Int HYPRE_SchwarzSetDomainType(HYPRE_Solver solver,
                               HYPRE_Int          domain_type);

HYPRE_Int HYPRE_SchwarzSetRelaxWeight(HYPRE_Solver solver,
                                double       relax_weight);

HYPRE_Int HYPRE_SchwarzSetDomainStructure(HYPRE_Solver    solver,
                                    HYPRE_CSRMatrix domain_structure);

HYPRE_Int HYPRE_SchwarzSetNumFunctions(HYPRE_Solver solver,
                                 HYPRE_Int          num_functions);

HYPRE_Int HYPRE_SchwarzSetDofFunc(HYPRE_Solver  solver,
                            HYPRE_Int          *dof_func);

HYPRE_Int HYPRE_SchwarzSetNonSymm(HYPRE_Solver solver,
                            HYPRE_Int          use_nonsymm);
   
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * @name ParCSR CGNR Solver
 **/

HYPRE_Int HYPRE_ParCSRCGNRCreate(MPI_Comm      comm,
                           HYPRE_Solver *solver);

HYPRE_Int HYPRE_ParCSRCGNRDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRCGNRSetup(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRCGNRSolve(HYPRE_Solver       solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector    b,
                          HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRCGNRSetTol(HYPRE_Solver solver,
                           double       tol);

HYPRE_Int HYPRE_ParCSRCGNRSetMinIter(HYPRE_Solver solver,
                               HYPRE_Int          min_iter);

HYPRE_Int HYPRE_ParCSRCGNRSetMaxIter(HYPRE_Solver solver,
                               HYPRE_Int          max_iter);

HYPRE_Int HYPRE_ParCSRCGNRSetStopCrit(HYPRE_Solver solver,
                                HYPRE_Int          stop_crit);

HYPRE_Int HYPRE_ParCSRCGNRSetPrecond(HYPRE_Solver            solver,
                               HYPRE_PtrToParSolverFcn precond,
                               HYPRE_PtrToParSolverFcn precondT,
                               HYPRE_PtrToParSolverFcn precond_setup,
                               HYPRE_Solver            precond_solver);

HYPRE_Int HYPRE_ParCSRCGNRGetPrecond(HYPRE_Solver  solver,
                               HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRCGNRSetLogging(HYPRE_Solver solver,
                               HYPRE_Int          logging);

HYPRE_Int HYPRE_ParCSRCGNRGetNumIterations(HYPRE_Solver  solver,
                                     HYPRE_Int          *num_iterations);

HYPRE_Int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                 double       *norm);

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix GenerateLaplacian(MPI_Comm comm,
                                     HYPRE_Int      nx,
                                     HYPRE_Int      ny,
                                     HYPRE_Int      nz,
                                     HYPRE_Int      P,
                                     HYPRE_Int      Q,
                                     HYPRE_Int      R,
                                     HYPRE_Int      p,
                                     HYPRE_Int      q,
                                     HYPRE_Int      r,
                                     double  *value);

HYPRE_ParCSRMatrix GenerateLaplacian27pt(MPI_Comm comm,
                                         HYPRE_Int      nx,
                                         HYPRE_Int      ny,
                                         HYPRE_Int      nz,
                                         HYPRE_Int      P,
                                         HYPRE_Int      Q,
                                         HYPRE_Int      R,
                                         HYPRE_Int      p,
                                         HYPRE_Int      q,
                                         HYPRE_Int      r,
                                         double  *value);

HYPRE_ParCSRMatrix GenerateLaplacian9pt(MPI_Comm comm,
                                        HYPRE_Int      nx,
                                        HYPRE_Int      ny,
                                        HYPRE_Int      P,
                                        HYPRE_Int      Q,
                                        HYPRE_Int      p,
                                        HYPRE_Int      q,
                                        double  *value);

HYPRE_ParCSRMatrix GenerateDifConv(MPI_Comm comm,
                                   HYPRE_Int      nx,
                                   HYPRE_Int      ny,
                                   HYPRE_Int      nz,
                                   HYPRE_Int      P,
                                   HYPRE_Int      Q,
                                   HYPRE_Int      R,
                                   HYPRE_Int      p,
                                   HYPRE_Int      q,
                                   HYPRE_Int      r,
                                   double  *value);

HYPRE_ParCSRMatrix
GenerateRotate7pt(MPI_Comm comm,
                  HYPRE_Int      nx,
                  HYPRE_Int      ny,
                  HYPRE_Int      P,
                  HYPRE_Int      Q,
                  HYPRE_Int      p,
                  HYPRE_Int      q,
                  double   alpha,
                  double   eps );
                                                                                
HYPRE_ParCSRMatrix
GenerateVarDifConv(MPI_Comm comm,
                   HYPRE_Int      nx,
                   HYPRE_Int      ny,
                   HYPRE_Int      nz,
                   HYPRE_Int      P,
                   HYPRE_Int      Q,
                   HYPRE_Int      R,
                   HYPRE_Int      p,
                   HYPRE_Int      q,
                   HYPRE_Int      r,
                   double eps,
                   HYPRE_ParVector *rhs_ptr);

float*
GenerateCoordinates(MPI_Comm comm,
                    HYPRE_Int      nx,
                    HYPRE_Int      ny,
                    HYPRE_Int      nz,
                    HYPRE_Int      P,
                    HYPRE_Int      Q,
                    HYPRE_Int      R,
                    HYPRE_Int      p,
                    HYPRE_Int      q,
                    HYPRE_Int      r,
                    HYPRE_Int      coorddim);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*
 * (Optional) Switches on use of Jacobi interpolation after computing
 * an original interpolation
 **/
HYPRE_Int HYPRE_BoomerAMGSetPostInterpType(HYPRE_Solver solver,
                                     HYPRE_Int          post_interp_type);

/*
 * (Optional) Sets a truncation threshold for Jacobi interpolation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetJacobiTruncThreshold(HYPRE_Solver solver,
                                           double       jacobi_trunc_threshold);

/*
 * (Optional) Defines the number of relaxation steps for CR
 * The default is 2.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumCRRelaxSteps(HYPRE_Solver solver,
                                      HYPRE_Int          num_CR_relax_steps);

/*
 * (Optional) Defines convergence rate for CR
 * The default is 0.7.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCRRate(HYPRE_Solver solver,
                             double       CR_rate);

/*
 * (Optional) Defines strong threshold for CR
 * The default is 0.0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCRStrongTh(HYPRE_Solver solver,
                                 double       CR_strong_th);

/*
 * (Optional) Defines whether to use CG 
 **/
HYPRE_Int HYPRE_BoomerAMGSetCRUseCG(HYPRE_Solver solver,
                              HYPRE_Int          CR_use_CG);

/*
 * (Optional) Defines the Type of independent set algorithm used for CR
 **/
HYPRE_Int HYPRE_BoomerAMGSetISType(HYPRE_Solver solver,
                             HYPRE_Int          IS_type);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* These includes shouldn't be here. (RDF) */
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"

/**
 * @name ParCSR LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{LOBPCG Eigensolver}.
 **/
/*@{*/

/**
 * Load interface interpreter.  Vector part loaded with hypre_ParKrylov
 * functions and multivector part loaded with mv_TempMultiVector functions.
 **/
HYPRE_Int
HYPRE_ParCSRSetupInterpreter(mv_InterfaceInterpreter *i);

/**
 * Load Matvec interpreter with hypre_ParKrylov functions.
 **/
HYPRE_Int
HYPRE_ParCSRSetupMatvec(HYPRE_MatvecFunctions *mv);

/* The next routines should not be here (lower-case prefix). (RDF) */

/*
 * Print multivector to file.
 **/
HYPRE_Int
hypre_ParCSRMultiVectorPrint(void *x_, const char *fileName);

/*
 * Read multivector from file.
 **/
void *
hypre_ParCSRMultiVectorRead(MPI_Comm comm, void *ii_, const char *fileName);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/ 
/*@}*/

#ifdef __cplusplus
}
#endif

#endif
