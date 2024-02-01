/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_PARCSR_LS_HEADER
#define HYPRE_PARCSR_LS_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_seq_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup ParCSRSolvers ParCSR Solvers
 *
 * Linear solvers for sparse matrix systems. These solvers use matrix/vector
 * storage schemes that are taylored for general sparse matrix systems.
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Solvers
 *
 * @{
 **/

/**
 * The solver object.
 **/

typedef HYPRE_Int (*HYPRE_PtrToParSolverFcn)(HYPRE_Solver,
                                             HYPRE_ParCSRMatrix,
                                             HYPRE_ParVector,
                                             HYPRE_ParVector);

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
 * @name ParCSR BoomerAMG Solver and Preconditioner
 *
 * Parallel unstructured algebraic multigrid solver and preconditioner
 *
 * @{
 **/

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
 * to the iterative solver \e SetPrecond function.
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
 * to the iterative solver \e SetPrecond function.
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
 * Solve the transpose system \f$A^T x = b\f$ or apply AMG as a preconditioner
 * to the transpose system . Note that this function should only be used
 * when preconditioning CGNR with BoomerAMG. It can only be used with
 * Jacobi smoothing (relax_type 0 or 7) and without CF smoothing,
 * i.e relax_order needs to be set to 0.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
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
 * Recovers old default for coarsening and interpolation, i.e Falgout
 * coarsening and untruncated modified classical interpolation.
 * This option might be preferred for 2 dimensional problems.
 **/
HYPRE_Int HYPRE_BoomerAMGSetOldDefault(HYPRE_Solver       solver);

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
 * Returns cumulative num of nonzeros for A and P operators
 **/
HYPRE_Int HYPRE_BoomerAMGGetCumNnzAP(HYPRE_Solver  solver,
                                     HYPRE_Real   *cum_nnz_AP);

/**
 * Activates cumulative num of nonzeros for A and P operators.
 * Needs to be set to a positive number for activation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCumNnzAP(HYPRE_Solver  solver,
                                     HYPRE_Real    cum_nnz_AP);

/**
 * Returns the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_BoomerAMGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                      HYPRE_Real   *rel_resid_norm);

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1, i.e. a scalar system.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumFunctions(HYPRE_Solver solver,
                                         HYPRE_Int          num_functions);

/**
 * (Optional) Sets the mapping that assigns the function to each variable,
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
HYPRE_Int HYPRE_BoomerAMGSetDofFunc(HYPRE_Solver  solver,
                                    HYPRE_Int    *dof_func);

/**
 * (Optional) Set the type convergence checking
 * 0: (default) norm(r)/norm(b), or norm(r) when b == 0
 * 1: nomr(r) / norm(r_0)
 **/
HYPRE_Int HYPRE_BoomerAMGSetConvergeType(HYPRE_Solver solver,
                                         HYPRE_Int    type);

/**
 * (Optional) Set the convergence tolerance, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, it should be set to 0.
 * The default is 1.e-6.
 **/
HYPRE_Int HYPRE_BoomerAMGSetTol(HYPRE_Solver solver,
                                HYPRE_Real   tol);

/**
 * (Optional) Sets maximum number of iterations, if BoomerAMG is used
 * as a solver. If it is used as a preconditioner, it should be set to 1.
 * The default is 20.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxIter(HYPRE_Solver solver,
                                    HYPRE_Int          max_iter);

/**
 * (Optional)
 **/
HYPRE_Int HYPRE_BoomerAMGSetMinIter(HYPRE_Solver solver,
                                    HYPRE_Int    min_iter);

/**
 * (Optional) Sets maximum size of coarsest grid.
 * The default is 9.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxCoarseSize(HYPRE_Solver solver,
                                          HYPRE_Int    max_coarse_size);

/**
 * (Optional) Sets minimum size of coarsest grid.
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMinCoarseSize(HYPRE_Solver solver,
                                          HYPRE_Int    min_coarse_size);

/**
 * (Optional) Sets maximum number of multigrid levels.
 * The default is 25.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxLevels(HYPRE_Solver solver,
                                      HYPRE_Int    max_levels);

/**
 * (Optional) Sets cut factor for choosing isolated points
 * during coarsening according to the rows' density. The default is 0.
 * If nnzrow > coarsen_cut_factor*avg_nnzrow, where avg_nnzrow is the
 * average number of nonzeros per row of the global matrix, holds for
 * a given row, it is set as fine, and interpolation weights are not computed.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCoarsenCutFactor(HYPRE_Solver solver,
                                             HYPRE_Int    coarsen_cut_factor);

/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For 2D Laplace operators, 0.25 is a good value, for 3D Laplace
 * operators, 0.5 or 0.6 is a better value. For elasticity problems,
 * a large strength threshold, such as 0.9, is often better.
 **/
HYPRE_Int HYPRE_BoomerAMGSetStrongThreshold(HYPRE_Solver solver,
                                            HYPRE_Real   strong_threshold);

/**
 * (Optional) The strong threshold for R is strong connections used
 * in building an approximate ideal restriction.
 * Default value is 0.25.
 **/
HYPRE_Int HYPRE_BoomerAMGSetStrongThresholdR(HYPRE_Solver solver,
                                             HYPRE_Real   strong_threshold);

/**
 * (Optional) The filter threshold for R is used to eliminate small entries
 * of the approximate ideal restriction after building it.
 * Default value is 0.0, which disables filtering.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFilterThresholdR(HYPRE_Solver solver,
                                             HYPRE_Real   filter_threshold);

/**
 * (Optional) Deprecated. This routine now has no effect.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSCommPkgSwitch(HYPRE_Solver solver,
                                           HYPRE_Real   S_commpkg_switch);

/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If \e max_row_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxRowSum(HYPRE_Solver solver,
                                      HYPRE_Real    max_row_sum);

/**
 * (Optional) Defines which parallel coarsening algorithm is used.
 * There are the following options for \e coarsen_type:
 *
 *    - 0  : CLJP-coarsening (a parallel coarsening algorithm using independent sets.
 *    - 1  : classical Ruge-Stueben coarsening on each processor, no boundary treatment
             (not recommended!)
 *    - 3  : classical Ruge-Stueben coarsening on each processor, followed by a third pass,
             which adds coarse points on the boundaries
 *    - 6  : Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points
 *           generated by 1 as its first independent set)
 *    - 7  : CLJP-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 8  : PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating
 *           lower complexities than CLJP, might also lead to slower convergence)
 *    - 9  : PMIS-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 10 : HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed
 *           by PMIS using the interior C-points generated as its first independent set)
 *    - 11 : one-pass Ruge-Stueben coarsening on each processor, no boundary treatment
             (not recommended!)
 *    - 21 : CGC coarsening by M. Griebel, B. Metsch and A. Schweitzer
 *    - 22 : CGC-E coarsening by M. Griebel, B. Metsch and A.Schweitzer
 *
 * The default is 10.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCoarsenType(HYPRE_Solver solver,
                                        HYPRE_Int    coarsen_type);

/**
 * (Optional) Defines the non-Galerkin drop-tolerance
 * for sparsifying coarse grid operators and thus reducing communication.
 * Value specified here is set on all levels.
 * This routine should be used before HYPRE_BoomerAMGSetLevelNonGalerkinTol, which
 * then can be used to change individual levels if desired
 **/
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkinTol (HYPRE_Solver solver,
                                            HYPRE_Real  nongalerkin_tol);

/**
 * (Optional) Defines the level specific non-Galerkin drop-tolerances
 * for sparsifying coarse grid operators and thus reducing communication.
 * A drop-tolerance of 0.0 means to skip doing non-Galerkin on that
 * level.  The maximum drop tolerance for a level is 1.0, although
 * much smaller values such as 0.03 or 0.01 are recommended.
 *
 * Note that if the user wants to set a  specific tolerance on all levels,
 * HYPRE_BooemrAMGSetNonGalerkinTol should be used. Individual levels
 * can then be changed using this routine.
 *
 * In general, it is safer to drop more aggressively on coarser levels.
 * For instance, one could use 0.0 on the finest level, 0.01 on the second level and
 * then using 0.05 on all remaining levels. The best way to achieve this is
 * to set 0.05 on all levels with HYPRE_BoomerAMGSetNonGalerkinTol and then
 * change the tolerance on level 0 to 0.0 and the tolerance on level 1 to 0.01
 * with HYPRE_BoomerAMGSetLevelNonGalerkinTol.
 * Like many AMG parameters, these drop tolerances can be tuned.  It is also common
 * to delay the start of the non-Galerkin process further to a later level than
 * level 1.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param nongalerkin_tol [IN] level specific drop tolerance
 * @param level [IN] level on which drop tolerance is used
 **/
HYPRE_Int HYPRE_BoomerAMGSetLevelNonGalerkinTol (HYPRE_Solver solver,
                                                 HYPRE_Real   nongalerkin_tol,
                                                 HYPRE_Int  level);

/**
 * (Optional) Defines the non-Galerkin drop-tolerance (old version)
 **/
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkTol (HYPRE_Solver solver,
                                          HYPRE_Int    nongalerk_num_tol,
                                          HYPRE_Real  *nongalerk_tol);

/**
 * (Optional) Defines whether local or global measures are used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMeasureType(HYPRE_Solver solver,
                                        HYPRE_Int    measure_type);

/**
 * (Optional) Defines the number of levels of aggressive coarsening.
 * The default is 0, i.e. no aggressive coarsening.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggNumLevels(HYPRE_Solver solver,
                                         HYPRE_Int    agg_num_levels);

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1. Larger numbers lead to less aggressive
 * coarsening.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumPaths(HYPRE_Solver solver,
                                     HYPRE_Int    num_paths);

/**
 * (optional) Defines the number of pathes for CGC-coarsening.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCGCIts (HYPRE_Solver solver,
                                    HYPRE_Int    its);

/**
 * (Optional) Sets whether to use the nodal systems coarsening.
 * Should be used for linear systems generated from systems of PDEs.
 * The default is 0 (unknown-based coarsening,
 *                   only coarsens within same function).
 * For the remaining options a nodal matrix is generated by
 * applying a norm to the nodal blocks and applying the coarsening
 * algorithm to this matrix.
 *    - 1 : Frobenius norm
 *    - 2 : sum of absolute values of elements in each block
 *    - 3 : largest element in each block (not absolute value)
 *    - 4 : row-sum norm
 *    - 6 : sum of all values in each block
 **/
HYPRE_Int HYPRE_BoomerAMGSetNodal(HYPRE_Solver solver,
                                  HYPRE_Int    nodal);
/**
 * (Optional) Sets whether to give special treatment to diagonal elements in
 * the nodal systems version.
 * The default is 0.
 * If set to 1, the diagonal entry is set to the negative sum of all off
 * diagonal entries.
 * If set to 2, the signs of all diagonal entries are inverted.
 */
HYPRE_Int HYPRE_BoomerAMGSetNodalDiag(HYPRE_Solver solver,
                                      HYPRE_Int    nodal_diag);


/*
 * (Optional) Sets whether to keep same sign in S for nodal > 0
 * The default is 0, i.e., discard those elements.
 */
HYPRE_Int HYPRE_BoomerAMGSetKeepSameSign(HYPRE_Solver solver,
                                         HYPRE_Int    keep_same_sign);

/**
 * (Optional) Defines which parallel interpolation operator is used.
 * There are the following options for \e interp_type:
 *
 *    - 0  : classical modified interpolation
 *    - 1  : LS interpolation (for use with GSMG)
 *    - 2  : classical modified interpolation for hyperbolic PDEs
 *    - 3  : direct interpolation (with separation of weights) (also for GPU use)
 *    - 4  : multipass interpolation
 *    - 5  : multipass interpolation (with separation of weights)
 *    - 6  : extended+i interpolation (also for GPU use)
 *    - 7  : extended+i (if no common C neighbor) interpolation
 *    - 8  : standard interpolation
 *    - 9  : standard interpolation (with separation of weights)
 *    - 10 : classical block interpolation (for use with nodal systems version only)
 *    - 11 : classical block interpolation (for use with nodal systems version only)
 *           with diagonalized diagonal blocks
 *    - 12 : FF interpolation
 *    - 13 : FF1 interpolation
 *    - 14 : extended interpolation (also for GPU use)
 *    - 15 : interpolation with adaptive weights (GPU use only)
 *    - 16 : extended interpolation in matrix-matrix form
 *    - 17 : extended+i interpolation in matrix-matrix form
 *    - 18 : extended+e interpolation in matrix-matrix form
 *
 * The default is ext+i interpolation (interp_type 6) trunctated to at most 4
 * elements per row. (see HYPRE_BoomerAMGSetPMaxElmts).
 **/
HYPRE_Int HYPRE_BoomerAMGSetInterpType(HYPRE_Solver solver,
                                       HYPRE_Int    interp_type);

/**
 * (Optional) Defines a truncation factor for the interpolation. The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetTruncFactor(HYPRE_Solver solver,
                                        HYPRE_Real   trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the interpolation.
 * The default is 4. To turn off truncation, it needs to be set to 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetPMaxElmts(HYPRE_Solver solver,
                                      HYPRE_Int    P_max_elmts);

/**
 * (Optional) Defines whether separation of weights is used
 * when defining strength for standard interpolation or
 * multipass interpolation.
 * Default: 0, i.e. no separation of weights used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSepWeight(HYPRE_Solver solver,
                                      HYPRE_Int    sep_weight);

/**
 * (Optional) Defines the interpolation used on levels of aggressive coarsening
 * The default is 4, i.e. multipass interpolation.
 * The following options exist:
 *
 *    - 1 : 2-stage extended+i interpolation
 *    - 2 : 2-stage standard interpolation
 *    - 3 : 2-stage extended interpolation
 *    - 4 : multipass interpolation
 *    - 5 : 2-stage extended interpolation in matrix-matrix form
 *    - 6 : 2-stage extended+i interpolation in matrix-matrix form
 *    - 7 : 2-stage extended+e interpolation in matrix-matrix form
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggInterpType(HYPRE_Solver solver,
                                          HYPRE_Int    agg_interp_type);

/**
 * (Optional) Defines the truncation factor for the
 * interpolation used for aggressive coarsening.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggTruncFactor(HYPRE_Solver solver,
                                           HYPRE_Real   agg_trunc_factor);

/**
 * (Optional) Defines the truncation factor for the
 * matrices P1 and P2 which are used to build 2-stage interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggP12TruncFactor(HYPRE_Solver solver,
                                              HYPRE_Real   agg_P12_trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the
 * interpolation used for aggressive coarsening.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggPMaxElmts(HYPRE_Solver solver,
                                         HYPRE_Int    agg_P_max_elmts);

/**
 * (Optional) Defines the maximal number of elements per row for the
 * matrices P1 and P2 which are used to build 2-stage interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAggP12MaxElmts(HYPRE_Solver solver,
                                           HYPRE_Int    agg_P12_max_elmts);

/**
 * (Optional) Allows the user to incorporate additional vectors
 * into the interpolation for systems AMG, e.g. rigid body modes for
 * linear elasticity problems.
 * This can only be used in context with nodal coarsening and still
 * requires the user to choose an interpolation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetInterpVectors (HYPRE_Solver     solver,
                                           HYPRE_Int        num_vectors,
                                           HYPRE_ParVector *interp_vectors );

/**
 * (Optional) Defines the interpolation variant used for
 * HYPRE_BoomerAMGSetInterpVectors:
 *    - 1 : GM approach 1
 *    - 2 : GM approach 2  (to be preferred over 1)
 *    - 3 : LN approach
 **/
HYPRE_Int HYPRE_BoomerAMGSetInterpVecVariant (HYPRE_Solver solver,
                                              HYPRE_Int    var );

/**
 * (Optional) Defines the maximal elements per row for Q, the additional
 * columns added to the original interpolation matrix P, to reduce complexity.
 * The default is no truncation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetInterpVecQMax (HYPRE_Solver solver,
                                           HYPRE_Int    q_max );

/**
 * (Optional) Defines a truncation factor for Q, the additional
 * columns added to the original interpolation matrix P, to reduce complexity.
 * The default is no truncation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetInterpVecAbsQTrunc (HYPRE_Solver solver,
                                                HYPRE_Real   q_trunc );

/**
 * (Optional) Specifies the use of GSMG - geometrically smooth
 * coarsening and interpolation. Currently any nonzero value for
 * gsmg will lead to the use of GSMG.
 * The default is 0, i.e. (GSMG is not used)
 **/
HYPRE_Int HYPRE_BoomerAMGSetGSMG(HYPRE_Solver solver,
                                 HYPRE_Int    gsmg);

/**
 * (Optional) Defines the number of sample vectors used in GSMG
 * or LS interpolation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumSamples(HYPRE_Solver solver,
                                       HYPRE_Int    num_samples);
/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set \e cycle_type to 1, for a W-cycle
 *  set \e cycle_type to 2. The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCycleType(HYPRE_Solver solver,
                                      HYPRE_Int    cycle_type);
/**
 * (Optional) Specifies the use of Full multigrid cycle.
 * The default is 0.
 **/
HYPRE_Int
HYPRE_BoomerAMGSetFCycle( HYPRE_Solver solver,
                          HYPRE_Int    fcycle  );

/**
 * (Optional) Defines use of an additive V(1,1)-cycle using the
 * classical additive method starting at level 'addlvl'.
 * The multiplicative approach is used on levels 0, ...'addlvl+1'.
 * 'addlvl' needs to be > -1 for this to have an effect.
 * Can only be used with weighted Jacobi and l1-Jacobi(default).
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
HYPRE_Int HYPRE_BoomerAMGSetAdditive(HYPRE_Solver solver,
                                     HYPRE_Int    addlvl);

/**
 * (Optional) Defines use of an additive V(1,1)-cycle using the
 * mult-additive method starting at level 'addlvl'.
 * The multiplicative approach is used on levels 0, ...'addlvl+1'.
 * 'addlvl' needs to be > -1 for this to have an effect.
 * Can only be used with weighted Jacobi and l1-Jacobi(default).
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
HYPRE_Int HYPRE_BoomerAMGSetMultAdditive(HYPRE_Solver solver,
                                         HYPRE_Int    addlvl);

/**
 * (Optional) Defines use of an additive V(1,1)-cycle using the
 * simplified mult-additive method starting at level 'addlvl'.
 * The multiplicative approach is used on levels 0, ...'addlvl+1'.
 * 'addlvl' needs to be > -1 for this to have an effect.
 * Can only be used with weighted Jacobi and l1-Jacobi(default).
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
HYPRE_Int HYPRE_BoomerAMGSetSimple(HYPRE_Solver solver,
                                   HYPRE_Int    addlvl);

/**
 * (Optional) Defines last level where additive, mult-additive
 * or simple cycle is used.
 * The multiplicative approach is used on levels > add_last_lvl.
 *
 * Can only be used when AMG is used as a preconditioner !!!
 **/
HYPRE_Int HYPRE_BoomerAMGSetAddLastLvl(HYPRE_Solver solver,
                                       HYPRE_Int    add_last_lvl);

/**
 * (Optional) Defines the truncation factor for the
 * smoothed interpolation used for mult-additive or simple method.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMultAddTruncFactor(HYPRE_Solver solver,
                                               HYPRE_Real   add_trunc_factor);

/**
 * (Optional) Defines the maximal number of elements per row for the
 * smoothed interpolation used for mult-additive or simple method.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMultAddPMaxElmts(HYPRE_Solver solver,
                                             HYPRE_Int    add_P_max_elmts);
/**
 * (Optional) Defines the relaxation type used in the (mult)additive cycle
 * portion (also affects simple method.)
 * The default is 18 (L1-Jacobi).
 * Currently the only other option allowed is 0 (Jacobi) which should be
 * used in combination with HYPRE_BoomerAMGSetAddRelaxWt.
 **/
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxType(HYPRE_Solver solver,
                                         HYPRE_Int    add_rlx_type);

/**
 * (Optional) Defines the relaxation weight used for Jacobi within the
 * (mult)additive or simple cycle portion.
 * The default is 1.
 * The weight only affects the Jacobi method, and has no effect on L1-Jacobi
 **/
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxWt(HYPRE_Solver solver,
                                       HYPRE_Real    add_rlx_wt);

/**
 * (Optional) Sets maximal size for agglomeration or redundant coarse grid solve.
 * When the system is smaller than this threshold, sequential AMG is used
 * on process 0 or on all remaining active processes (if redundant = 1 ).
 **/
HYPRE_Int HYPRE_BoomerAMGSetSeqThreshold(HYPRE_Solver solver,
                                         HYPRE_Int    seq_threshold);
/**
 * (Optional) operates switch for redundancy. Needs to be used with
 * HYPRE_BoomerAMGSetSeqThreshold. Default is 0, i.e. no redundancy.
 **/
HYPRE_Int HYPRE_BoomerAMGSetRedundant(HYPRE_Solver solver,
                                      HYPRE_Int    redundant);

/**
 * (Optional) Defines the number of sweeps for the fine and coarse grid,
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE_BoomerAMGSetNumSweeps or HYPRE_BoomerAMGSetCycleNumSweeps instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumGridSweeps(HYPRE_Solver  solver,
                                          HYPRE_Int    *num_grid_sweeps);

/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and
 * the down cycle the number of sweeps are set to \e num_sweeps and on the
 * coarsest level to 1. The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumSweeps(HYPRE_Solver  solver,
                                      HYPRE_Int     num_sweeps);

/**
 * (Optional) Sets the number of sweeps at a specified cycle.
 * There are the following options for \e k:
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
HYPRE_Int HYPRE_BoomerAMGSetCycleNumSweeps(HYPRE_Solver  solver,
                                           HYPRE_Int     num_sweeps,
                                           HYPRE_Int     k);

/**
 * (Optional) Defines which smoother is used on the fine and coarse grid,
 * the up and down cycle.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE_BoomerAMGSetRelaxType or HYPRE_BoomerAMGSetCycleRelaxType instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxType(HYPRE_Solver  solver,
                                          HYPRE_Int    *grid_relax_type);

/**
 * (Optional) Defines the smoother to be used. It uses the given
 * smoother on the fine grid, the up and
 * the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is \f$\ell_1\f$-Gauss-Seidel, forward solve (13)
 * on the down cycle and backward solve (14) on the up cycle.
 *
 * There are the following options for \e relax_type:
 *
 *    - 0  : Jacobi
 *    - 1  : Gauss-Seidel, sequential (very slow!)
 *    - 2  : Gauss-Seidel, interior points in parallel, boundary sequential (slow!)
 *    - 3  : hybrid Gauss-Seidel or SOR, forward solve
 *    - 4  : hybrid Gauss-Seidel or SOR, backward solve
 *    - 5  : hybrid chaotic Gauss-Seidel (works only with OpenMP)
 *    - 6  : hybrid symmetric Gauss-Seidel or SSOR
 *    - 7  : Jacobi (uses Matvec)
 *    - 8  : \f$\ell_1\f$-scaled hybrid symmetric Gauss-Seidel
 *    - 9  : Gaussian elimination (only on coarsest level)
 *    - 10 : On-processor direct forward solve for matrices with
 *           triangular structure
 *    - 11 : Two Stage approximation to GS. Uses the strict lower
 *           part of the diagonal matrix
 *    - 12 : Two Stage approximation to GS. Uses the strict lower
 *           part of the diagonal matrix and a second iteration
 *           for additional error approximation
 *    - 13 : \f$\ell_1\f$ Gauss-Seidel, forward solve
 *    - 14 : \f$\ell_1\f$ Gauss-Seidel, backward solve
 *    - 15 : CG (warning - not a fixed smoother - may require FGMRES)
 *    - 16 : Chebyshev
 *    - 17 : FCF-Jacobi
 *    - 18 : \f$\ell_1\f$-scaled jacobi
 *    - 19 : Gaussian elimination (old version)
 *    - 21 : The same as 8 except forcing serialization on CPU (#OMP-thread = 1)
 *    - 29 : Direct solve: use Gaussian elimination & BLAS
 *                        (with pivoting) (old version)
 *    - 30 : Kaczmarz
 *    - 88:  The same methods as 8 with a convergent l1-term
 *    - 89:  Symmetric l1-hybrid Gauss-Seidel (i.e., 13 followed by 14)
 *    - 98 : LU with pivoting
 *    - 99 : LU with pivoting
 *    -199 : Matvec with the inverse
 **/
HYPRE_Int HYPRE_BoomerAMGSetRelaxType(HYPRE_Solver  solver,
                                      HYPRE_Int     relax_type);

/**
 * (Optional) Defines the smoother at a given cycle.
 *
 * For options of \e relax_type see description of HYPRE_BoomerAMGSetRelaxType.
 * In addition, the following options for \e relax_type are available when choosing
 * the coarsest level solver (k = 3):
 *
 *   For coarsest level systems formed via a sub-communicator defined with active ranks:
 *      - 9   : hypre's internal Gaussian elimination (host only).
 *      - 99  : LU factorization with pivoting.
 *      - 199 : explicit (dense) inverse.
 *
 *   For coarsest level systems formed via hypre_DataExchangeList:
 *      - 19  : hypre's internal Gaussian elimination (host only).
 *      - 98  : LU factorization with pivoting.
 *      - 198 : explicit (dense) inverse.
 *
 * Options for \e k are
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
HYPRE_Int HYPRE_BoomerAMGSetCycleRelaxType(HYPRE_Solver  solver,
                                           HYPRE_Int     relax_type,
                                           HYPRE_Int     k);

/**
 * (Optional) Defines in which order the points are relaxed. There are
 * the following options for \e relax_order:
 *
 *    - 0 : the points are relaxed in natural or lexicographic order on each processor
 *    - 1 : CF-relaxation is used, i.e on the fine grid and the down cycle the
 *          coarse points are relaxed first, followed by the fine points; on the
 *          up cycle the F-points are relaxed first, followed by the C-points.
 *          On the coarsest level, if an iterative scheme is used, the points
 *          are relaxed in lexicographic order.
 *
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetRelaxOrder(HYPRE_Solver  solver,
                                       HYPRE_Int     relax_order);

/**
 * (Optional) Defines in which order the points are relaxed.
 *
 * See also HYPRE_BoomerAMGSetRelaxOrder.
 **/
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxPoints(HYPRE_Solver   solver,
                                            HYPRE_Int    **grid_relax_points);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR.
 *
 * Note: This routine will be phased out!!!!
 * Use HYPRE_BoomerAMGSetRelaxWt or HYPRE_BoomerAMGSetLevelRelaxWt instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetRelaxWeight(HYPRE_Solver  solver,
                                        HYPRE_Real   *relax_weight);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on all levels.
 *
 * Values for \e relax_weight are
 *    - > 0  : this assigns the given relaxation weight on all levels
 *    - = 0  : the weight is determined on each level with the estimate
 *             \f$3 \over {4\|D^{-1/2}AD^{-1/2}\|}\f$, where \f$D\f$ is the diagonal of \f$A\f$
 *             (this should only be used with Jacobi)
 *    - = -k : the relaxation weight is determined with at most k CG steps on each level
 *             (this should only be used for symmetric positive definite problems)
 *
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetRelaxWt(HYPRE_Solver  solver,
                                    HYPRE_Real    relax_weight);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive \e relax_weight, the parameter is
 * determined on the given level as described for HYPRE_BoomerAMGSetRelaxWt.
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLevelRelaxWt(HYPRE_Solver  solver,
                                         HYPRE_Real    relax_weight,
                                         HYPRE_Int     level);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR.
 * Note: This routine will be phased out!!!!
 * Use HYPRE_BoomerAMGSetOuterWt or HYPRE_BoomerAMGSetLevelOuterWt instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetOmega(HYPRE_Solver  solver,
                                  HYPRE_Real   *omega);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR and SSOR
 * on all levels.
 *
 * Values for \e omega are
 *    - > 0  : this assigns the same outer relaxation weight omega on each level
 *    - = -k : an outer relaxation weight is determined with at most k CG steps on each level
 *             (this only makes sense for symmetric positive definite problems and smoothers
 *              such as SSOR)
 *
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetOuterWt(HYPRE_Solver  solver,
                                    HYPRE_Real    omega);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for HYPRE_BoomerAMGSetOuterWt.
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLevelOuterWt(HYPRE_Solver  solver,
                                         HYPRE_Real    omega,
                                         HYPRE_Int     level);

/**
 * (Optional) Defines the Order for Chebyshev smoother.
 *  The default is 2 (valid options are 1-4).
 **/
HYPRE_Int HYPRE_BoomerAMGSetChebyOrder(HYPRE_Solver solver,
                                       HYPRE_Int    order);

/**
 * (Optional) Fraction of the spectrum to use for the Chebyshev smoother.
 *  The default is .3 (i.e., damp on upper 30% of the spectrum).
 **/
HYPRE_Int HYPRE_BoomerAMGSetChebyFraction (HYPRE_Solver solver,
                                           HYPRE_Real   ratio);

/**
 * (Optional) Defines whether matrix should be scaled.
 *  The default is 1 (i.e., scaled).
 **/
HYPRE_Int HYPRE_BoomerAMGSetChebyScale (HYPRE_Solver solver,
                                        HYPRE_Int   scale);

/**
 * (Optional) Defines which polynomial variant should be used.
 *  The default is 0 (i.e., scaled).
 **/
HYPRE_Int HYPRE_BoomerAMGSetChebyVariant (HYPRE_Solver solver,
                                          HYPRE_Int   variant);

/**
 * (Optional) Defines how to estimate eigenvalues.
 *  The default is 10 (i.e., 10 CG iterations are used to find extreme
 *  eigenvalues.) If eig_est=0, the largest eigenvalue is estimated
 *  using Gershgorin, the smallest is set to 0.
 *  If eig_est is a positive number n, n iterations of CG are used to
 *  determine the smallest and largest eigenvalue.
 **/
HYPRE_Int HYPRE_BoomerAMGSetChebyEigEst (HYPRE_Solver solver,
                                         HYPRE_Int   eig_est);

/**
 * (Optional) Enables the use of more complex smoothers.
 * The following options exist for \e smooth_type:
 *
 *    - 4 : FSAI (routines needed to set: HYPRE_BoomerAMGSetFSAIMaxSteps,
 *          HYPRE_BoomerAMGSetFSAIMaxStepSize, HYPRE_BoomerAMGSetFSAIEigMaxIters,
 *          HYPRE_BoomerAMGSetFSAIKapTolerance)
 *    - 5 : ParILUK (routines needed to set: HYPRE_ILUSetLevelOfFill, HYPRE_ILUSetType)
 *    - 6 : Schwarz (routines needed to set: HYPRE_BoomerAMGSetDomainType,
 *          HYPRE_BoomerAMGSetOverlap, HYPRE_BoomerAMGSetVariant,
 *          HYPRE_BoomerAMGSetSchwarzRlxWeight)
 *    - 7 : Pilut (routines needed to set: HYPRE_BoomerAMGSetDropTol,
 *          HYPRE_BoomerAMGSetMaxNzPerRow)
 *    - 8 : ParaSails (routines needed to set: HYPRE_BoomerAMGSetSym,
 *          HYPRE_BoomerAMGSetLevel, HYPRE_BoomerAMGSetFilter,
 *          HYPRE_BoomerAMGSetThreshold)
 *    - 9 : Euclid (routines needed to set: HYPRE_BoomerAMGSetEuclidFile)
 *
 * The default is 6.  Also, if no smoother parameters are set via the routines
 * mentioned in the table above, default values are used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSmoothType(HYPRE_Solver solver,
                                       HYPRE_Int    smooth_type);

/**
 * (Optional) Sets the number of levels for more complex smoothers.
 * The smoothers,
 * as defined by HYPRE_BoomerAMGSetSmoothType, will be used
 * on level 0 (the finest level) through level \e smooth_num_levels-1.
 * The default is 0, i.e. no complex smoothers are used.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumLevels(HYPRE_Solver solver,
                                            HYPRE_Int    smooth_num_levels);

/**
 * (Optional) Sets the number of sweeps for more complex smoothers.
 * The default is 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumSweeps(HYPRE_Solver solver,
                                            HYPRE_Int    smooth_num_sweeps);

/**
 * (Optional) Defines which variant of the Schwarz method is used.
 * The following options exist for \e variant:
 *
 *    - 0 : hybrid multiplicative Schwarz method (no overlap across processor boundaries)
 *    - 1 : hybrid additive Schwarz method (no overlap across processor boundaries)
 *    - 2 : additive Schwarz method
 *    - 3 : hybrid multiplicative Schwarz method (with overlap across processor boundaries)
 *
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetVariant(HYPRE_Solver solver,
                                    HYPRE_Int    variant);

/**
 * (Optional) Defines the overlap for the Schwarz method.
 * The following options exist for overlap:
 *
 *    - 0 : no overlap
 *    - 1 : minimal overlap (default)
 *    - 2 : overlap generated by including all neighbors of domain boundaries
 **/
HYPRE_Int HYPRE_BoomerAMGSetOverlap(HYPRE_Solver solver,
                                    HYPRE_Int    overlap);

/**
 * (Optional) Defines the type of domain used for the Schwarz method.
 * The following options exist for \e domain_type:
 *
 *    - 0 : each point is a domain
 *    - 1 : each node is a domain (only of interest in "systems" AMG)
 *    - 2 : each domain is generated by agglomeration (default)
 **/
HYPRE_Int HYPRE_BoomerAMGSetDomainType(HYPRE_Solver solver,
                                       HYPRE_Int    domain_type);

/**
 * (Optional) Defines a smoothing parameter for the additive Schwarz method.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSchwarzRlxWeight(HYPRE_Solver solver,
                                             HYPRE_Real   schwarz_rlx_weight);

/**
 *  (Optional) Indicates that the aggregates may not be SPD for the Schwarz method.
 * The following options exist for \e use_nonsymm:
 *
 *    - 0 : assume SPD (default)
 *    - 1 : assume non-symmetric
**/
HYPRE_Int HYPRE_BoomerAMGSetSchwarzUseNonSymm(HYPRE_Solver solver,
                                              HYPRE_Int    use_nonsymm);

/**
 * (Optional) Defines symmetry for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetSym(HYPRE_Solver solver,
                                HYPRE_Int    sym);

/**
 * (Optional) Defines number of levels for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLevel(HYPRE_Solver solver,
                                  HYPRE_Int    level);

/**
 * (Optional) Defines threshold for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetThreshold(HYPRE_Solver solver,
                                      HYPRE_Real   threshold);

/**
 * (Optional) Defines filter for ParaSAILS.
 * For further explanation see description of ParaSAILS.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFilter(HYPRE_Solver solver,
                                   HYPRE_Real   filter);

/**
 * (Optional) Defines drop tolerance for PILUT.
 * For further explanation see description of PILUT.
 **/
HYPRE_Int HYPRE_BoomerAMGSetDropTol(HYPRE_Solver solver,
                                    HYPRE_Real   drop_tol);

/**
 * (Optional) Defines maximal number of nonzeros for PILUT.
 * For further explanation see description of PILUT.
 **/
HYPRE_Int HYPRE_BoomerAMGSetMaxNzPerRow(HYPRE_Solver solver,
                                        HYPRE_Int    max_nz_per_row);

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
                                    HYPRE_Int    eu_level);

/**
 * (Optional) Defines filter for ILU(k) for Euclid.
 * For further explanation see description of Euclid.
 **/
HYPRE_Int HYPRE_BoomerAMGSetEuSparseA(HYPRE_Solver solver,
                                      HYPRE_Real   eu_sparse_A);

/**
 * (Optional) Defines use of block jacobi ILUT for Euclid.
 * For further explanation see description of Euclid.
 **/
HYPRE_Int HYPRE_BoomerAMGSetEuBJ(HYPRE_Solver solver,
                                 HYPRE_Int    eu_bj);


/**
 * Defines type of ILU smoother to use
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUType( HYPRE_Solver  solver,
                                     HYPRE_Int     ilu_type);

/**
 * Defines level k for ILU(k) smoother
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILULevel( HYPRE_Solver  solver,
                                      HYPRE_Int     ilu_lfil);

/**
 * Defines max row nonzeros for ILUT smoother
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUMaxRowNnz( HYPRE_Solver  solver,
                                          HYPRE_Int     ilu_max_row_nnz);

/**
 * Defines number of iterations for ILU smoother on each level
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUMaxIter( HYPRE_Solver  solver,
                                        HYPRE_Int     ilu_max_iter);

/**
 * Defines drop tolorance for iLUT smoother
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUDroptol( HYPRE_Solver  solver,
                                        HYPRE_Real    ilu_droptol);

/**
 * (Optional) Defines triangular solver for ILU(k,T) smoother: 0-iterative, 1-direct (default)
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUTriSolve( HYPRE_Solver  solver,
                                         HYPRE_Int     ilu_tri_solve);

/**
 * (Optional) Defines number of lower Jacobi iterations for ILU(k,T) smoother triangular solve.
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILULowerJacobiIters( HYPRE_Solver  solver,
                                                 HYPRE_Int     ilu_lower_jacobi_iters);

/**
 * (Optional) Defines number of upper Jacobi iterations for ILU(k,T) smoother triangular solve.
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUUpperJacobiIters( HYPRE_Solver  solver,
                                                 HYPRE_Int     ilu_upper_jacobi_iters);

/**
 * (Optional) Set Local Reordering paramter (1==RCM, 0==None)
 * For further explanation see description of ILU.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILULocalReordering( HYPRE_Solver solver,
                                                HYPRE_Int    ilu_reordering_type);

/**
 * (Optional) Set iterative ILU's algorithm type.
 * For further explanation see \e HYPRE_ILUSetIterativeSetupType.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUIterSetupType( HYPRE_Solver solver,
                                              HYPRE_Int    ilu_iter_setup_type);

/**
 * (Optional) Set iterative ILU's option.
 * For further explanation see \e HYPRE_ILUSetIterativeSetupOption.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUIterSetupOption( HYPRE_Solver solver,
                                                HYPRE_Int    ilu_iter_setup_option);

/**
 * (Optional) Set iterative ILU's max. number of iterations.
 * For further explanation see \e HYPRE_ILUSetIterativeSetupMaxIter.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUIterSetupMaxIter( HYPRE_Solver solver,
                                                 HYPRE_Int    ilu_iter_setup_max_iter);

/**
 * (Optional) Set iterative ILU's tolerance.
 * For further explanation see \e HYPRE_ILUSetIterativeSetupTolerance.
 **/
HYPRE_Int HYPRE_BoomerAMGSetILUIterSetupTolerance( HYPRE_Solver solver,
                                                   HYPRE_Real    ilu_iter_setup_tolerance);

/**
 * (Optional) Defines the algorithm type for setting up FSAI
 * For further explanation see \e HYPRE_FSAISetAlgoType.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAIAlgoType(HYPRE_Solver solver,
                                         HYPRE_Int    algo_type);

/**
 * (Optional) Sets the solver type for solving local linear systems in FSAI.
 * For further explanation see \e HYPRE_FSAISetLocalSolveType.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAILocalSolveType( HYPRE_Solver solver,
                                                HYPRE_Int    local_solve_type );

/**
 * (Optional) Defines maximum number of steps for FSAI.
 * For further explanation see \e HYPRE_FSAISetMaxSteps.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxSteps(HYPRE_Solver solver,
                                         HYPRE_Int    max_steps);

/**
 * (Optional) Defines maximum step size for FSAI.
 * For further explanation see \e HYPRE_FSAISetMaxStepSize.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxStepSize(HYPRE_Solver solver,
                                            HYPRE_Int    max_step_size);

/**
 * (Optional) Defines maximum number of nonzero entries per row for FSAI.
 * For further explanation see \e HYPRE_FSAISetMaxNnzRow.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxNnzRow(HYPRE_Solver solver,
                                          HYPRE_Int    max_nnz_row);

/**
 * (Optional) Defines number of levels for computing the candidate pattern for FSAI
 * For further explanation see \e HYPRE_FSAISetNumLevels.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAINumLevels(HYPRE_Solver solver,
                                          HYPRE_Int    num_levels);

/**
 * (Optional) Defines the threshold for computing the candidate pattern for FSAI
 * For further explanation see \e HYPRE_FSAISetThreshold.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAIThreshold(HYPRE_Solver solver,
                                          HYPRE_Real   threshold);

/**
 * (Optional) Defines maximum number of iterations for estimating the
 * largest eigenvalue of the FSAI preconditioned matrix (G^T * G * A).
 * For further explanation see \e HYPRE_FSAISetEigMaxIters.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAIEigMaxIters(HYPRE_Solver solver,
                                            HYPRE_Int    eig_max_iters);

/**
 * (Optional) Defines the kaporin dropping tolerance.
 * For further explanation see \e HYPRE_FSAISetKapTolerance.
 **/
HYPRE_Int HYPRE_BoomerAMGSetFSAIKapTolerance(HYPRE_Solver solver,
                                             HYPRE_Real   kap_tolerance);

/**
 * (Optional) Defines which parallel restriction operator is used.
 * There are the following options for restr_type:
 *
 *    - 0 : \f$P^T\f$ - Transpose of the interpolation operator
 *    - 1 : AIR-1 - Approximate Ideal Restriction (distance 1)
 *    - 2 : AIR-2 - Approximate Ideal Restriction (distance 2)
 *
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetRestriction(HYPRE_Solver solver,
                                        HYPRE_Int    restr_par);

/**
 * (Optional) Assumes the matrix is triangular in some ordering
 * to speed up the setup time of approximate ideal restriction.
 *
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetIsTriangular(HYPRE_Solver solver,
                                         HYPRE_Int   is_triangular);

/**
 * (Optional) Set local problem size at which GMRES is used over
 * a direct solve in approximating ideal restriction.
 * The default is 0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR(HYPRE_Solver solver,
                                         HYPRE_Int   gmres_switch);

/**
 * (Optional) Defines the drop tolerance for the A-matrices
 * from the 2nd level of AMG.
 * The default is 0.
 **/
HYPRE_Int
HYPRE_BoomerAMGSetADropTol( HYPRE_Solver  solver,
                            HYPRE_Real    A_drop_tol  );

/**
 * (Optional) Drop the entries that are not on the diagonal and smaller than
 * its row norm: type 1: 1-norm, 2: 2-norm, -1: infinity norm
 **/
HYPRE_Int
HYPRE_BoomerAMGSetADropType( HYPRE_Solver  solver,
                             HYPRE_Int     A_drop_type  );

/**
 * (Optional) Name of file to which BoomerAMG will print;
 * cf HYPRE_BoomerAMGSetPrintLevel.  (Presently this is ignored).
 **/
HYPRE_Int HYPRE_BoomerAMGSetPrintFileName(HYPRE_Solver  solver,
                                          const char   *print_file_name);

/**
 * (Optional) Requests automatic printing of setup and solve information.
 *
 *    - 0 : no printout (default)
 *    - 1 : print setup information
 *    - 2 : print solve information
 *    - 3 : print both setup and solve information
 *
 * Note, that if one desires to print information and uses BoomerAMG as a
 * preconditioner, suggested \e print_level is 1 to avoid excessive output,
 * and use \e print_level of solver for solve phase information.
 **/
HYPRE_Int HYPRE_BoomerAMGSetPrintLevel(HYPRE_Solver solver,
                                       HYPRE_Int    print_level);

/**
 * (Optional) Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default to 0 to do nothing.  The latest
 * residual will be available if logging > 1.
 **/
HYPRE_Int HYPRE_BoomerAMGSetLogging(HYPRE_Solver solver,
                                    HYPRE_Int    logging);


/**
 * (Optional)
 **/
HYPRE_Int HYPRE_BoomerAMGSetDebugFlag(HYPRE_Solver solver,
                                      HYPRE_Int    debug_flag);

/**
 * (Optional) This routine will be eliminated in the future.
 **/
HYPRE_Int HYPRE_BoomerAMGInitGridRelaxation(HYPRE_Int    **num_grid_sweeps_ptr,
                                            HYPRE_Int    **grid_relax_type_ptr,
                                            HYPRE_Int   ***grid_relax_points_ptr,
                                            HYPRE_Int      coarsen_type,
                                            HYPRE_Real **relax_weights_ptr,
                                            HYPRE_Int      max_levels);

/**
 * (Optional) If rap2 not equal 0, the triple matrix product RAP is
 * replaced by two matrix products.
 * (Required for triple matrix product generation on GPUs)
 **/
HYPRE_Int HYPRE_BoomerAMGSetRAP2(HYPRE_Solver solver,
                                 HYPRE_Int    rap2);

/**
 * (Optional) If mod_rap2 not equal 0, the triple matrix product RAP is
 * replaced by two matrix products with modularized kernels
 * (Required for triple matrix product generation on GPUs)
 **/
HYPRE_Int HYPRE_BoomerAMGSetModuleRAP2(HYPRE_Solver solver,
                                       HYPRE_Int    mod_rap2);

/**
 * (Optional) If set to 1, the local interpolation transposes will
 * be saved to use more efficient matvecs instead of matvecTs
 * (Recommended for efficient use on GPUs)
 **/
HYPRE_Int HYPRE_BoomerAMGSetKeepTranspose(HYPRE_Solver solver,
                                          HYPRE_Int    keepTranspose);

/**
 * HYPRE_BoomerAMGSetPlotGrids
 **/
HYPRE_Int HYPRE_BoomerAMGSetPlotGrids (HYPRE_Solver solver,
                                       HYPRE_Int    plotgrids);

/**
 * HYPRE_BoomerAMGSetPlotFilename
 **/
HYPRE_Int HYPRE_BoomerAMGSetPlotFileName (HYPRE_Solver  solver,
                                          const char   *plotfilename);

/**
 * HYPRE_BoomerAMGSetCoordDim
 **/
HYPRE_Int HYPRE_BoomerAMGSetCoordDim (HYPRE_Solver solver,
                                      HYPRE_Int    coorddim);

/**
 * HYPRE_BoomerAMGSetCoordinates
 **/
HYPRE_Int HYPRE_BoomerAMGSetCoordinates (HYPRE_Solver  solver,
                                         float        *coordinates);

/**
 * (Optional) Get the coarse grid hierarchy. Assumes input/ output array is
 * preallocated to the size of the local matrix. On return, \e cgrid[i] returns
 * the last grid level containing node \e i.
 *
 * @param solver [IN] solver or preconditioner
 * @param cgrid [IN/ OUT] preallocated array. On return, contains grid hierarchy info.
 **/
HYPRE_Int HYPRE_BoomerAMGGetGridHierarchy(HYPRE_Solver solver,
                                          HYPRE_Int *cgrid );

#ifdef HYPRE_USING_DSUPERLU
/**
 * HYPRE_BoomerAMGSetDSLUThreshold
 *
 * Usage:
 *  Set slu_threshold >= max_coarse_size (from HYPRE_BoomerAMGSetMaxCoarseSize(...))
 *  to turn on use of superLU for the coarse grid solve. SuperLU is used if the
 *  coarse grid size > max_coarse_size and the grid level is < (max_num_levels - 1)
 *  (set with HYPRE_BoomerAMGSetMaxLevels(...)).
 **/

HYPRE_Int HYPRE_BoomerAMGSetDSLUThreshold (HYPRE_Solver solver,
                                           HYPRE_Int    slu_threshold);
#endif

/**
 * (Optional) Fix C points to be kept till a specified coarse level.
 *
 * @param solver [IN] solver or preconditioner
 * @param cpt_coarse_level [IN] coarse level up to which to keep C points
 * @param num_cpt_coarse [IN] number of C points to be kept
 * @param cpt_coarse_index [IN] indexes of C points to be kept
 **/
HYPRE_Int HYPRE_BoomerAMGSetCPoints(HYPRE_Solver  solver,
                                    HYPRE_Int     cpt_coarse_level,
                                    HYPRE_Int     num_cpt_coarse,
                                    HYPRE_BigInt *cpt_coarse_index);

/**
 * (Optional) Deprecated function. Use HYPRE_BoomerAMGSetCPoints instead.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCpointsToKeep(HYPRE_Solver solver,
                                          HYPRE_Int  cpt_coarse_level,
                                          HYPRE_Int  num_cpt_coarse,
                                          HYPRE_BigInt *cpt_coarse_index);

/**
 * (Optional) Set fine points in the first level.
 *
 * @param solver [IN] solver or preconditioner
 * @param num_fpt [IN] number of fine points
 * @param fpt_index [IN] global indices of fine points
 **/
HYPRE_Int HYPRE_BoomerAMGSetFPoints(HYPRE_Solver  solver,
                                    HYPRE_Int     num_fpt,
                                    HYPRE_BigInt *fpt_index);

/**
 * (Optional) Set isolated fine points in the first level.
 * Interpolation weights are not computed for these points.
 *
 * @param solver [IN] solver or preconditioner
 * @param num_isolated_fpt [IN] number of isolated fine points
 * @param isolated_fpt_index [IN] global indices of isolated fine points
 **/
HYPRE_Int HYPRE_BoomerAMGSetIsolatedFPoints(HYPRE_Solver  solver,
                                            HYPRE_Int     num_isolated_fpt,
                                            HYPRE_BigInt *isolated_fpt_index);

/**
 * (Optional) if Sabs equals 1, the strength of connection test is based
 * on the absolute value of the matrix coefficients
 **/
HYPRE_Int HYPRE_BoomerAMGSetSabs (HYPRE_Solver solver,
                                  HYPRE_Int Sabs );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BoomerAMGDD Solver and Preconditioner
 *
 * Communication reducing solver and preconditioner built on top of algebraic multigrid
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_BoomerAMGDDCreate( HYPRE_Solver *solver );

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_BoomerAMGDDDestroy( HYPRE_Solver solver );

/**
 * Set up the BoomerAMGDD solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
HYPRE_Int HYPRE_BoomerAMGDDSetup( HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x );

/**
 * Solve the system or apply AMG-DD as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
HYPRE_Int HYPRE_BoomerAMGDDSolve( HYPRE_Solver       solver,
                                  HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector    b,
                                  HYPRE_ParVector    x );

/**
 * (Optional) Set the number of pre- and post-relaxations per level for
 * AMG-DD inner FAC cycles. Default is 1.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetFACNumRelax( HYPRE_Solver solver,
                                 HYPRE_Int    amgdd_fac_num_relax );

/**
 * (Optional) Set the number of inner FAC cycles per AMG-DD iteration.
 * Default is 2.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetFACNumCycles( HYPRE_Solver solver,
                                  HYPRE_Int    amgdd_fac_num_cycles );

/**
 * (Optional) Set the cycle type for the AMG-DD inner FAC cycles.
 * 1 (default) = V-cycle, 2 = W-cycle, 3 = F-cycle
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetFACCycleType( HYPRE_Solver solver,
                                  HYPRE_Int    amgdd_fac_cycle_type );

/**
 * (Optional) Set the relaxation type for the AMG-DD inner FAC cycles.
 * 0 = Jacobi, 1 = Gauss-Seidel, 2 = ordered Gauss-Seidel, 3 (default) = C/F L1-scaled Jacobi
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetFACRelaxType( HYPRE_Solver solver,
                                  HYPRE_Int    amgdd_fac_relax_type );

/**
 * (Optional) Set the relaxation weight for the AMG-DD inner FAC cycles. Default is 1.0.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetFACRelaxWeight( HYPRE_Solver solver,
                                    HYPRE_Real   amgdd_fac_relax_weight );

/**
 * (Optional) Set the AMG-DD start level. Default is 0.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetStartLevel( HYPRE_Solver solver,
                                HYPRE_Int    start_level );

/**
 * (Optional) Set the AMG-DD padding. Default is 1.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetPadding( HYPRE_Solver solver,
                             HYPRE_Int    padding );

/**
 * (Optional) Set the AMG-DD number of ghost layers. Default is 1.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetNumGhostLayers( HYPRE_Solver solver,
                                    HYPRE_Int    num_ghost_layers );

/**
 * (Optional) Pass a custom user-defined function as a relaxation method for the AMG-DD FAC cycles.
 * Function should have the following form, where amgdd_solver is of type hypre_ParAMGDDData* and level is the level on which to relax:
 * HYPRE_Int userFACRelaxation( HYPRE_Solver amgdd_solver, HYPRE_Int level )
 **/
HYPRE_Int
HYPRE_BoomerAMGDDSetUserFACRelaxation( HYPRE_Solver solver,
                                       HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );

/**
 * (Optional) Get the underlying AMG hierarchy as a HYPRE_Solver object.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDGetAMG( HYPRE_Solver  solver,
                         HYPRE_Solver *amg_solver );

/**
 * Returns the norm of the final relative residual.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               HYPRE_Real   *rel_resid_norm );

/**
 * Returns the number of iterations taken.
 **/
HYPRE_Int
HYPRE_BoomerAMGDDGetNumIterations( HYPRE_Solver   solver,
                                   HYPRE_Int     *num_iterations );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR FSAI Solver and Preconditioner
 *
 * An adaptive factorized sparse approximate inverse solver/preconditioner/smoother
 * that computes a sparse approximation G to the inverse of the lower cholesky
 * factor of A such that M^{-1} \approx G^T * G.
 *
 * @{
 **/

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_FSAICreate( HYPRE_Solver *solver );

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_FSAIDestroy( HYPRE_Solver solver );

/**
 * Set up the FSAI solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
HYPRE_Int HYPRE_FSAISetup( HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x );

/**
 * Solve the system or apply FSAI as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
HYPRE_Int HYPRE_FSAISolve( HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix A,
                           HYPRE_ParVector    b,
                           HYPRE_ParVector    x );

/**
 * (Optional) Sets the algorithm type used to compute the lower triangular factor G
 *
 *      - 1: Adaptive (can use OpenMP with static scheduling)
 *      - 2: Adaptive OpenMP with dynamic scheduling
 *      - 3: Static - power pattern
 **/
HYPRE_Int HYPRE_FSAISetAlgoType( HYPRE_Solver solver,
                                 HYPRE_Int    algo_type );

/**
 * (Optional) Sets the solver type for solving local linear systems in FSAI. This
 * option makes sense only for GPU runs.
 *
 *      - 0: Gauss-Jordan solver
 *      - 1: Vendor solver (cuSOLVER/rocSOLVER)
 *      - 2: MAGMA solver
 **/
HYPRE_Int HYPRE_FSAISetLocalSolveType( HYPRE_Solver solver,
                                       HYPRE_Int    local_solve_type );

/**
 * (Optional) Sets the maximum number of steps for computing the sparsity
 * pattern of G. This input parameter makes sense when using adaptive FSAI,
 * i.e., algorithm type 1 or 2.
 **/
HYPRE_Int HYPRE_FSAISetMaxSteps( HYPRE_Solver solver,
                                 HYPRE_Int    max_steps );

/**
 * (Optional) Sets the maximum step size for computing the sparsity pattern of G.
 * This input parameter makes sense when using adaptive FSAI, i.e., algorithm
 * type 1 or 2.
 **/
HYPRE_Int HYPRE_FSAISetMaxStepSize( HYPRE_Solver solver,
                                    HYPRE_Int    max_step_size );

/**
 * (Optional) Sets the maximum number of off-diagonal entries per row of G.
 * This input parameter makes sense when using static FSAI, i.e., algorithm
 * type 3.
 **/
HYPRE_Int HYPRE_FSAISetMaxNnzRow( HYPRE_Solver solver,
                                  HYPRE_Int    max_nnz_row );

/**
 * (Optional) Sets the number of levels for computing the candidate pattern of G.
 * This input parameter must be a positive integer and it makes sense
 * when using static FSAI, i.e., algorithm type 3.
 **/
HYPRE_Int HYPRE_FSAISetNumLevels( HYPRE_Solver solver,
                                  HYPRE_Int    num_levels );

/**
 * (Optional) Sets the threshold for computing the candidate pattern of G
 * This input parameter makes sense when using static FSAI, i.e., algorithm
 * type 3.
 **/
HYPRE_Int HYPRE_FSAISetThreshold( HYPRE_Solver solver,
                                  HYPRE_Real   threshold );

/**
 * (Optional) Sets the kaporin gradient reduction factor for computing the
 * sparsity pattern of G. This input parameter makes sense when using adaptive
 * FSAI, i.e., algorithm types 1 or 2.
 **/
HYPRE_Int HYPRE_FSAISetKapTolerance( HYPRE_Solver solver,
                                     HYPRE_Real   kap_tolerance );

/**
 * (Optional) Sets the relaxation factor for FSAI. This input parameter makes
 * sense to all algorithm types for setting up FSAI.
 **/
HYPRE_Int HYPRE_FSAISetOmega( HYPRE_Solver solver,
                              HYPRE_Real   omega );

/**
 * (Optional) Sets the maximum number of iterations (sweeps) for FSAI. This
 * input parameter makes sense to all algorithm types for setting up FSAI.
 **/
HYPRE_Int HYPRE_FSAISetMaxIterations( HYPRE_Solver solver,
                                      HYPRE_Int    max_iterations );

/**
 * (Optional) Set number of iterations for computing maximum
 * eigenvalue of the preconditioned operator. This input parameter makes
 * sense to all algorithm types for setting up FSAI.
 **/
HYPRE_Int HYPRE_FSAISetEigMaxIters( HYPRE_Solver solver,
                                    HYPRE_Int    eig_max_iters );

/**
 * (Optional) Set the convergence tolerance, if FSAI is used
 * as a solver. This input parameter makes sense to all algorithm types
 * for setting up FSAI. When using FSAI as a preconditioner, set the
 * tolerance to 0.0. The default is \f$10^{-6}\f$.
 **/
HYPRE_Int HYPRE_FSAISetTolerance( HYPRE_Solver solver,
                                  HYPRE_Real   tolerance );

/**
 * (Optional) Requests automatic printing of setup information.
 *
 *    - 0 : no printout (default)
 *    - 1 : print setup information
 **/
HYPRE_Int HYPRE_FSAISetPrintLevel(HYPRE_Solver solver,
                                  HYPRE_Int    print_level);

/**
 * (Optional) Use a zero initial guess. This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
HYPRE_Int HYPRE_FSAISetZeroGuess(HYPRE_Solver solver,
                                 HYPRE_Int    zero_guess);


/**@}*/


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR ParaSails Preconditioner
 *
 * Parallel sparse approximate inverse preconditioner for the
 * ParCSR matrix format.
 *
 * @{
 **/

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
 * to the iterative solver \e SetPrecond function.
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
 * to the iterative solver \e SetPrecond function.
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
 * @param thresh [IN] Value of threshold parameter, \f$0 \le\f$ thresh \f$\le 1\f$.
 *                    The default value is 0.1.
 * @param nlevels [IN] Value of levels parameter, \f$0 \le\f$ nlevels.
 *                     The default value is 1.
 **/
HYPRE_Int HYPRE_ParaSailsSetParams(HYPRE_Solver solver,
                                   HYPRE_Real   thresh,
                                   HYPRE_Int    nlevels);
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
                                   HYPRE_Real   filter);

/**
 * Set the symmetry parameter for the ParaSails preconditioner.
 *
 * Values for \e sym
 *    - 0 : nonsymmetric and/or indefinite problem, and nonsymmetric preconditioner
 *    - 1 : SPD problem, and SPD (factored) preconditioner
 *    - 2 : nonsymmetric, definite problem, and SPD (factored) preconditioner
 *
 * @param solver [IN] Preconditioner object for which to set symmetry parameter.
 * @param sym [IN] Symmetry parameter.
 **/
HYPRE_Int HYPRE_ParaSailsSetSym(HYPRE_Solver solver,
                                HYPRE_Int    sym);

/**
 * Set the load balance parameter for the
 * ParaSails preconditioner.
 *
 * @param solver [IN] Preconditioner object for which to set the load balance
 *                    parameter.
 * @param loadbal [IN] Value of the load balance parameter,
 *                     \f$0 \le\f$ loadbal \f$\le 1\f$.  A zero value indicates that
 *                     no load balance is attempted; a value of unity indicates
 *                     that perfect load balance will be attempted.  The
 *                     recommended value is 0.9 to balance the overhead of
 *                     data exchanges for load balancing.  No load balancing
 *                     is needed if the preconditioner is very sparse and
 *                     fast to construct.  The default value when this
 *                     parameter is not set is 0.
 **/
HYPRE_Int HYPRE_ParaSailsSetLoadbal(HYPRE_Solver solver,
                                    HYPRE_Real   loadbal);

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
                                  HYPRE_Int    reuse);

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
                                    HYPRE_Int    logging);

/**
 * Build IJ Matrix of the sparse approximate inverse (factor).
 * This function explicitly creates the IJ Matrix corresponding to
 * the sparse approximate inverse or the inverse factor.
 * Example:  HYPRE_IJMatrix ij_A;
 *           HYPRE_ParaSailsBuildIJMatrix(solver, \&ij_A);
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
                                         HYPRE_Real   thresh,
                                         HYPRE_Int    nlevels);

HYPRE_Int HYPRE_ParCSRParaSailsSetFilter(HYPRE_Solver solver,
                                         HYPRE_Real   filter);

HYPRE_Int HYPRE_ParCSRParaSailsSetSym(HYPRE_Solver solver,
                                      HYPRE_Int    sym);

HYPRE_Int HYPRE_ParCSRParaSailsSetLoadbal(HYPRE_Solver solver,
                                          HYPRE_Real   loadbal);

HYPRE_Int HYPRE_ParCSRParaSailsSetReuse(HYPRE_Solver solver,
                                        HYPRE_Int    reuse);

HYPRE_Int HYPRE_ParCSRParaSailsSetLogging(HYPRE_Solver solver,
                                          HYPRE_Int    logging);

/**@}*/

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Euclid Preconditioner
 *
 * MPI Parallel ILU preconditioner
 *
 * Options summary:
 *
 *    | Option    | Default   | Synopsis                                      |
 *    | :-------- | --------- | :-------------------------------------------- |
 *    | -level    | 1         | ILU(k) factorization level                    |
 *    | -bj       | 0 (false) | Use Block Jacobi ILU instead of PILU          |
 *    | -eu_stats | 0 (false) | Print  internal timing and statistics         |
 *    | -eu_mem   | 0 (false) | Print  internal memory usage                  |
 *
 * @{
 **/

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
 * to the iterative solver \e SetPrecond function.
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
 * to the iterative solver \e SetPrecond function.
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
 * See also: HYPRE_EuclidSetParamsFromFile.
 *
 * @param argc [IN] Length of argv array
 * @param argv [IN] Array of strings
 **/
HYPRE_Int HYPRE_EuclidSetParams(HYPRE_Solver  solver,
                                HYPRE_Int     argc,
                                char         *argv[]);

/**
 * Insert (name, value) pairs in Euclid's options database.
 * Each line of the file should either begin with a "\#",
 * indicating a comment line, or contain a (name value)
 * pair, e.g:
 *
   \verbatim
   >cat optionsFile
   \#sample runtime parameter file
   -blockJacobi 3
   -matFile     /home/hysom/myfile.euclid
   -doSomething true
   -xx_coeff -1.0
   \endverbatim
 *
 * See also: HYPRE_EuclidSetParams.
 *
 * @param filename[IN] Pathname/filename to read
 **/
HYPRE_Int HYPRE_EuclidSetParamsFromFile(HYPRE_Solver  solver,
                                        char         *filename);

/**
 * Set level k for ILU(k) factorization, default: 1
 **/
HYPRE_Int HYPRE_EuclidSetLevel(HYPRE_Solver solver,
                               HYPRE_Int    level);

/**
 * Use block Jacobi ILU preconditioning instead of PILU
 **/
HYPRE_Int HYPRE_EuclidSetBJ(HYPRE_Solver solver,
                            HYPRE_Int    bj);

/**
 * If \e eu_stats not equal 0, a summary of runtime settings and
 * timing information is printed to stdout.
 **/
HYPRE_Int HYPRE_EuclidSetStats(HYPRE_Solver solver,
                               HYPRE_Int    eu_stats);

/**
 * If \e eu_mem not equal 0, a summary of Euclid's memory usage
 * is printed to stdout.
 **/
HYPRE_Int HYPRE_EuclidSetMem(HYPRE_Solver solver,
                             HYPRE_Int    eu_mem);

/**
 * Defines a drop tolerance for ILU(k). Default: 0
 * Use with HYPRE_EuclidSetRowScale.
 * Note that this can destroy symmetry in a matrix.
 **/
HYPRE_Int HYPRE_EuclidSetSparseA(HYPRE_Solver solver,
                                 HYPRE_Real   sparse_A);

/**
 * If \e row_scale not equal 0, values are scaled prior to factorization
 * so that largest value in any row is +1 or -1.
 * Note that this can destroy symmetry in a matrix.
 **/
HYPRE_Int HYPRE_EuclidSetRowScale(HYPRE_Solver solver,
                                  HYPRE_Int    row_scale);

/**
 * uses ILUT and defines a drop tolerance relative to the largest
 * absolute value of any entry in the row being factored.
 **/
HYPRE_Int HYPRE_EuclidSetILUT(HYPRE_Solver solver,
                              HYPRE_Real   drop_tol);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Pilut Preconditioner
 *
 * @{
 **/

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
                                      HYPRE_Int    max_iter);

/**
 * (Optional)
 **/
HYPRE_Int HYPRE_ParCSRPilutSetDropTolerance(HYPRE_Solver solver,
                                            HYPRE_Real   tol);

/**
 * (Optional)
 **/
HYPRE_Int HYPRE_ParCSRPilutSetFactorRowSize(HYPRE_Solver solver,
                                            HYPRE_Int    size);


HYPRE_Int HYPRE_ParCSRPilutSetLogging(HYPRE_Solver solver,
                                      HYPRE_Int    logging );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR AMS Solver and Preconditioner
 *
 * Parallel auxiliary space Maxwell solver and preconditioner
 *
 * @{
 **/

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
 * to the iterative solver \e SetPrecond function.
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
 * to the iterative solver \e SetPrecond function.
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
                                HYPRE_Int    dim);

/**
 * Sets the discrete gradient matrix \e G.
 * This function should be called before HYPRE_AMSSetup()!
 **/
HYPRE_Int HYPRE_AMSSetDiscreteGradient(HYPRE_Solver       solver,
                                       HYPRE_ParCSRMatrix G);

/**
 * Sets the \e x, \e y and \e z coordinates of the vertices in the mesh.
 *
 * Either HYPRE_AMSSetCoordinateVectors() or HYPRE_AMSSetEdgeConstantVectors()
 * should be called before HYPRE_AMSSetup()!
 **/
HYPRE_Int HYPRE_AMSSetCoordinateVectors(HYPRE_Solver    solver,
                                        HYPRE_ParVector x,
                                        HYPRE_ParVector y,
                                        HYPRE_ParVector z);

/**
 * Sets the vectors \e Gx, \e Gy and \e Gz which give the representations of
 * the constant vector fields (1,0,0), (0,1,0) and (0,0,1) in the
 * edge element basis.
 *
 * Either HYPRE_AMSSetCoordinateVectors() or HYPRE_AMSSetEdgeConstantVectors()
 * should be called before HYPRE_AMSSetup()!
 **/
HYPRE_Int HYPRE_AMSSetEdgeConstantVectors(HYPRE_Solver    solver,
                                          HYPRE_ParVector Gx,
                                          HYPRE_ParVector Gy,
                                          HYPRE_ParVector Gz);

/**
 * (Optional) Set the (components of) the Nedelec interpolation matrix
 * \f$\Pi = [ \Pi^x, \Pi^y, \Pi^z ]\f$.
 *
 * This function is generally intended to be used only for high-order Nedelec
 * discretizations (in the lowest order case, \f$\Pi\f$ is constructed internally in
 * AMS from the discreet gradient matrix and the coordinates of the vertices),
 * though it can also be used in the lowest-order case or for other types of
 * discretizations (e.g. ones based on the second family of Nedelec elements).
 *
 * By definition, \f$\Pi\f$ is the matrix representation of the linear operator that
 * interpolates (high-order) vector nodal finite elements into the (high-order)
 * Nedelec space. The component matrices are defined as \f$\Pi^x \varphi = \Pi
 * (\varphi,0,0)\f$ and similarly for \f$\Pi^y\f$ and \f$\Pi^z\f$. Note that all these
 * operators depend on the choice of the basis and degrees of freedom in the
 * high-order spaces.
 *
 * The column numbering of Pi should be node-based, i.e. the \f$x\f$/\f$y\f$/\f$z\f$
 * components of the first node (vertex or high-order dof) should be listed
 * first, followed by the \f$x\f$/\f$y\f$/\f$z\f$ components of the second node and so on
 * (see the documentation of HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before HYPRE_AMSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * \f$\{\Pi\}\f$ and \f$\{\Pi^x,\Pi^y,\Pi^z\}\f$ needs to be specified (though it is OK
 * to provide both).  If Pix is NULL, then scalar \f$\Pi\f$-based AMS cycles,
 * i.e. those with \e cycle_type > 10, will be unavailable. Similarly, AMS cycles
 * based on monolithic \f$\Pi\f$ (\e cycle_type < 10) require that Pi is not NULL.
 **/
HYPRE_Int HYPRE_AMSSetInterpolations(HYPRE_Solver       solver,
                                     HYPRE_ParCSRMatrix Pi,
                                     HYPRE_ParCSRMatrix Pix,
                                     HYPRE_ParCSRMatrix Piy,
                                     HYPRE_ParCSRMatrix Piz);

/**
 * (Optional) Sets the matrix \f$A_\alpha\f$ corresponding to the Poisson
 * problem with coefficient \f$\alpha\f$ (the curl-curl term coefficient in
 * the Maxwell problem).
 *
 * If this function is called, the coarse space solver on the range
 * of \f$\Pi^T\f$ is a block-diagonal version of \f$A_\Pi\f$. If this function is not
 * called, the coarse space solver on the range of \f$\Pi^T\f$ is constructed
 * as \f$\Pi^T A \Pi\f$ in HYPRE_AMSSetup(). See the user's manual for more details.
 **/
HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix(HYPRE_Solver       solver,
                                         HYPRE_ParCSRMatrix A_alpha);

/**
 * (Optional) Sets the matrix \f$A_\beta\f$ corresponding to the Poisson
 * problem with coefficient \f$\beta\f$ (the mass term coefficient in the
 * Maxwell problem).
 *
 * If not given, the Poisson matrix will be computed in HYPRE_AMSSetup().
 * If the given matrix is NULL, we assume that \f$\beta\f$ is identically 0
 * and use two-level (instead of three-level) methods. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix(HYPRE_Solver       solver,
                                        HYPRE_ParCSRMatrix A_beta);

/**
 * (Optional) Set the list of nodes which are interior to a zero-conductivity
 * region. This way, a more robust solver is constructed, that can be iterated
 * to lower tolerance levels. A node is interior if its entry in the array is
 * 1.0. This function should be called before HYPRE_AMSSetup()!
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
 * number of iterations to 1. The default is 20.
 **/
HYPRE_Int HYPRE_AMSSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int    maxit);

/**
 * (Optional) Set the convergence tolerance, if AMS is used
 * as a solver. When using AMS as a preconditioner, set the tolerance
 * to 0.0. The default is \f$10^{-6}\f$.
 **/
HYPRE_Int HYPRE_AMSSetTol(HYPRE_Solver solver,
                          HYPRE_Real   tol);

/**
 * (Optional) Choose which three-level solver to use. Possible values are:
 *
 *    - 1  : 3-level multiplicative solver (01210)
 *    - 2  : 3-level additive solver (0+1+2)
 *    - 3  : 3-level multiplicative solver (02120)
 *    - 4  : 3-level additive solver (010+2)
 *    - 5  : 3-level multiplicative solver (0102010)
 *    - 6  : 3-level additive solver (1+020)
 *    - 7  : 3-level multiplicative solver (0201020)
 *    - 8  : 3-level additive solver (0(1+2)0)
 *    - 11 : 5-level multiplicative solver (013454310)
 *    - 12 : 5-level additive solver (0+1+3+4+5)
 *    - 13 : 5-level multiplicative solver (034515430)
 *    - 14 : 5-level additive solver (01(3+4+5)10)
 *
 * The default is 1. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_AMSSetCycleType(HYPRE_Solver solver,
                                HYPRE_Int    cycle_type);

/**
 * (Optional) Control how much information is printed during the
 * solution iterations.
 * The default is 1 (print residual norm at each step).
 **/
HYPRE_Int HYPRE_AMSSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int    print_level);

/**
 * (Optional) Sets relaxation parameters for \f$A\f$.
 * The defaults are 2, 1, 1.0, 1.0.
 *
 * The available options for \e relax_type are:
 *
 *    - 1 : \f$\ell_1\f$-scaled Jacobi
 *    - 2 : \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 3 : Kaczmarz
 *    - 4 : truncated version of \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 16 : Chebyshev
 **/
HYPRE_Int HYPRE_AMSSetSmoothingOptions(HYPRE_Solver solver,
                                       HYPRE_Int    relax_type,
                                       HYPRE_Int    relax_times,
                                       HYPRE_Real   relax_weight,
                                       HYPRE_Real   omega);

/**
 * (Optional) Sets AMG parameters for \f$B_\Pi\f$.
 * The defaults are 10, 1, 3, 0.25, 0, 0. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_AMSSetAlphaAMGOptions(HYPRE_Solver solver,
                                      HYPRE_Int    alpha_coarsen_type,
                                      HYPRE_Int    alpha_agg_levels,
                                      HYPRE_Int    alpha_relax_type,
                                      HYPRE_Real   alpha_strength_threshold,
                                      HYPRE_Int    alpha_interp_type,
                                      HYPRE_Int    alpha_Pmax);

/**
 * (Optional) Sets the coarsest level relaxation in the AMG solver for \f$B_\Pi\f$.
 * The default is 8 (l1-GS). Use 9, 19, 29 or 99 for a direct solver.
 **/
HYPRE_Int HYPRE_AMSSetAlphaAMGCoarseRelaxType(HYPRE_Solver solver,
                                              HYPRE_Int    alpha_coarse_relax_type);

/**
 * (Optional) Sets AMG parameters for \f$B_G\f$.
 * The defaults are 10, 1, 3, 0.25, 0, 0. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_AMSSetBetaAMGOptions(HYPRE_Solver solver,
                                     HYPRE_Int    beta_coarsen_type,
                                     HYPRE_Int    beta_agg_levels,
                                     HYPRE_Int    beta_relax_type,
                                     HYPRE_Real   beta_strength_threshold,
                                     HYPRE_Int    beta_interp_type,
                                     HYPRE_Int    beta_Pmax);

/**
 * (Optional) Sets the coarsest level relaxation in the AMG solver for \f$B_G\f$.
 * The default is 8 (l1-GS). Use 9, 19, 29 or 99 for a direct solver.
 **/
HYPRE_Int HYPRE_AMSSetBetaAMGCoarseRelaxType(HYPRE_Solver solver,
                                             HYPRE_Int    beta_coarse_relax_type);

/**
 * Returns the number of iterations taken.
 **/
HYPRE_Int HYPRE_AMSGetNumIterations(HYPRE_Solver  solver,
                                    HYPRE_Int    *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                HYPRE_Real   *rel_resid_norm);

/**
 * For problems with zero-conductivity regions, project the vector onto the
 * compatible subspace: \f$x = (I - G_0 (G_0^t G_0)^{-1} G_0^T) x\f$, where \f$G_0\f$ is
 * the discrete gradient restricted to the interior nodes of the regions with
 * zero conductivity. This ensures that x is orthogonal to the gradients in the
 * range of \f$G_0\f$.
 *
 * This function is typically called after the solution iteration is complete,
 * in order to facilitate the visualization of the computed field. Without it
 * the values in the zero-conductivity regions contain kernel components.
 **/
HYPRE_Int HYPRE_AMSProjectOutGradients(HYPRE_Solver    solver,
                                       HYPRE_ParVector x);

/**
 * Construct and return the lowest-order discrete gradient matrix G using some
 * edge and vertex information. We assume that \e edge_vertex lists the edge
 * vertices consecutively, and that the orientation of all edges is consistent.
 *
 * If \e edge_orientation = 1, the edges are already oriented.
 *
 * If \e edge_orientation = 2, the orientation of edge i depends only
 * on the sign of \e edge_vertex[2*i+1] - \e edge_vertex[2*i].
 **/
HYPRE_Int HYPRE_AMSConstructDiscreteGradient(HYPRE_ParCSRMatrix  A,
                                             HYPRE_ParVector     x_coord,
                                             HYPRE_BigInt       *edge_vertex,
                                             HYPRE_Int           edge_orientation,
                                             HYPRE_ParCSRMatrix *G);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR ADS Solver and Preconditioner
 *
 * Parallel auxiliary space divergence solver and preconditioner
 *
 * @{
 **/

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
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b Ignored by this function.
 * @param x Ignored by this function.
 **/
HYPRE_Int HYPRE_ADSSetup(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Solve the system or apply ADS as a preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] solver or preconditioner object to be applied.
 * @param A [IN] ParCSR matrix, matrix of the linear system to be solved
 * @param b [IN] right hand side of the linear system to be solved
 * @param x [OUT] approximated solution of the linear system to be solved
 **/
HYPRE_Int HYPRE_ADSSolve(HYPRE_Solver       solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector    b,
                         HYPRE_ParVector    x);

/**
 * Sets the discrete curl matrix \e C.
 * This function should be called before HYPRE_ADSSetup()!
 **/
HYPRE_Int HYPRE_ADSSetDiscreteCurl(HYPRE_Solver       solver,
                                   HYPRE_ParCSRMatrix C);

/**
 * Sets the discrete gradient matrix \e G.
 * This function should be called before HYPRE_ADSSetup()!
 **/
HYPRE_Int HYPRE_ADSSetDiscreteGradient(HYPRE_Solver       solver,
                                       HYPRE_ParCSRMatrix G);

/**
 * Sets the \e x, \e y and \e z coordinates of the vertices in the mesh.
 * This function should be called before HYPRE_ADSSetup()!
 **/
HYPRE_Int HYPRE_ADSSetCoordinateVectors(HYPRE_Solver    solver,
                                        HYPRE_ParVector x,
                                        HYPRE_ParVector y,
                                        HYPRE_ParVector z);

/**
 * (Optional) Set the (components of) the Raviart-Thomas (\f$\Pi_{RT}\f$) and the Nedelec
 * (\f$\Pi_{ND}\f$) interpolation matrices.
 *
 * This function is generally intended to be used only for high-order \f$H(div)\f$
 * discretizations (in the lowest order case, these matrices are constructed
 * internally in ADS from the discreet gradient and curl matrices and the
 * coordinates of the vertices), though it can also be used in the lowest-order
 * case or for other types of discretizations.
 *
 * By definition, \e RT_Pi and \e ND_Pi are the matrix representations of the linear
 * operators \f$\Pi_{RT}\f$ and \f$\Pi_{ND}\f$ that interpolate (high-order) vector
 * nodal finite elements into the (high-order) Raviart-Thomas and Nedelec
 * spaces. The component matrices are defined in both cases as \f$\Pi^x \varphi =
 * \Pi (\varphi,0,0)\f$ and similarly for \f$\Pi^y\f$ and \f$\Pi^z\f$. Note that all these
 * operators depend on the choice of the basis and degrees of freedom in the
 * high-order spaces.
 *
 * The column numbering of \e RT_Pi and \e ND_Pi should be node-based, i.e. the
 * \f$x\f$/\f$y\f$/\f$z\f$ components of the first node (vertex or high-order dof) should be
 * listed first, followed by the \f$x\f$/\f$y\f$/\f$z\f$ components of the second node and
 * so on (see the documentation of HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before hypre_ADSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * \f$\{\Pi_{RT}\}\f$ and \f$\{\Pi_{RT}^x,\Pi_{RT}^y,\Pi_{RT}^z\}\f$ needs to be
 * specified (though it is OK to provide both).  If \e RT_Pix is NULL, then scalar
 * \f$\Pi\f$-based ADS cycles, i.e. those with \e cycle_type > 10, will be
 * unavailable. Similarly, ADS cycles based on monolithic \f$\Pi\f$ (\e cycle_type <
 * 10) require that \e RT_Pi is not NULL. The same restrictions hold for the sets
 * \f$\{\Pi_{ND}\}\f$ and \f$\{\Pi_{ND}^x,\Pi_{ND}^y,\Pi_{ND}^z\}\f$ -- only one of them
 * needs to be specified, and the availability of each enables different AMS
 * cycle type options.
 **/
HYPRE_Int HYPRE_ADSSetInterpolations(HYPRE_Solver       solver,
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
 * number of iterations to 1. The default is 20.
 **/
HYPRE_Int HYPRE_ADSSetMaxIter(HYPRE_Solver solver,
                              HYPRE_Int    maxit);

/**
 * (Optional) Set the convergence tolerance, if ADS is used
 * as a solver. When using ADS as a preconditioner, set the tolerance
 * to 0.0. The default is \f$10^{-6}\f$.
 **/
HYPRE_Int HYPRE_ADSSetTol(HYPRE_Solver solver,
                          HYPRE_Real   tol);

/**
 * (Optional) Choose which auxiliary-space solver to use. Possible values are:
 *
 *    -  1 : 3-level multiplicative solver (01210)
 *    -  2 : 3-level additive solver (0+1+2)
 *    -  3 : 3-level multiplicative solver (02120)
 *    -  4 : 3-level additive solver (010+2)
 *    -  5 : 3-level multiplicative solver (0102010)
 *    -  6 : 3-level additive solver (1+020)
 *    -  7 : 3-level multiplicative solver (0201020)
 *    -  8 : 3-level additive solver (0(1+2)0)
 *    - 11 : 5-level multiplicative solver (013454310)
 *    - 12 : 5-level additive solver (0+1+3+4+5)
 *    - 13 : 5-level multiplicative solver (034515430)
 *    - 14 : 5-level additive solver (01(3+4+5)10)
 *
 * The default is 1. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_ADSSetCycleType(HYPRE_Solver solver,
                                HYPRE_Int    cycle_type);

/**
 * (Optional) Control how much information is printed during the
 * solution iterations.
 * The default is 1 (print residual norm at each step).
 **/
HYPRE_Int HYPRE_ADSSetPrintLevel(HYPRE_Solver solver,
                                 HYPRE_Int    print_level);

/**
 * (Optional) Sets relaxation parameters for \f$A\f$.
 * The defaults are 2, 1, 1.0, 1.0.
 *
 * The available options for \e relax_type are:
 *
 *    - 1  : \f$\ell_1\f$-scaled Jacobi
 *    - 2  : \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 3  : Kaczmarz
 *    - 4  : truncated version of \f$\ell_1\f$-scaled block symmetric Gauss-Seidel/SSOR
 *    - 16 : Chebyshev
 **/
HYPRE_Int HYPRE_ADSSetSmoothingOptions(HYPRE_Solver solver,
                                       HYPRE_Int    relax_type,
                                       HYPRE_Int    relax_times,
                                       HYPRE_Real   relax_weight,
                                       HYPRE_Real   omega);

/**
 * (Optional) Sets parameters for Chebyshev relaxation.
 * The defaults are 2, 0.3.
 **/
HYPRE_Int HYPRE_ADSSetChebySmoothingOptions(HYPRE_Solver solver,
                                            HYPRE_Int    cheby_order,
                                            HYPRE_Real   cheby_fraction);

/**
 * (Optional) Sets AMS parameters for \f$B_C\f$.
 * The defaults are 11, 10, 1, 3, 0.25, 0, 0.
 * Note that \e cycle_type should be greater than 10, unless the high-order
 * interface of HYPRE_ADSSetInterpolations is being used!
 * See the user's manual for more details.
 **/
HYPRE_Int HYPRE_ADSSetAMSOptions(HYPRE_Solver solver,
                                 HYPRE_Int    cycle_type,
                                 HYPRE_Int    coarsen_type,
                                 HYPRE_Int    agg_levels,
                                 HYPRE_Int    relax_type,
                                 HYPRE_Real   strength_threshold,
                                 HYPRE_Int    interp_type,
                                 HYPRE_Int    Pmax);

/**
 * (Optional) Sets AMG parameters for \f$B_\Pi\f$.
 * The defaults are 10, 1, 3, 0.25, 0, 0. See the user's manual for more details.
 **/
HYPRE_Int HYPRE_ADSSetAMGOptions(HYPRE_Solver solver,
                                 HYPRE_Int    coarsen_type,
                                 HYPRE_Int    agg_levels,
                                 HYPRE_Int    relax_type,
                                 HYPRE_Real   strength_threshold,
                                 HYPRE_Int    interp_type,
                                 HYPRE_Int    Pmax);

/**
 * Returns the number of iterations taken.
 **/
HYPRE_Int HYPRE_ADSGetNumIterations(HYPRE_Solver  solver,
                                    HYPRE_Int    *num_iterations);

/**
 * Returns the norm of the final relative residual.
 **/
HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                HYPRE_Real   *rel_resid_norm);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR PCG Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

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
                                HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRPCGSetAbsoluteTol(HYPRE_Solver solver,
                                        HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRPCGSetMaxIter(HYPRE_Solver solver,
                                    HYPRE_Int    max_iter);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRPCGSetStopCrit(HYPRE_Solver solver,
                                     HYPRE_Int    stop_crit);

HYPRE_Int HYPRE_ParCSRPCGSetTwoNorm(HYPRE_Solver solver,
                                    HYPRE_Int    two_norm);

HYPRE_Int HYPRE_ParCSRPCGSetRelChange(HYPRE_Solver solver,
                                      HYPRE_Int    rel_change);

HYPRE_Int HYPRE_ParCSRPCGSetPrecond(HYPRE_Solver            solver,
                                    HYPRE_PtrToParSolverFcn precond,
                                    HYPRE_PtrToParSolverFcn precond_setup,
                                    HYPRE_Solver            precond_solver);

HYPRE_Int HYPRE_ParCSRPCGSetPreconditioner(HYPRE_Solver solver,
                                           HYPRE_Solver precond);

HYPRE_Int HYPRE_ParCSRPCGGetPrecond(HYPRE_Solver  solver,
                                    HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRPCGSetLogging(HYPRE_Solver solver,
                                    HYPRE_Int    logging);

HYPRE_Int HYPRE_ParCSRPCGSetPrintLevel(HYPRE_Solver solver,
                                       HYPRE_Int    print_level);

HYPRE_Int HYPRE_ParCSRPCGGetNumIterations(HYPRE_Solver  solver,
                                          HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                      HYPRE_Real   *norm);
/**
 * Returns the residual.
 **/
HYPRE_Int HYPRE_ParCSRPCGGetResidual(HYPRE_Solver     solver,
                                     HYPRE_ParVector *residual);

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

/**
 * Setup routine for on-processor triangular solve as preconditioning.
 **/
HYPRE_Int HYPRE_ParCSROnProcTriSetup(HYPRE_Solver       solver,
                                     HYPRE_ParCSRMatrix HA,
                                     HYPRE_ParVector    Hy,
                                     HYPRE_ParVector    Hx);

/**
 * Solve routine for on-processor triangular solve as preconditioning.
 **/
HYPRE_Int HYPRE_ParCSROnProcTriSolve(HYPRE_Solver       solver,
                                     HYPRE_ParCSRMatrix HA,
                                     HYPRE_ParVector    Hy,
                                     HYPRE_ParVector    Hx);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

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
                                   HYPRE_Int    k_dim);

HYPRE_Int HYPRE_ParCSRGMRESSetTol(HYPRE_Solver solver,
                                  HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                          HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRGMRESSetMinIter(HYPRE_Solver solver,
                                      HYPRE_Int    min_iter);

HYPRE_Int HYPRE_ParCSRGMRESSetMaxIter(HYPRE_Solver solver,
                                      HYPRE_Int    max_iter);

/*
 * Obsolete
 **/
HYPRE_Int HYPRE_ParCSRGMRESSetStopCrit(HYPRE_Solver solver,
                                       HYPRE_Int    stop_crit);

HYPRE_Int HYPRE_ParCSRGMRESSetPrecond(HYPRE_Solver             solver,
                                      HYPRE_PtrToParSolverFcn  precond,
                                      HYPRE_PtrToParSolverFcn  precond_setup,
                                      HYPRE_Solver             precond_solver);

HYPRE_Int HYPRE_ParCSRGMRESGetPrecond(HYPRE_Solver  solver,
                                      HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRGMRESSetLogging(HYPRE_Solver solver,
                                      HYPRE_Int    logging);

HYPRE_Int HYPRE_ParCSRGMRESSetPrintLevel(HYPRE_Solver solver,
                                         HYPRE_Int    print_level);

HYPRE_Int HYPRE_ParCSRGMRESGetNumIterations(HYPRE_Solver  solver,
                                            HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                        HYPRE_Real   *norm);
/**
 * Returns the residual.
 **/
HYPRE_Int HYPRE_ParCSRGMRESGetResidual(HYPRE_Solver     solver,
                                       HYPRE_ParVector *residual);


/* ParCSR CO-GMRES, author: KS */

/**
 * Create a solver object.
 **/
HYPRE_Int HYPRE_ParCSRCOGMRESCreate(MPI_Comm      comm,
                                    HYPRE_Solver *solver);

/**
 * Destroy a solver object.
 **/
HYPRE_Int HYPRE_ParCSRCOGMRESDestroy(HYPRE_Solver solver);

HYPRE_Int HYPRE_ParCSRCOGMRESSetup(HYPRE_Solver       solver,
                                   HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector    b,
                                   HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRCOGMRESSolve(HYPRE_Solver       solver,
                                   HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector    b,
                                   HYPRE_ParVector    x);

HYPRE_Int HYPRE_ParCSRCOGMRESSetKDim(HYPRE_Solver solver,
                                     HYPRE_Int    k_dim);

HYPRE_Int HYPRE_ParCSRCOGMRESSetUnroll(HYPRE_Solver solver,
                                       HYPRE_Int    unroll);

HYPRE_Int HYPRE_ParCSRCOGMRESSetCGS(HYPRE_Solver solver,
                                    HYPRE_Int    cgs);

HYPRE_Int HYPRE_ParCSRCOGMRESSetTol(HYPRE_Solver solver,
                                    HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRCOGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                            HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRCOGMRESSetMinIter(HYPRE_Solver solver,
                                        HYPRE_Int    min_iter);

HYPRE_Int HYPRE_ParCSRCOGMRESSetMaxIter(HYPRE_Solver solver,
                                        HYPRE_Int    max_iter);

HYPRE_Int HYPRE_ParCSRCOGMRESSetPrecond(HYPRE_Solver             solver,
                                        HYPRE_PtrToParSolverFcn  precond,
                                        HYPRE_PtrToParSolverFcn  precond_setup,
                                        HYPRE_Solver             precond_solver);

HYPRE_Int HYPRE_ParCSRCOGMRESGetPrecond(HYPRE_Solver  solver,
                                        HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRCOGMRESSetLogging(HYPRE_Solver solver,
                                        HYPRE_Int    logging);

HYPRE_Int HYPRE_ParCSRCOGMRESSetPrintLevel(HYPRE_Solver solver,
                                           HYPRE_Int    print_level);

HYPRE_Int HYPRE_ParCSRCOGMRESGetNumIterations(HYPRE_Solver  solver,
                                              HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                          HYPRE_Real   *norm);

/**
 * Returns the residual.
 **/
HYPRE_Int HYPRE_ParCSRCOGMRESGetResidual(HYPRE_Solver     solver,
                                         HYPRE_ParVector *residual);

/* end of parCSR CO-GMRES */

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

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
                                       HYPRE_Int    k_dim);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetTol(HYPRE_Solver solver,
                                      HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                              HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMinIter(HYPRE_Solver solver,
                                          HYPRE_Int    min_iter);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetMaxIter(HYPRE_Solver solver,
                                          HYPRE_Int    max_iter);


HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrecond(HYPRE_Solver             solver,
                                          HYPRE_PtrToParSolverFcn  precond,
                                          HYPRE_PtrToParSolverFcn  precond_setup,
                                          HYPRE_Solver             precond_solver);

HYPRE_Int HYPRE_ParCSRFlexGMRESGetPrecond(HYPRE_Solver  solver,
                                          HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetLogging(HYPRE_Solver solver,
                                          HYPRE_Int    logging);

HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrintLevel(HYPRE_Solver solver,
                                             HYPRE_Int    print_level);

HYPRE_Int HYPRE_ParCSRFlexGMRESGetNumIterations(HYPRE_Solver  solver,
                                                HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                            HYPRE_Real   *norm);

HYPRE_Int HYPRE_ParCSRFlexGMRESGetResidual(HYPRE_Solver     solver,
                                           HYPRE_ParVector *residual);


HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC( HYPRE_Solver           solver,
                                            HYPRE_PtrToModifyPCFcn modify_pc);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

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
                                    HYPRE_Int    k_dim);

HYPRE_Int HYPRE_ParCSRLGMRESSetAugDim(HYPRE_Solver solver,
                                      HYPRE_Int    aug_dim);

HYPRE_Int HYPRE_ParCSRLGMRESSetTol(HYPRE_Solver solver,
                                   HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRLGMRESSetAbsoluteTol(HYPRE_Solver solver,
                                           HYPRE_Real   a_tol);

/*
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRLGMRESSetMinIter(HYPRE_Solver solver,
                                       HYPRE_Int    min_iter);

HYPRE_Int HYPRE_ParCSRLGMRESSetMaxIter(HYPRE_Solver solver,
                                       HYPRE_Int    max_iter);

HYPRE_Int HYPRE_ParCSRLGMRESSetPrecond(HYPRE_Solver             solver,
                                       HYPRE_PtrToParSolverFcn  precond,
                                       HYPRE_PtrToParSolverFcn  precond_setup,
                                       HYPRE_Solver             precond_solver);

HYPRE_Int HYPRE_ParCSRLGMRESGetPrecond(HYPRE_Solver  solver,
                                       HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRLGMRESSetLogging(HYPRE_Solver solver,
                                       HYPRE_Int    logging);

HYPRE_Int HYPRE_ParCSRLGMRESSetPrintLevel(HYPRE_Solver solver,
                                          HYPRE_Int    print_level);

HYPRE_Int HYPRE_ParCSRLGMRESGetNumIterations(HYPRE_Solver  solver,
                                             HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                         HYPRE_Real   *norm);

HYPRE_Int HYPRE_ParCSRLGMRESGetResidual(HYPRE_Solver     solver,
                                        HYPRE_ParVector *residual);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref KrylovSolvers.
 *
 * @{
 **/

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
                                     HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetAbsoluteTol(HYPRE_Solver solver,
                                             HYPRE_Real   a_tol);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetMinIter(HYPRE_Solver solver,
                                         HYPRE_Int    min_iter);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetMaxIter(HYPRE_Solver solver,
                                         HYPRE_Int    max_iter);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetStopCrit(HYPRE_Solver solver,
                                          HYPRE_Int    stop_crit);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrecond(HYPRE_Solver            solver,
                                         HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup,
                                         HYPRE_Solver            precond_solver);

HYPRE_Int HYPRE_ParCSRBiCGSTABGetPrecond(HYPRE_Solver  solver,
                                         HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetLogging(HYPRE_Solver solver,
                                         HYPRE_Int    logging);

HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrintLevel(HYPRE_Solver solver,
                                            HYPRE_Int    print_level);

HYPRE_Int HYPRE_ParCSRBiCGSTABGetNumIterations(HYPRE_Solver  solver,
                                               HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                           HYPRE_Real   *norm);

HYPRE_Int HYPRE_ParCSRBiCGSTABGetResidual(HYPRE_Solver     solver,
                                          HYPRE_ParVector *residual);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR Hybrid Solver
 *
 * @{
 **/

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
 *  Set the convergence tolerance for the Krylov solver. The default is 1.e-6.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetTol(HYPRE_Solver solver,
                                   HYPRE_Real   tol);
/**
 *  Set the absolute convergence tolerance for the Krylov solver. The default is 0.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetAbsoluteTol(HYPRE_Solver solver,
                                           HYPRE_Real   tol);

/**
 *  Set the desired convergence factor
 **/
HYPRE_Int HYPRE_ParCSRHybridSetConvergenceTol(HYPRE_Solver solver,
                                              HYPRE_Real   cf_tol);

/**
 *  Set the maximal number of iterations for the diagonally
 *  preconditioned solver
 **/
HYPRE_Int HYPRE_ParCSRHybridSetDSCGMaxIter(HYPRE_Solver solver,
                                           HYPRE_Int    dscg_max_its);

/**
 *  Set the maximal number of iterations for the AMG
 *  preconditioned solver
 **/
HYPRE_Int HYPRE_ParCSRHybridSetPCGMaxIter(HYPRE_Solver solver,
                                          HYPRE_Int    pcg_max_its);

/*
 *
 **/
HYPRE_Int HYPRE_ParCSRHybridSetSetupType(HYPRE_Solver solver,
                                         HYPRE_Int    setup_type);

/**
 *  Set the desired solver type. There are the following options:
 *     -  1 : PCG (default)
 *     -  2 : GMRES
 *     -  3 : BiCGSTAB
 **/
HYPRE_Int HYPRE_ParCSRHybridSetSolverType(HYPRE_Solver solver,
                                          HYPRE_Int    solver_type);

/**
 * (Optional) Set recompute residual (don't rely on 3-term recurrence).
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetRecomputeResidual( HYPRE_Solver  solver,
                                        HYPRE_Int     recompute_residual );

/**
 * (Optional) Get recompute residual option.
 **/
HYPRE_Int
HYPRE_ParCSRHybridGetRecomputeResidual( HYPRE_Solver  solver,
                                        HYPRE_Int    *recompute_residual );

/**
 * (Optional) Set recompute residual period (don't rely on 3-term recurrence).
 *
 * Recomputes residual after every specified number of iterations.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetRecomputeResidualP( HYPRE_Solver  solver,
                                         HYPRE_Int     recompute_residual_p );

/**
 * (Optional) Get recompute residual period option.
 **/
HYPRE_Int
HYPRE_ParCSRHybridGetRecomputeResidualP( HYPRE_Solver  solver,
                                         HYPRE_Int    *recompute_residual_p );

/**
 * Set the Krylov dimension for restarted GMRES.
 * The default is 5.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetKDim(HYPRE_Solver solver,
                                    HYPRE_Int    k_dim);

/**
 * Set the type of norm for PCG.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetTwoNorm(HYPRE_Solver solver,
                                       HYPRE_Int    two_norm);

/**
 * RE-VISIT
 **/
HYPRE_Int HYPRE_ParCSRHybridSetStopCrit(HYPRE_Solver solver,
                                        HYPRE_Int    stop_crit);

/**
 *
 **/
HYPRE_Int HYPRE_ParCSRHybridSetRelChange(HYPRE_Solver solver,
                                         HYPRE_Int    rel_change);

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
                                       HYPRE_Int    logging);

/**
 * Set print level (default: 0, no printing)
 * 2 will print residual norms per iteration
 * 10 will print AMG setup information if AMG is used
 * 12 both Setup information and iterations.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetPrintLevel(HYPRE_Solver solver,
                                          HYPRE_Int    print_level);

/**
 * (Optional) Sets AMG strength threshold. The default is 0.25.
 * For elasticity problems, a larger strength threshold, such as 0.7 or 0.8,
 * is often better.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetStrongThreshold(HYPRE_Solver solver,
                                     HYPRE_Real   strong_threshold);

/**
 * (Optional) Sets a parameter to modify the definition of strength for
 * diagonal dominant portions of the matrix. The default is 0.9.
 * If \e max_row_sum is 1, no checking for diagonally dominant rows is
 * performed.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMaxRowSum(HYPRE_Solver solver,
                               HYPRE_Real   max_row_sum);

/**
 * (Optional) Defines a truncation factor for the interpolation.
 * The default is 0.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetTruncFactor(HYPRE_Solver solver,
                                 HYPRE_Real   trunc_factor);


/**
 * (Optional) Defines the maximal number of elements per row for the interpolation.
 * The default is 0.
 **/
HYPRE_Int HYPRE_ParCSRHybridSetPMaxElmts(HYPRE_Solver solver,
                                         HYPRE_Int    P_max_elmts);

/**
 * (Optional) Defines the maximal number of levels used for AMG.
 * The default is 25.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMaxLevels(HYPRE_Solver solver,
                               HYPRE_Int    max_levels);

/**
 * (Optional) Defines whether local or global measures are used.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMeasureType(HYPRE_Solver solver,
                                 HYPRE_Int    measure_type);

/**
 * (Optional) Defines which parallel coarsening algorithm is used.
 * There are the following options for \e coarsen_type:
 *
 *    - 0  : CLJP-coarsening (a parallel coarsening algorithm using independent sets).
 *    - 1  : classical Ruge-Stueben coarsening on each processor, no boundary treatment
 *    - 3  : classical Ruge-Stueben coarsening on each processor, followed by a third
 *           pass, which adds coarse points on the boundaries
 *    - 6  : Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse
 *           points generated by 1 as its first independent set)
 *    - 7  : CLJP-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 8  : PMIS-coarsening (a parallel coarsening algorithm using independent sets
 *           with lower complexities than CLJP, might also lead to slower convergence)
 *    - 9  : PMIS-coarsening (using a fixed random vector, for debugging purposes only)
 *    - 10 : HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently,
 *           followed by PMIS using the interior C-points as its first independent set)
 *    - 11 : one-pass Ruge-Stueben coarsening on each processor, no boundary treatment
 *
 * The default is 10.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetCoarsenType(HYPRE_Solver solver,
                                 HYPRE_Int    coarsen_type);

/**
 * (Optional) Specifies which interpolation operator is used
 * The default is ext+i interpolation truncated to at most 4 elements per row.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetInterpType(HYPRE_Solver solver,
                                HYPRE_Int    interp_type);

/**
 * (Optional) Defines the type of cycle.
 * For a V-cycle, set \e cycle_type to 1, for a W-cycle
 *  set \e cycle_type to 2. The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetCycleType(HYPRE_Solver solver,
                               HYPRE_Int    cycle_type);

/*
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetGridRelaxType(HYPRE_Solver  solver,
                                   HYPRE_Int    *grid_relax_type);

/*
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetGridRelaxPoints(HYPRE_Solver   solver,
                                     HYPRE_Int    **grid_relax_points);

/**
 * (Optional) Sets the number of sweeps. On the finest level, the up and
 * the down cycle the number of sweeps are set to \e num_sweeps and on the
 * coarsest level to 1. The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetNumSweeps(HYPRE_Solver solver,
                               HYPRE_Int    num_sweeps);

/**
 * (Optional) Sets the number of sweeps at a specified cycle.
 * There are the following options for \e k:
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetCycleNumSweeps(HYPRE_Solver solver,
                                    HYPRE_Int    num_sweeps,
                                    HYPRE_Int    k);

/**
 * (Optional) Defines the smoother to be used. It uses the given
 * smoother on the fine grid, the up and
 * the down cycle and sets the solver on the coarsest level to Gaussian
 * elimination (9). The default is l1-Gauss-Seidel, forward solve on the down
 * cycle (13) and backward solve on the up cycle (14).
 *
 * There are the following options for \e relax_type:
 *
 *    - 0  : Jacobi
 *    - 1  : Gauss-Seidel, sequential (very slow!)
 *    - 2  : Gauss-Seidel, interior points in parallel, boundary sequential (slow!)
 *    - 3  : hybrid Gauss-Seidel or SOR, forward solve
 *    - 4  : hybrid Gauss-Seidel or SOR, backward solve
 *    - 6  : hybrid symmetric Gauss-Seidel or SSOR
 *    - 8  : hybrid symmetric l1-Gauss-Seidel or SSOR
 *    - 13 : l1-Gauss-Seidel, forward solve
 *    - 14 : l1-Gauss-Seidel, backward solve
 *    - 18 : l1-Jacobi
 *    - 9  : Gaussian elimination (only on coarsest level)
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxType(HYPRE_Solver solver,
                               HYPRE_Int    relax_type);

/**
 * (Optional) Defines the smoother at a given cycle.
 * For options of \e relax_type see
 * description of HYPRE_BoomerAMGSetRelaxType). Options for k are
 *
 *    - 1 : the down cycle
 *    - 2 : the up cycle
 *    - 3 : the coarsest level
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetCycleRelaxType(HYPRE_Solver solver,
                                    HYPRE_Int    relax_type,
                                    HYPRE_Int    k);

/**
 * (Optional) Defines in which order the points are relaxed. There are
 * the following options for \e relax_order:
 *
 *    - 0 : the points are relaxed in natural or lexicographic order on each processor
 *    - 1 : CF-relaxation is used, i.e on the fine grid and the down cycle the
 *          coarse points are relaxed first, followed by the fine points; on the
 *          up cycle the F-points are relaxed first, followed by the C-points.
 *          On the coarsest level, if an iterative scheme is used, the points
 *          are relaxed in lexicographic order.
 *
 * The default is 0 (CF-relaxation).
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxOrder(HYPRE_Solver solver,
                                HYPRE_Int    relax_order);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on all levels.
 *
 * Values for \e relax_wt are
 *    - > 0  : this assigns the given relaxation weight on all levels
 *    - = 0  : the weight is determined on each level with the estimate
 *             \f$3 \over {4\|D^{-1/2}AD^{-1/2}\|}\f$, where \f$D\f$ is the diagonal of \f$A\f$
 *             (this should only be used with Jacobi)
 *    - = -k : the relaxation weight is determined with at most k CG steps on each level
 *             (this should only be used for symmetric positive definite problems)
 *
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxWt(HYPRE_Solver solver,
                             HYPRE_Real   relax_wt);

/**
 * (Optional) Defines the relaxation weight for smoothed Jacobi and hybrid SOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive \e relax_weight, the parameter is
 * determined on the given level as described for HYPRE_BoomerAMGSetRelaxWt.
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetLevelRelaxWt(HYPRE_Solver solver,
                                  HYPRE_Real   relax_wt,
                                  HYPRE_Int    level);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR and SSOR
 * on all levels.
 *
 * Values for \e outer_wt are
 *    - > 0  : this assigns the same outer relaxation weight omega on each level
 *    - = -k : an outer relaxation weight is determined with at most k CG steps on each level
 *             (this only makes sense for symmetric positive definite problems and smoothers
 *             such as SSOR)
 *
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetOuterWt(HYPRE_Solver solver,
                             HYPRE_Real   outer_wt);

/**
 * (Optional) Defines the outer relaxation weight for hybrid SOR or SSOR
 * on the user defined level. Note that the finest level is denoted 0, the
 * next coarser level 1, etc. For nonpositive omega, the parameter is
 * determined on the given level as described for HYPRE_BoomerAMGSetOuterWt.
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetLevelOuterWt(HYPRE_Solver solver,
                                  HYPRE_Real   outer_wt,
                                  HYPRE_Int    level);

/**
 * (Optional) Defines the maximal coarse grid size.
 * The default is 9.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMaxCoarseSize(HYPRE_Solver solver,
                                   HYPRE_Int    max_coarse_size);

/**
 * (Optional) Defines the minimal coarse grid size.
 * The default is 0.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetMinCoarseSize(HYPRE_Solver solver,
                                   HYPRE_Int    min_coarse_size);

/**
 * (Optional) enables redundant coarse grid size. If the system size becomes
 * smaller than seq_threshold, sequential AMG is used on all remaining processors.
 * The default is 0.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetSeqThreshold(HYPRE_Solver solver,
                                  HYPRE_Int    seq_threshold);

/**
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetRelaxWeight(HYPRE_Solver  solver,
                                 HYPRE_Real   *relax_weight);

/**
 *
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetOmega(HYPRE_Solver  solver,
                           HYPRE_Real   *omega);

/**
 * (Optional) Defines the number of levels of aggressive coarsening,
 * starting with the finest level.
 * The default is 0, i.e. no aggressive coarsening.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetAggNumLevels(HYPRE_Solver solver,
                                  HYPRE_Int    agg_num_levels);

/**
 * (Optional) Defines the interpolation used on levels of aggressive coarsening
 * The default is 4, i.e. multipass interpolation.
 * The following options exist:
 *
 *    - 1 : 2-stage extended+i interpolation
 *    - 2 : 2-stage standard interpolation
 *    - 3 : 2-stage extended interpolation
 *    - 4 : multipass interpolation
 *    - 5 : 2-stage extended interpolation in matrix-matrix form
 *    - 6 : 2-stage extended+i interpolation in matrix-matrix form
 *    - 7 : 2-stage extended+e interpolation in matrix-matrix form
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetAggInterpType( HYPRE_Solver solver,
                                    HYPRE_Int    agg_interp_type);

/**
 * (Optional) Defines the degree of aggressive coarsening.
 * The default is 1, which leads to the most aggressive coarsening.
 * Setting \e num_paths to 2 will increase complexity somewhat,
 * but can lead to better convergence.**/
HYPRE_Int
HYPRE_ParCSRHybridSetNumPaths(HYPRE_Solver solver,
                              HYPRE_Int    num_paths);

/**
 * (Optional) Sets the size of the system of PDEs, if using the systems version.
 * The default is 1.
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetNumFunctions(HYPRE_Solver solver,
                                  HYPRE_Int    num_functions);

/**
 * (Optional) Sets the mapping that assigns the function to each variable,
 * if using the systems version. If no assignment is made and the number of
 * functions is k > 1, the mapping generated is (0,1,...,k-1,0,1,...,k-1,...).
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetDofFunc(HYPRE_Solver  solver,
                             HYPRE_Int    *dof_func);
/**
 * (Optional) Sets whether to use the nodal systems version.
 * The default is 0 (the unknown based approach).
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetNodal(HYPRE_Solver solver,
                           HYPRE_Int    nodal);

/**
 * (Optional) Sets whether to store local transposed interpolation
 * The default is 0 (don't store).
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetKeepTranspose(HYPRE_Solver solver,
                                   HYPRE_Int    keepT);

/**
 * (Optional) Sets whether to use non-Galerkin option
 * The default is no non-Galerkin option
 * num_levels sets the number of levels where to use it
 * nongalerkin_tol contains the tolerances for <num_levels> levels
 **/
HYPRE_Int
HYPRE_ParCSRHybridSetNonGalerkinTol(HYPRE_Solver solver,
                                    HYPRE_Int   num_levels,
                                    HYPRE_Real *nongalerkin_tol);

/**
 * Retrieves the total number of iterations.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetNumIterations(HYPRE_Solver  solver,
                                             HYPRE_Int    *num_its);

/**
 * Retrieves the number of iterations used by the diagonally scaled solver.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetDSCGNumIterations(HYPRE_Solver  solver,
                                                 HYPRE_Int    *dscg_num_its);

/**
 * Retrieves the number of iterations used by the AMG preconditioned solver.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetPCGNumIterations(HYPRE_Solver  solver,
                                                HYPRE_Int    *pcg_num_its);

/**
 * Retrieves the final relative residual norm.
 **/
HYPRE_Int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                         HYPRE_Real   *norm);

/* Is this a retired function? (RDF) */
HYPRE_Int
HYPRE_ParCSRHybridSetNumGridSweeps(HYPRE_Solver  solver,
                                   HYPRE_Int    *num_grid_sweeps);


HYPRE_Int
HYPRE_ParCSRHybridGetSetupSolveTime( HYPRE_Solver solver,
                                     HYPRE_Real  *time    );
/**@}*/

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
                                  HYPRE_Int    variant);

HYPRE_Int HYPRE_SchwarzSetOverlap(HYPRE_Solver solver,
                                  HYPRE_Int    overlap);

HYPRE_Int HYPRE_SchwarzSetDomainType(HYPRE_Solver solver,
                                     HYPRE_Int    domain_type);

HYPRE_Int HYPRE_SchwarzSetRelaxWeight(HYPRE_Solver solver,
                                      HYPRE_Real   relax_weight);

HYPRE_Int HYPRE_SchwarzSetDomainStructure(HYPRE_Solver    solver,
                                          HYPRE_CSRMatrix domain_structure);

HYPRE_Int HYPRE_SchwarzSetNumFunctions(HYPRE_Solver solver,
                                       HYPRE_Int    num_functions);

HYPRE_Int HYPRE_SchwarzSetDofFunc(HYPRE_Solver  solver,
                                  HYPRE_Int    *dof_func);

HYPRE_Int HYPRE_SchwarzSetNonSymm(HYPRE_Solver solver,
                                  HYPRE_Int    use_nonsymm);

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
                                 HYPRE_Real   tol);

HYPRE_Int HYPRE_ParCSRCGNRSetMinIter(HYPRE_Solver solver,
                                     HYPRE_Int    min_iter);

HYPRE_Int HYPRE_ParCSRCGNRSetMaxIter(HYPRE_Solver solver,
                                     HYPRE_Int    max_iter);

HYPRE_Int HYPRE_ParCSRCGNRSetStopCrit(HYPRE_Solver solver,
                                      HYPRE_Int    stop_crit);

HYPRE_Int HYPRE_ParCSRCGNRSetPrecond(HYPRE_Solver            solver,
                                     HYPRE_PtrToParSolverFcn precond,
                                     HYPRE_PtrToParSolverFcn precondT,
                                     HYPRE_PtrToParSolverFcn precond_setup,
                                     HYPRE_Solver            precond_solver);

HYPRE_Int HYPRE_ParCSRCGNRGetPrecond(HYPRE_Solver  solver,
                                     HYPRE_Solver *precond_data);

HYPRE_Int HYPRE_ParCSRCGNRSetLogging(HYPRE_Solver solver,
                                     HYPRE_Int    logging);

HYPRE_Int HYPRE_ParCSRCGNRGetNumIterations(HYPRE_Solver  solver,
                                           HYPRE_Int    *num_iterations);

HYPRE_Int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm(HYPRE_Solver  solver,
                                                       HYPRE_Real   *norm);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR MGR Solver
 *
 * Parallel multigrid reduction solver and preconditioner.
 * This solver or preconditioner is designed with systems of
 * PDEs in mind. However, it can also be used for scalar linear
 * systems, particularly for problems where the user can exploit
 * information from the physics of the problem. In this way, the
 * MGR solver could potentially be used as a foundation
 * for a physics-based preconditioner.
 *
 * @{
 **/

#ifdef HYPRE_USING_DSUPERLU
/**
 * Create a MGR direct solver object
 **/
HYPRE_Int HYPRE_MGRDirectSolverCreate( HYPRE_Solver *solver );

/**
 * Destroy a MGR direct solver object
 **/
HYPRE_Int HYPRE_MGRDirectSolverDestroy( HYPRE_Solver solver );

/**
 * Setup the MGR direct solver using SuperLU_dist
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b right-hand-side of the linear system to be solved (Ignored by this function).
 * @param x approximate solution of the linear system to be solved (Ignored by this function).
 **/
HYPRE_Int HYPRE_MGRDirectSolverSetup( HYPRE_Solver solver,
                                      HYPRE_ParCSRMatrix A,
                                      HYPRE_ParVector b,
                                      HYPRE_ParVector x );

/**
* Solve the system using SuperLU_dist.
*
* @param solver [IN] solver or preconditioner object to be applied.
* @param A [IN] ParCSR matrix, matrix of the linear system to be solved (Ignored by this function).
* @param b [IN] right hand side of the linear system to be solved
* @param x [OUT] approximated solution of the linear system to be solved
**/
HYPRE_Int HYPRE_MGRDirectSolverSolve( HYPRE_Solver solver,
                                      HYPRE_ParCSRMatrix A,
                                      HYPRE_ParVector b,
                                      HYPRE_ParVector x );
#endif

/**
 * Create a solver object
 **/
HYPRE_Int HYPRE_MGRCreate( HYPRE_Solver *solver );

/**
 * Destroy a solver object
 **/
HYPRE_Int HYPRE_MGRDestroy( HYPRE_Solver solver );

/**
 * Setup the MGR solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b right-hand-side of the linear system to be solved (Ignored by this function).
 * @param x approximate solution of the linear system to be solved (Ignored by this function).
 **/
HYPRE_Int HYPRE_MGRSetup( HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x );

/**
* Solve the system or apply MGR as a preconditioner.
* If used as a preconditioner, this function should be passed
* to the iterative solver \e SetPrecond function.
*
* @param solver [IN] solver or preconditioner object to be applied.
* @param A [IN] ParCSR matrix, matrix of the linear system to be solved
* @param b [IN] right hand side of the linear system to be solved
* @param x [OUT] approximated solution of the linear system to be solved
**/
HYPRE_Int HYPRE_MGRSolve( HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x );

/**
 * Set the block data assuming that the physical variables are ordered contiguously,
 * i.e. p_1, p_2, ..., p_n, s_1, s_2, ..., s_n, ...
 *
 * @param solver [IN] solver or preconditioner object
 * @param block_size [IN] system block size
 * @param max_num_levels [IN] maximum number of reduction levels
 * @param num_block_coarse_points [IN] number of coarse points per block per level
 * @param block_coarse_indexes [IN] index for each block coarse point per level
 **/
HYPRE_Int HYPRE_MGRSetCpointsByContiguousBlock( HYPRE_Solver solver,
                                                HYPRE_Int  block_size,
                                                HYPRE_Int max_num_levels,
                                                HYPRE_BigInt *idx_array,
                                                HYPRE_Int *num_block_coarse_points,
                                                HYPRE_Int  **block_coarse_indexes);

/**
 * Set the block data (by grid points) and prescribe the coarse indexes per block
 * for each reduction level.
 *
 * @param solver [IN] solver or preconditioner object
 * @param block_size [IN] system block size
 * @param max_num_levels [IN] maximum number of reduction levels
 * @param num_block_coarse_points [IN] number of coarse points per block per level
 * @param block_coarse_indexes [IN] index for each block coarse point per level
 **/
HYPRE_Int HYPRE_MGRSetCpointsByBlock( HYPRE_Solver solver,
                                      HYPRE_Int  block_size,
                                      HYPRE_Int max_num_levels,
                                      HYPRE_Int *num_block_coarse_points,
                                      HYPRE_Int  **block_coarse_indexes);

/**
 * Set the coarse indices for the levels using an array of tags for all the
 * local degrees of freedom.
 * TODO: Rename the function to make it more descriptive.
 *
 * @param solver [IN] solver or preconditioner object
 * @param block_size [IN] system block size
 * @param max_num_levels [IN] maximum number of reduction levels
 * @param num_block_coarse_points [IN] number of coarse points per block per level
 * @param lvl_block_coarse_indexes [IN] indices for the coarse points per level
 * @param point_marker_array [IN] array of tags for the local degrees of freedom
 **/
HYPRE_Int HYPRE_MGRSetCpointsByPointMarkerArray( HYPRE_Solver solver,
                                                 HYPRE_Int  block_size,
                                                 HYPRE_Int  max_num_levels,
                                                 HYPRE_Int  *num_block_coarse_points,
                                                 HYPRE_Int  **lvl_block_coarse_indexes,
                                                 HYPRE_Int  *point_marker_array);

/**
 * (Optional) Set non C-points to F-points.
 * This routine determines how the coarse points are selected for the next level
 * reduction. Options for \e nonCptToFptFlag are:
 *
 *    - 0 : Allow points not prescribed as C points to be potentially set as C points
 *          using classical AMG coarsening strategies (currently uses CLJP-coarsening).
 *    - 1 : Fix points not prescribed as C points to be F points for the next reduction
 *
 **/
HYPRE_Int
HYPRE_MGRSetNonCpointsToFpoints( HYPRE_Solver solver,
                                 HYPRE_Int nonCptToFptFlag);

/**
 * (Optional) Set maximum number of coarsening (or reduction) levels.
 * The default is 10.
 **/
HYPRE_Int
HYPRE_MGRSetMaxCoarseLevels( HYPRE_Solver solver,
                             HYPRE_Int maxlev );

/**
 * (Optional) Set the system block size.
 * This should match the block size set in the MGRSetCpointsByBlock function.
 * The default is 1.
 **/
HYPRE_Int
HYPRE_MGRSetBlockSize( HYPRE_Solver solver,
                       HYPRE_Int bsize );

/**
 * (Optional) Defines indexes of coarse nodes to be kept to the coarsest level.
 * These indexes are passed down through the MGR hierarchy to the coarsest grid
 * of the coarse grid (BoomerAMG) solver.
 *
 * @param solver [IN] solver or preconditioner object
 * @param reserved_coarse_size [IN] number of reserved coarse points
 * @param reserved_coarse_nodes [IN] (global) indexes of reserved coarse points
 **/
HYPRE_Int
HYPRE_MGRSetReservedCoarseNodes( HYPRE_Solver solver,
                                 HYPRE_Int reserved_coarse_size,
                                 HYPRE_BigInt *reserved_coarse_nodes );

/**
 * (Optional) Set the level for reducing the reserved Cpoints before the coarse
 * grid solve. This is necessary for some applications, such as phase transitions.
 * The default is 0 (no reduction, i.e. keep the reserved cpoints in the coarse grid solve).
 *
 * The default setup for the reduction is as follows:
 *    interp_type = 2
 *    restrict_type = 0
 *    F-relax method = 99
 *    Galerkin coarse grid
 **/
HYPRE_Int
HYPRE_MGRSetReservedCpointsLevelToKeep( HYPRE_Solver solver, HYPRE_Int level);

/**
 * (Optional) Set the relaxation type for F-relaxation.
 * Currently supports the following flavors of relaxation types
 * as described in the \e BoomerAMGSetRelaxType:
 * \e relax_type 0, 3 - 8, 13, 14, 18. Also supports AMG (options 1 and 2)
 *    and direct solver variants (9, 99, 199). See \e HYPRE_MGRSetLevelFRelaxType for details.
 **/
HYPRE_Int
HYPRE_MGRSetRelaxType(HYPRE_Solver solver,
                      HYPRE_Int relax_type );

/**
 * (Optional) Set the strategy for F-relaxation.
 * Options for \e relax_method are:
 *
 *    - 0 : Single-level relaxation sweeps for F-relaxation as prescribed by \e MGRSetRelaxType
 *    - 1 : Multi-level relaxation strategy for F-relaxation (V(1,0) cycle currently supported).
 *
 *    NOTE: This function will be removed in favor of \e HYPRE_MGRSetLevelFRelaxType!!
 **/
HYPRE_Int
HYPRE_MGRSetFRelaxMethod(HYPRE_Solver solver,
                         HYPRE_Int relax_method );

/**
 * (Optional) This function is an extension of HYPRE_MGRSetFRelaxMethod. It allows setting
 * the F-relaxation strategy for each MGR level.
 **/
HYPRE_Int
HYPRE_MGRSetLevelFRelaxMethod(HYPRE_Solver solver,
                              HYPRE_Int *relax_method );

/**
 * (Optional) Set the relaxation type for F-relaxation at each level.
 * This function takes precedence over, and will replace \e HYPRE_MGRSetFRelaxMethod
 * and HYPRE_MGRSetRelaxType.
 * Options for \e relax_type entries are:
 *
 *    - 0, 3 - 8, 13, 14, 18: (as described in \e BoomerAMGSetRelaxType)
 *    - 1 : Multi-level relaxation strategy for F-relaxation (V(1,0) cycle currently supported).
 *    - 2 : AMG
 *    - 9, 99, 199 : Gaussian Elimination variants (GE, GE with pivoting, direct inversion respectively)
 **/
HYPRE_Int
HYPRE_MGRSetLevelFRelaxType(HYPRE_Solver solver,
                            HYPRE_Int *relax_type );

/**
 * (Optional) Set the strategy for coarse grid computation.
 * Options for \e cg_method are:
 *
 *    - 0 : Galerkin coarse grid computation using RAP.
 *    - 5 : Galerkin coarse grid computation using RAI (injective prolongation).
 *    - 1 - 4 : Non-Galerkin coarse grid computation with dropping strategy.
 *         - 1: inv(A_FF) approximated by its (block) diagonal inverse
 *         - 2: CPR-like approximation with inv(A_FF) approximated by its diagonal inverse
 *         - 3: CPR-like approximation with inv(A_FF) approximated by its block diagonal inverse
 *         - 4: inv(A_FF) approximated by sparse approximate inverse
 **/
HYPRE_Int
HYPRE_MGRSetCoarseGridMethod(HYPRE_Solver solver,
                             HYPRE_Int *cg_method );

/**
 * (Optional) Set the number of functions for F-relaxation V-cycle.
 * For problems like elasticity, one may want to perform coarsening and
 * interpolation for block matrices. The number of functions corresponds
 * to the number of scalar PDEs in the system.
 **/
HYPRE_Int
HYPRE_MGRSetLevelFRelaxNumFunctions(HYPRE_Solver solver,
                                    HYPRE_Int *num_functions);

/**
 * (Optional) Set the strategy for computing the MGR restriction operator.
 *
 * Options for \e restrict_type are:
 *
 *    - 0    : injection \f$[0  I]\f$
 *    - 1    : unscaled (not recommended)
 *    - 2    : diagonal scaling (Jacobi)
 *    - 3    : approximate inverse
 *    - 4    : pAIR distance 1
 *    - 5    : pAIR distance 2
 *    - 12   : Block Jacobi
 *    - 13   : CPR-like restriction operator
 *    - 14   : (Block) Column-lumped restriction
 *    - else : use classical modified interpolation
 *
 * The default is injection.
 **/
HYPRE_Int
HYPRE_MGRSetRestrictType( HYPRE_Solver solver,
                          HYPRE_Int restrict_type);

/**
 * (Optional) This function is an extension of \e HYPRE_MGRSetRestrictType. It allows setting
 * the restriction operator strategy for each MGR level.
 **/
HYPRE_Int
HYPRE_MGRSetLevelRestrictType( HYPRE_Solver solver,
                               HYPRE_Int *restrict_type);

/**
 * (Optional) Set number of restriction sweeps.
 * This option is for \e restrict_type > 2.
 **/
HYPRE_Int
HYPRE_MGRSetNumRestrictSweeps( HYPRE_Solver solver,
                               HYPRE_Int nsweeps );

/**
 * (Optional) Set the strategy for computing the MGR interpolation operator.
 * Options for \e interp_type are:
 *
 *    - 0    : injection \f$[0  I]^{T}\f$
 *    - 1    : L1-Jacobi
 *    - 2    : diagonal scaling (Jacobi)
 *    - 3    : classical modified interpolation
 *    - 4    : approximate inverse
 *    - 12   : Block Jacobi
 *    - else : classical modified interpolation
 *
 * The default is diagonal scaling.
 **/
HYPRE_Int
HYPRE_MGRSetInterpType( HYPRE_Solver solver,
                        HYPRE_Int interp_type );

/**
 * (Optional) This function is an extension of \e HYPRE_MGRSetInterpType. It allows setting
 * the prolongation (interpolation) operator strategy for each MGR level.
 **/
HYPRE_Int
HYPRE_MGRSetLevelInterpType( HYPRE_Solver solver,
                             HYPRE_Int *interp_type );

/**
 * (Optional) Set number of relaxation sweeps.
 * This option is for the "single level" F-relaxation (\e relax_method = 0).
 **/
HYPRE_Int
HYPRE_MGRSetNumRelaxSweeps( HYPRE_Solver solver,
                            HYPRE_Int nsweeps );

/**
 * (Optional) This function is an extension of \e HYPRE_MGRSetNumRelaxSweeps. It allows setting
 * the number of single-level relaxation sweeps for each MGR level.
 **/
HYPRE_Int
HYPRE_MGRSetLevelNumRelaxSweeps( HYPRE_Solver solver,
                                 HYPRE_Int *nsweeps );

/**
 * (Optional) Set number of interpolation sweeps.
 * This option is for \e interp_type > 2.
 **/
HYPRE_Int
HYPRE_MGRSetNumInterpSweeps( HYPRE_Solver solver,
                             HYPRE_Int nsweeps );

/**
 * (Optional) Set block size for block (global) smoother and interp/restriction.
 * This option is for \e interp_type/restrict_type == 12, and
 * \e smooth_type == 0 or 1.
 **/
HYPRE_Int
HYPRE_MGRSetBlockJacobiBlockSize( HYPRE_Solver solver,
                                  HYPRE_Int blk_size );

/**
 * (Optional) Set the fine grid solver.
 *
 * @param solver [IN] MGR solver/preconditioner object
 * @param fine_grid_solver_solve [IN] solve routine
 * @param fine_grid_solver_setup [IN] setup routine
 * @param fine_grid_solver [IN] fine grid solver object
 **/
HYPRE_Int HYPRE_MGRSetFSolver(HYPRE_Solver             solver,
                              HYPRE_PtrToParSolverFcn  fine_grid_solver_solve,
                              HYPRE_PtrToParSolverFcn  fine_grid_solver_setup,
                              HYPRE_Solver             fsolver );

/**
 * (Optional) Set the F-relaxation solver at a given level.
 *
 * @param level [IN] MGR solver level
 * @param solver [IN] MGR solver/preconditioner object
 * @param fsolver [IN] F-relaxation solver object
 **/
HYPRE_Int HYPRE_MGRSetFSolverAtLevel(HYPRE_Int     level,
                                     HYPRE_Solver  solver,
                                     HYPRE_Solver  fsolver );

/**
 * (Optional) Extract A_FF block from matrix A.
 *
 * TODO (VPM): Does this need to be exposed? Move to parcsr_mv?
 **/
HYPRE_Int HYPRE_MGRBuildAff(HYPRE_ParCSRMatrix A,
                            HYPRE_Int *CF_marker,
                            HYPRE_Int debug_flag,
                            HYPRE_ParCSRMatrix *A_ff);

/**
 * (Optional) Set the coarse grid solver.
 * Currently uses BoomerAMG.
 * The default, if not set, is BoomerAMG with default options.
 *
 * @param solver [IN] MGR solver/preconditioner object
 * @param coarse_grid_solver_solve [IN] solve routine for BoomerAMG
 * @param coarse_grid_solver_setup [IN] setup routine for BoomerAMG
 * @param coarse_grid_solver [IN] coarse grid solver object
 **/
HYPRE_Int HYPRE_MGRSetCoarseSolver(HYPRE_Solver             solver,
                                   HYPRE_PtrToParSolverFcn  coarse_grid_solver_solve,
                                   HYPRE_PtrToParSolverFcn  coarse_grid_solver_setup,
                                   HYPRE_Solver             coarse_grid_solver );

/**
 * @brief (Optional) Set the verbosity level for MGR.
 *
 * @details Control what information gets printed by specifying the output levels
 * using this function. Each option corresponds to a specific type of information, and you
 * can activate several of them at the same time by summing their respective numeric codes,
*  which are given below:
 *
 *   - 1:   Print MGR's setup information.
 *   - 2:   Print MGR's solve information.
 *   - 4:   Print MGR's parameters information.
 *   - 8:   Set print mode for matrices and vectors to ASCII (binary mode is used by default)
 *   - 16:  Print the finest level matrix to NP files where NP is the number of ranks.
 *   - 32:  Print the finest level right-hand-side to NP files.
 *   - 64:  Print the coarsest level matrix to NP files.
 *   - 128: Print the full MGR hierarchy (operator, interpolation, and restriction).
 *
 * @param solver [IN] The solver to configure.
 * @param print_level [IN] The desired output level.
 *
 * @example To print setup information (1); fine matrix (16) and rhs (32) to binary files,
 * set \c print_level to 49 (1 + 16 + 32). In the previous example, to use ASCII
 * files for matrices and vectors, set \c print_level to 57 (1 + 8 + 16 + 32).
 *
 * @note The default print level is zero, which means no information will be
 * printed by default. Options starting from 8 are intended for developers' usage.
 **/
HYPRE_Int
HYPRE_MGRSetPrintLevel( HYPRE_Solver solver,
                        HYPRE_Int print_level );

/**
 * (Optional) Set the print level of the F-relaxation solver
 **/
HYPRE_Int
HYPRE_MGRSetFrelaxPrintLevel( HYPRE_Solver solver,
                              HYPRE_Int print_level );

/**
 * (Optional) Set the print level of the coarse grid solver
 **/
HYPRE_Int
HYPRE_MGRSetCoarseGridPrintLevel( HYPRE_Solver solver,
                                  HYPRE_Int print_level );

/**
 * (Optional) Set the threshold for dropping small entries on the coarse grid at each level.
 * No dropping is applied if \e threshold = 0.0 (default).
 **/
HYPRE_Int
HYPRE_MGRSetTruncateCoarseGridThreshold( HYPRE_Solver solver,
                                         HYPRE_Real threshold);

/**
 * (Optional) Requests logging of solver diagnostics.
 * Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default is 0, do nothing.  The latest
 * residual will be available if logging > 1.
 **/
HYPRE_Int
HYPRE_MGRSetLogging( HYPRE_Solver solver,
                     HYPRE_Int logging );

/**
 * (Optional) Set maximum number of iterations if used as a solver.
 * Set this to 1 if MGR is used as a preconditioner. The default is 20.
 **/
HYPRE_Int
HYPRE_MGRSetMaxIter( HYPRE_Solver solver,
                     HYPRE_Int max_iter );

/**
 * (Optional) Set the convergence tolerance for the MGR solver.
 * Use tol = 0.0 if MGR is used as a preconditioner. The default is 1.e-6.
 **/
HYPRE_Int
HYPRE_MGRSetTol( HYPRE_Solver solver,
                 HYPRE_Real tol );

/**
 * (Optional) Determines how many sweeps of global smoothing to do.
 * Default is 0 (no global smoothing).
 **/
HYPRE_Int
HYPRE_MGRSetMaxGlobalSmoothIters( HYPRE_Solver solver,
                                  HYPRE_Int smooth_iter );

/**
 * (Optional) Determines how many sweeps of global smoothing to do on each level.
 * Default is 0 (no global smoothing).
 **/
HYPRE_Int
HYPRE_MGRSetLevelSmoothIters( HYPRE_Solver solver,
                              HYPRE_Int *smooth_iters );
/**
 * (Optional) Set the cycle for global smoothing.
 * Options for \e global_smooth_cycle are:
 *    - 1 : Pre-smoothing - Down cycle (default)
 *    - 2 : Post-smoothing - Up cycle
 **/
HYPRE_Int
HYPRE_MGRSetGlobalSmoothCycle( HYPRE_Solver solver,
                               HYPRE_Int global_smooth_cycle );

/**
 * (Optional) Determines type of global smoother.
 * Options for \e smooth_type are:
 *
 *    -  0 : block Jacobi (default)
 *    -  1 : block Gauss-Seidel
 *    -  2 : Jacobi
 *    -  3 : Gauss-Seidel, sequential (very slow!)
 *    -  4 : Gauss-Seidel, interior points in parallel, boundary sequential (slow!)
 *    -  5 : hybrid Gauss-Seidel or SOR, forward solve
 *    -  6 : hybrid Gauss-Seidel or SOR, backward solve
 *    -  8 : Euclid (ILU)
 *    - 16 : HYPRE_ILU
 *    - 18 : L1-Jacobi
 **/
HYPRE_Int
HYPRE_MGRSetGlobalSmoothType( HYPRE_Solver solver,
                              HYPRE_Int smooth_type );

/**
 * (Optional) Determines type of global smoother for each level.
 * See \e HYPRE_MGRSetGlobalSmoothType for global smoother options.
 **/
HYPRE_Int
HYPRE_MGRSetLevelSmoothType( HYPRE_Solver solver,
                             HYPRE_Int *smooth_type );

/**
 * (Optional) Return the number of MGR iterations.
 **/
HYPRE_Int
HYPRE_MGRGetNumIterations( HYPRE_Solver solver,
                           HYPRE_Int *num_iterations );

/**
 * (Optional) Return the relative residual for the coarse level system.
 **/
HYPRE_Int
HYPRE_MGRGetCoarseGridConvergenceFactor( HYPRE_Solver solver,
                                         HYPRE_Real *conv_factor );

/**
 * (Optional) Set the maximum number of nonzeros per row for interpolation operators.
 **/
HYPRE_Int
HYPRE_MGRSetPMaxElmts( HYPRE_Solver solver,
                       HYPRE_Int P_max_elmts );

/**
 * (Optional) Set the maximum number of nonzeros per row for interpolation operators for each level.
 **/
HYPRE_Int
HYPRE_MGRSetLevelPMaxElmts( HYPRE_Solver solver,
                            HYPRE_Int *P_max_elmts );

/**
 * (Optional) Return the norm of the final relative residual.
 **/
HYPRE_Int
HYPRE_MGRGetFinalRelativeResidualNorm( HYPRE_Solver solver,
                                       HYPRE_Real *res_norm );

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/**
 * @name ParCSR ILU Solver
 *
 * (Parallel) Incomplete LU factorization.
 *
 * @{
 **/

/**
 * Create a solver object
 **/
HYPRE_Int
HYPRE_ILUCreate( HYPRE_Solver *solver );

/**
 * Destroy a solver object
 **/
HYPRE_Int
HYPRE_ILUDestroy( HYPRE_Solver solver );

/**
 * Setup the ILU solver or preconditioner.
 * If used as a preconditioner, this function should be passed
 * to the iterative solver \e SetPrecond function.
 *
 * @param solver [IN] object to be set up.
 * @param A [IN] ParCSR matrix used to construct the solver/preconditioner.
 * @param b right-hand-side of the linear system to be solved (Ignored by this function).
 * @param x approximate solution of the linear system to be solved (Ignored by this function).
 **/
HYPRE_Int
HYPRE_ILUSetup( HYPRE_Solver solver,
                HYPRE_ParCSRMatrix A,
                HYPRE_ParVector b,
                HYPRE_ParVector x      );
/**
* Solve the system or apply ILU as a preconditioner.
* If used as a preconditioner, this function should be passed
* to the iterative solver \e SetPrecond function.
*
* @param solver [IN] solver or preconditioner object to be applied.
* @param A [IN] ParCSR matrix, matrix of the linear system to be solved
* @param b [IN] right hand side of the linear system to be solved
* @param x [OUT] approximated solution of the linear system to be solved
**/
HYPRE_Int
HYPRE_ILUSolve( HYPRE_Solver solver,
                HYPRE_ParCSRMatrix A,
                HYPRE_ParVector b,
                HYPRE_ParVector x      );

/**
 * (Optional) Set maximum number of iterations if used as a solver.
 * Set this to 1 if ILU is used as a preconditioner. The default is 20.
 **/
HYPRE_Int
HYPRE_ILUSetMaxIter( HYPRE_Solver solver, HYPRE_Int max_iter );

/**
 * (Optional) Set the algorithm type to compute the ILU factorization. Options are:
 *
 *    -  0 : Non-iterative algorithm (default)
 *    -  1 : Asynchronous with in-place storage
 *    -  2 : Asynchronous with explicit storage splitting
 *    -  3 : Synchronous with explicit storage splitting
 *    -  4 : Semi-synchronous with explicit storage splitting
 *
 * Note: Iterative ILU is available only for zero fill-in and it depends on rocSPARSE.
 **/
HYPRE_Int
HYPRE_ILUSetIterativeSetupType( HYPRE_Solver solver, HYPRE_Int iter_setup_type );

/**
 * (Optional) Set the compute option for iterative ILU in an additive fashion, i.e.; multiple
 * options can be turned on by summing their respective numeric codes as given below:
 *
 *    -  2: Use stopping tolerance to finish the algorithm
 *    -  4: Compute correction norms
 *    -  8: Compute residual norms
 *    - 16: Save convergence history
 *    - 32: Use rocSPARSE's internal COO format
 *
 * The iterative ILU algorithm can terminate based on the maximum number of iterations (default)
 * or a target tolerance (option 2). In the tolerance-based case, the max. number of iterations
 * is still used to terminate the algorithm in case it does not converge to the requested
 * tolerance. In addition, the tolerance-based mode uses residual norms by default (option 8).
 * To use correction norms instead, enable option 4. Lastly, the convergence history for
 * computing the triangular factors can be saved and printed out by enabling option 16.
 *
 * Note: Iterative ILU is available only for zero fill-in, and it depends on rocSPARSE.
 **/
HYPRE_Int
HYPRE_ILUSetIterativeSetupOption( HYPRE_Solver solver, HYPRE_Int iter_setup_option );

/**
 * (Optional) Set the max. number of iterations for the iterative ILU algorithm.
 *
 * Note: Iterative ILU is available only for zero fill-in and it depends on rocSPARSE.
 **/
HYPRE_Int
HYPRE_ILUSetIterativeSetupMaxIter( HYPRE_Solver solver, HYPRE_Int iter_setup_max_iter );

/**
 * (Optional) Set the stop tolerance for the iterative ILU algorithm.
 *
 * Note: Iterative ILU is available only for zero fill-in and it depends on rocSPARSE.
 **/
HYPRE_Int
HYPRE_ILUSetIterativeSetupTolerance( HYPRE_Solver solver, HYPRE_Real iter_setup_tolerance );

/**
 * (Optional) Set triangular solver type. Options are:
 *
 *    -  0 : iterative
 *    -  1 : direct (default)
 **/
HYPRE_Int
HYPRE_ILUSetTriSolve( HYPRE_Solver solver, HYPRE_Int tri_solve );

/**
 * (Optional) Set number of lower Jacobi iterations for the triangular L solves
 * Set this to integer > 0 when using iterative tri_solve (0). The default is 5 iterations.
 **/
HYPRE_Int
HYPRE_ILUSetLowerJacobiIters( HYPRE_Solver solver, HYPRE_Int lower_jacobi_iterations );

/**
 * (Optional) Set number of upper Jacobi iterations for the triangular U solves
 * Set this to integer > 0 when using iterative tri_solve (0). The default is 5 iterations.
 **/
HYPRE_Int
HYPRE_ILUSetUpperJacobiIters( HYPRE_Solver solver, HYPRE_Int upper_jacobi_iterations );

/**
 * (Optional) Set the convergence tolerance for ILU.
 * Use tol = 0.0 if ILU is used as a preconditioner. The default is 1.e-7.
 **/
HYPRE_Int
HYPRE_ILUSetTol( HYPRE_Solver solver, HYPRE_Real tol );

/**
 * (Optional) Set the level of fill k, for level-based ILU(k)
 * The default is 0 (for ILU(0)).
 **/
HYPRE_Int
HYPRE_ILUSetLevelOfFill( HYPRE_Solver solver, HYPRE_Int lfil );

/**
 * (Optional) Set the max non-zeros per row in L and U factors (for ILUT)
 * The default is 1000.
 **/
HYPRE_Int
HYPRE_ILUSetMaxNnzPerRow( HYPRE_Solver solver, HYPRE_Int nzmax );

/**
 * (Optional) Set the threshold for dropping in L and U factors (for ILUT).
 * Any fill-in less than this threshold is dropped in the factorization.
 * The default is 1.0e-2.
 **/
HYPRE_Int
HYPRE_ILUSetDropThreshold( HYPRE_Solver solver, HYPRE_Real threshold );

/**
 * (Optional) Set the array of thresholds for dropping in ILUT.
 * B, E, and F correspond to upper left, lower left and upper right
 * of 2 x 2 block decomposition respectively.
 * Any fill-in less than threshold is dropped in the factorization.
 *    - threshold[0] : threshold for matrix B.
 *    - threshold[1] : threshold for matrix E and F.
 *    - threshold[2] : threshold for matrix S (Schur Complement).
 * The default is 1.0e-2.
 **/
HYPRE_Int
HYPRE_ILUSetDropThresholdArray( HYPRE_Solver solver, HYPRE_Real *threshold );

/**
 * (Optional) Set the threshold for dropping in Newton–Schulz–Hotelling iteration (NSH-ILU).
 * Any entries less than this threshold are dropped when forming the approximate inverse matrix.
 * The default is 1.0e-2.
 **/
HYPRE_Int
HYPRE_ILUSetNSHDropThreshold( HYPRE_Solver solver, HYPRE_Real threshold );

/**
 * (Optional) Set the array of thresholds for dropping in Newton–Schulz–Hotelling
 * iteration (for NSH-ILU).  Any fill-in less than thresholds is dropped when
 * forming the approximate inverse matrix.
 *
 *    - threshold[0] : threshold for Minimal Residual iteration (initial guess for NSH).
 *    - threshold[1] : threshold for Newton–Schulz–Hotelling iteration.
 *
 * The default is 1.0e-2.
 **/
HYPRE_Int
HYPRE_ILUSetNSHDropThresholdArray( HYPRE_Solver solver, HYPRE_Real *threshold );

/**
 * (Optional) Set maximum number of iterations for Schur System Solve.
 * For GMRES-ILU, this is the maximum number of iterations for GMRES.
 * The Krylov dimension for GMRES is set equal to this value to avoid restart.
 * For NSH-ILU, this is the maximum number of iterations for NSH solve.
 * The default is 5.
 **/
HYPRE_Int
HYPRE_ILUSetSchurMaxIter( HYPRE_Solver solver, HYPRE_Int ss_max_iter );

/**
 * Set the type of ILU factorization.
 *
 * Options for \e ilu_type are:
 *    - 0 : BJ with ILU(k) (default, with k = 0)
 *    - 1 : BJ with ILUT
 *    - 10 : GMRES with ILU(k)
 *    - 11 : GMRES with ILUT
 *    - 20 : NSH with ILU(k)
 *    - 21 : NSH with ILUT
 *    - 30 : RAS with ILU(k)
 *    - 31 : RAS with ILUT
 *    - 40 : (nonsymmetric permutation) DDPQ-GMRES with ILU(k)
 *    - 41 : (nonsymmetric permutation) DDPQ-GMRES with ILUT
 *    - 50 : GMRES with RAP-ILU(0) using MILU(0) for P
 **/
HYPRE_Int
HYPRE_ILUSetType( HYPRE_Solver solver, HYPRE_Int ilu_type );

/**
 * Set the type of reordering for the local matrix.
 *
 * Options for \e reordering_type are:
 *    - 0 : No reordering
 *    - 1 : RCM (default)
 **/
HYPRE_Int
HYPRE_ILUSetLocalReordering( HYPRE_Solver solver, HYPRE_Int reordering_type );

/**
 * (Optional) Set the print level to print setup and solve information.
 *
 *    - 0 : no printout (default)
 *    - 1 : print setup information
 *    - 2 : print solve information
 *    - 3 : print both setup and solve information
 **/
HYPRE_Int
HYPRE_ILUSetPrintLevel( HYPRE_Solver solver, HYPRE_Int print_level );

/**
 * (Optional) Requests logging of solver diagnostics.
 * Requests additional computations for diagnostic and similar
 * data to be logged by the user. Default is 0, do nothing.  The latest
 * residual will be available if logging > 1.
 **/
HYPRE_Int
HYPRE_ILUSetLogging( HYPRE_Solver solver, HYPRE_Int logging );

/**
 * (Optional) Return the number of ILU iterations.
 **/
HYPRE_Int
HYPRE_ILUGetNumIterations( HYPRE_Solver solver, HYPRE_Int *num_iterations );

/**
 * (Optional) Return the norm of the final relative residual.
 **/
HYPRE_Int
HYPRE_ILUGetFinalRelativeResidualNorm(  HYPRE_Solver solver, HYPRE_Real *res_norm );

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Miscellaneous: These probably do not belong in the interface.
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix GenerateLaplacian(MPI_Comm    comm,
                                     HYPRE_BigInt   nx,
                                     HYPRE_BigInt   ny,
                                     HYPRE_BigInt   nz,
                                     HYPRE_Int   P,
                                     HYPRE_Int   Q,
                                     HYPRE_Int   R,
                                     HYPRE_Int   p,
                                     HYPRE_Int   q,
                                     HYPRE_Int   r,
                                     HYPRE_Real *value);

HYPRE_ParCSRMatrix GenerateLaplacian27pt(MPI_Comm    comm,
                                         HYPRE_BigInt   nx,
                                         HYPRE_BigInt   ny,
                                         HYPRE_BigInt   nz,
                                         HYPRE_Int   P,
                                         HYPRE_Int   Q,
                                         HYPRE_Int   R,
                                         HYPRE_Int   p,
                                         HYPRE_Int   q,
                                         HYPRE_Int   r,
                                         HYPRE_Real *value);

HYPRE_ParCSRMatrix GenerateLaplacian9pt(MPI_Comm    comm,
                                        HYPRE_BigInt   nx,
                                        HYPRE_BigInt   ny,
                                        HYPRE_Int   P,
                                        HYPRE_Int   Q,
                                        HYPRE_Int   p,
                                        HYPRE_Int   q,
                                        HYPRE_Real *value);

HYPRE_ParCSRMatrix GenerateDifConv(MPI_Comm    comm,
                                   HYPRE_BigInt   nx,
                                   HYPRE_BigInt   ny,
                                   HYPRE_BigInt   nz,
                                   HYPRE_Int   P,
                                   HYPRE_Int   Q,
                                   HYPRE_Int   R,
                                   HYPRE_Int   p,
                                   HYPRE_Int   q,
                                   HYPRE_Int   r,
                                   HYPRE_Real *value);

HYPRE_ParCSRMatrix
GenerateRotate7pt(MPI_Comm   comm,
                  HYPRE_BigInt  nx,
                  HYPRE_BigInt  ny,
                  HYPRE_Int  P,
                  HYPRE_Int  Q,
                  HYPRE_Int  p,
                  HYPRE_Int  q,
                  HYPRE_Real alpha,
                  HYPRE_Real eps );

HYPRE_ParCSRMatrix
GenerateVarDifConv(MPI_Comm         comm,
                   HYPRE_BigInt        nx,
                   HYPRE_BigInt        ny,
                   HYPRE_BigInt        nz,
                   HYPRE_Int        P,
                   HYPRE_Int        Q,
                   HYPRE_Int        R,
                   HYPRE_Int        p,
                   HYPRE_Int        q,
                   HYPRE_Int        r,
                   HYPRE_Real       eps,
                   HYPRE_ParVector *rhs_ptr);

HYPRE_ParCSRMatrix
GenerateRSVarDifConv(MPI_Comm         comm,
                     HYPRE_BigInt        nx,
                     HYPRE_BigInt        ny,
                     HYPRE_BigInt        nz,
                     HYPRE_Int        P,
                     HYPRE_Int        Q,
                     HYPRE_Int        R,
                     HYPRE_Int        p,
                     HYPRE_Int        q,
                     HYPRE_Int        r,
                     HYPRE_Real       eps,
                     HYPRE_ParVector *rhs_ptr,
                     HYPRE_Int        type);

float*
hypre_GenerateCoordinates(MPI_Comm  comm,
                          HYPRE_BigInt nx,
                          HYPRE_BigInt ny,
                          HYPRE_BigInt nz,
                          HYPRE_Int P,
                          HYPRE_Int Q,
                          HYPRE_Int R,
                          HYPRE_Int p,
                          HYPRE_Int q,
                          HYPRE_Int r,
                          HYPRE_Int coorddim);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * (Optional) Switches on use of Jacobi interpolation after computing
 * an original interpolation
 **/
HYPRE_Int HYPRE_BoomerAMGSetPostInterpType(HYPRE_Solver solver,
                                           HYPRE_Int    post_interp_type);

/**
 * (Optional) Sets a truncation threshold for Jacobi interpolation.
 **/
HYPRE_Int HYPRE_BoomerAMGSetJacobiTruncThreshold(HYPRE_Solver solver,
                                                 HYPRE_Real   jacobi_trunc_threshold);

/**
 * (Optional) Defines the number of relaxation steps for CR
 * The default is 2.
 **/
HYPRE_Int HYPRE_BoomerAMGSetNumCRRelaxSteps(HYPRE_Solver solver,
                                            HYPRE_Int    num_CR_relax_steps);

/**
 * (Optional) Defines convergence rate for CR
 * The default is 0.7.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCRRate(HYPRE_Solver solver,
                                   HYPRE_Real   CR_rate);

/**
 * (Optional) Defines strong threshold for CR
 * The default is 0.0.
 **/
HYPRE_Int HYPRE_BoomerAMGSetCRStrongTh(HYPRE_Solver solver,
                                       HYPRE_Real   CR_strong_th);

/**
 * (Optional) Defines whether to use CG
 **/
HYPRE_Int HYPRE_BoomerAMGSetCRUseCG(HYPRE_Solver solver,
                                    HYPRE_Int    CR_use_CG);

/**
 * (Optional) Defines the Type of independent set algorithm used for CR
 **/
HYPRE_Int HYPRE_BoomerAMGSetISType(HYPRE_Solver solver,
                                   HYPRE_Int    IS_type);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name ParCSR LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \ref Eigensolvers.
 *
 * @{
 **/

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

/*
 * Print multivector to file.
 **/
HYPRE_Int
HYPRE_ParCSRMultiVectorPrint(void *x_,
                             const char *fileName);

/*
 * Read multivector from file.
 **/
void *
HYPRE_ParCSRMultiVectorRead(MPI_Comm comm,
                            void *ii_,
                            const char *fileName);

/**@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
