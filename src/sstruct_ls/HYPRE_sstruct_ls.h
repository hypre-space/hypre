/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.26 $
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

typedef HYPRE_Int (*HYPRE_PtrToSStructSolverFcn)(HYPRE_SStructSolver,
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

typedef HYPRE_Int (*HYPRE_PtrToModifyPCFcn)(HYPRE_Solver,
                                            HYPRE_Int,
                                            double);
#endif

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct SysPFMG Solver
 *
 * SysPFMG is a semicoarsening multigrid solver similar to PFMG, but for systems
 * of PDEs.  For periodic problems, users should try to set the grid size in
 * periodic dimensions to be as close to a power-of-two as possible (for more
 * details, see \Ref{Struct PFMG Solver}).
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructSysPFMGDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetup(HYPRE_SStructSolver solver,
                          HYPRE_SStructMatrix A,
                          HYPRE_SStructVector b,
                          HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSolve(HYPRE_SStructSolver solver,
                          HYPRE_SStructMatrix A,
                          HYPRE_SStructVector b,
                          HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetTol(HYPRE_SStructSolver solver,
                           double              tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetMaxIter(HYPRE_SStructSolver solver,
                               HYPRE_Int           max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetRelChange(HYPRE_SStructSolver solver,
                                 HYPRE_Int           rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetNonZeroGuess(HYPRE_SStructSolver solver);

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
HYPRE_Int
HYPRE_SStructSysPFMGSetRelaxType(HYPRE_SStructSolver solver,
                                 HYPRE_Int           relax_type);

/**
 * (Optional) Set Jacobi Weight.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetJacobiWeight(HYPRE_SStructSolver solver,
                                    double              weight);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetNumPreRelax(HYPRE_SStructSolver solver,
                                   HYPRE_Int           num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetNumPostRelax(HYPRE_SStructSolver solver,
                                    HYPRE_Int           num_post_relax);

/**
 * (Optional) Skip relaxation on certain grids for isotropic problems.  This can
 * greatly improve efficiency by eliminating unnecessary relaxations when the
 * underlying problem is isotropic.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetSkipRelax(HYPRE_SStructSolver solver,
                                 HYPRE_Int           skip_relax);

/*
 * RE-VISIT
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetDxyz(HYPRE_SStructSolver  solver,
                            double              *dxyz);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetLogging(HYPRE_SStructSolver solver,
                               HYPRE_Int           logging);

/**
 * (Optional) Set the amount of printing to do to the screen.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGSetPrintLevel(HYPRE_SStructSolver solver,
                                  HYPRE_Int           print_level);


/**
 * Return the number of iterations taken.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGGetNumIterations(HYPRE_SStructSolver  solver,
                                     HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int
HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(HYPRE_SStructSolver solver,
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
HYPRE_Int
HYPRE_SStructSplitCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructSplitDestroy(HYPRE_SStructSolver solver);

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int
HYPRE_SStructSplitSetup(HYPRE_SStructSolver solver,
                        HYPRE_SStructMatrix A,
                        HYPRE_SStructVector b,
                        HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int
HYPRE_SStructSplitSolve(HYPRE_SStructSolver solver,
                        HYPRE_SStructMatrix A,
                        HYPRE_SStructVector b,
                        HYPRE_SStructVector x);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int
HYPRE_SStructSplitSetTol(HYPRE_SStructSolver solver,
                         double              tol);

/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int
HYPRE_SStructSplitSetMaxIter(HYPRE_SStructSolver solver,
                             HYPRE_Int           max_iter);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
HYPRE_Int
HYPRE_SStructSplitSetZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
HYPRE_Int
HYPRE_SStructSplitSetNonZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Set up the type of diagonal struct solver.  Either {\tt ssolver} is
 * set to {\tt HYPRE\_SMG} or {\tt HYPRE\_PFMG}.
 **/
HYPRE_Int
HYPRE_SStructSplitSetStructSolver(HYPRE_SStructSolver solver,
                                  HYPRE_Int           ssolver );

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int
HYPRE_SStructSplitGetNumIterations(HYPRE_SStructSolver  solver,
                                   HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int
HYPRE_SStructSplitGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
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
HYPRE_Int
HYPRE_SStructFACCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructFACDestroy2(HYPRE_SStructSolver solver);

/**
 * Re-distribute the composite matrix so that the amr hierachy is approximately
 * nested. Coarse underlying operators are also formed.
 **/
HYPRE_Int
HYPRE_SStructFACAMR_RAP(HYPRE_SStructMatrix  A,
                        HYPRE_Int          (*rfactors)[3],
                        HYPRE_SStructMatrix *fac_A);

/**
 * Set up the FAC solver structure .
 **/
HYPRE_Int
HYPRE_SStructFACSetup2(HYPRE_SStructSolver solver,
                       HYPRE_SStructMatrix A,
                       HYPRE_SStructVector b,
                       HYPRE_SStructVector x);

/**
 * Solve the system.
 **/
HYPRE_Int
HYPRE_SStructFACSolve3(HYPRE_SStructSolver solver,
                       HYPRE_SStructMatrix A,
                       HYPRE_SStructVector b,
                       HYPRE_SStructVector x);

/**
 * Set up amr structure
 **/
HYPRE_Int
HYPRE_SStructFACSetPLevels(HYPRE_SStructSolver solver,
                           HYPRE_Int           nparts,
                           HYPRE_Int          *plevels);
/**
 * Set up amr refinement factors
 **/
HYPRE_Int
HYPRE_SStructFACSetPRefinements(HYPRE_SStructSolver  solver,
                                HYPRE_Int            nparts,
                                HYPRE_Int          (*rfactors)[3] );

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 * Zero off the coarse level stencils reaching into a fine level grid.
 **/
HYPRE_Int
HYPRE_SStructFACZeroCFSten(HYPRE_SStructMatrix  A,
                           HYPRE_SStructGrid    grid,
                           HYPRE_Int            part,
                           HYPRE_Int            rfactors[3]);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 * Zero off the fine level stencils reaching into a coarse level grid.
 **/
HYPRE_Int
HYPRE_SStructFACZeroFCSten(HYPRE_SStructMatrix  A,
                           HYPRE_SStructGrid    grid,
                           HYPRE_Int            part);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 *  Places the identity in the coarse grid matrix underlying the fine patches.
 *  Required between each pair of amr levels.
 **/
HYPRE_Int
HYPRE_SStructFACZeroAMRMatrixData(HYPRE_SStructMatrix  A,
                                  HYPRE_Int            part_crse,
                                  HYPRE_Int            rfactors[3]);

/**
 * (Optional, but user must make sure that they do this function otherwise.)
 *  Places zeros in the coarse grid vector underlying the fine patches.
 *  Required between each pair of amr levels.
 **/
HYPRE_Int
HYPRE_SStructFACZeroAMRVectorData(HYPRE_SStructVector  b,
                                  HYPRE_Int           *plevels,
                                  HYPRE_Int          (*rfactors)[3] );

/**
 * (Optional) Set maximum number of FAC levels.
 **/
HYPRE_Int
HYPRE_SStructFACSetMaxLevels( HYPRE_SStructSolver solver , 
                              HYPRE_Int           max_levels );
/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int
HYPRE_SStructFACSetTol(HYPRE_SStructSolver solver,
                       double              tol);
/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int
HYPRE_SStructFACSetMaxIter(HYPRE_SStructSolver solver,
                           HYPRE_Int           max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int
HYPRE_SStructFACSetRelChange(HYPRE_SStructSolver solver,
                             HYPRE_Int           rel_change);

/**
 * (Optional) Use a zero initial guess.  This allows the solver to cut corners
 * in the case where a zero initial guess is needed (e.g., for preconditioning)
 * to reduce compuational cost.
 **/
HYPRE_Int
HYPRE_SStructFACSetZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Use a nonzero initial guess.  This is the default behavior, but
 * this routine allows the user to switch back after using {\tt SetZeroGuess}.
 **/
HYPRE_Int
HYPRE_SStructFACSetNonZeroGuess(HYPRE_SStructSolver solver);

/**
 * (Optional) Set relaxation type.  See \Ref{HYPRE_SStructSysPFMGSetRelaxType}
 * for appropriate values of {\tt relax\_type}.
 **/
HYPRE_Int
HYPRE_SStructFACSetRelaxType(HYPRE_SStructSolver solver,
                             HYPRE_Int           relax_type);
/**
 * (Optional) Set Jacobi weight if weighted Jacobi is used.
 **/
HYPRE_Int
HYPRE_SStructFACSetJacobiWeight(HYPRE_SStructSolver solver,
                                double              weight);
/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
HYPRE_Int
HYPRE_SStructFACSetNumPreRelax(HYPRE_SStructSolver solver,
                               HYPRE_Int           num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
HYPRE_Int
HYPRE_SStructFACSetNumPostRelax(HYPRE_SStructSolver solver,
                                HYPRE_Int           num_post_relax);
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
HYPRE_Int
HYPRE_SStructFACSetCoarseSolverType(HYPRE_SStructSolver solver,
                                    HYPRE_Int           csolver_type);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int
HYPRE_SStructFACSetLogging(HYPRE_SStructSolver solver,
                           HYPRE_Int           logging);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int
HYPRE_SStructFACGetNumIterations(HYPRE_SStructSolver  solver,
                                 HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int
HYPRE_SStructFACGetFinalRelativeResidualNorm(HYPRE_SStructSolver solver,
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
HYPRE_Int
HYPRE_SStructMaxwellCreate( MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructMaxwellDestroy( HYPRE_SStructSolver solver );

/**
 * Prepare to solve the system.  The coefficient data in {\tt b} and {\tt x} is
 * ignored here, but information about the layout of the data may be used.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetup(HYPRE_SStructSolver solver,
                          HYPRE_SStructMatrix A,
                          HYPRE_SStructVector b,
                          HYPRE_SStructVector x);

/**
 * Solve the system. Full coupling of the augmented system used
 * throughout the multigrid hierarchy.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSolve(HYPRE_SStructSolver solver,
                          HYPRE_SStructMatrix A,
                          HYPRE_SStructVector b,
                          HYPRE_SStructVector x);

/**
 * Solve the system. Full coupling of the augmented system used
 * only on the finest level, i.e., the node and edge multigrid
 * cycles are coupled only on the finest level.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSolve2(HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x);

/**
 * Sets the gradient operator in the Maxwell solver.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetGrad(HYPRE_SStructSolver solver,
                            HYPRE_ParCSRMatrix  T);

/**
 * Sets the coarsening factor.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetRfactors(HYPRE_SStructSolver solver,
                                HYPRE_Int           rfactors[3]);

/**
 * Finds the physical boundary row ranks on all levels.
 **/
HYPRE_Int
HYPRE_SStructMaxwellPhysBdy(HYPRE_SStructGrid  *grid_l,
                            HYPRE_Int           num_levels,
                            HYPRE_Int           rfactors[3],
                            HYPRE_Int        ***BdryRanks_ptr,
                            HYPRE_Int         **BdryRanksCnt_ptr );

/**
 * Eliminates the rows and cols corresponding to the physical boundary in
 * a parcsr matrix.
 **/
HYPRE_Int
HYPRE_SStructMaxwellEliminateRowsCols(HYPRE_ParCSRMatrix  parA,
                                      HYPRE_Int           nrows,
                                      HYPRE_Int          *rows );

/**
 * Zeros the rows corresponding to the physical boundary in
 * a par vector.
 **/
HYPRE_Int
HYPRE_SStructMaxwellZeroVector(HYPRE_ParVector  b,
                               HYPRE_Int       *rows,
                               HYPRE_Int        nrows );

/**
 * (Optional) Set the constant coefficient flag- Nedelec interpolation
 * used.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetSetConstantCoef(HYPRE_SStructSolver solver,
                                       HYPRE_Int           flag);

/**
 * (Optional) Creates a gradient matrix from the grid. This presupposes
 * a particular orientation of the edge elements.
 **/
HYPRE_Int
HYPRE_SStructMaxwellGrad(HYPRE_SStructGrid    grid,
                         HYPRE_ParCSRMatrix  *T);

/**
 * (Optional) Set the convergence tolerance.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetTol(HYPRE_SStructSolver solver,
                           double              tol);
/**
 * (Optional) Set maximum number of iterations.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetMaxIter(HYPRE_SStructSolver solver,
                               HYPRE_Int           max_iter);

/**
 * (Optional) Additionally require that the relative difference in
 * successive iterates be small.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetRelChange(HYPRE_SStructSolver solver,
                                 HYPRE_Int           rel_change);

/**
 * (Optional) Set number of relaxation sweeps before coarse-grid correction.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetNumPreRelax(HYPRE_SStructSolver solver,
                                   HYPRE_Int           num_pre_relax);

/**
 * (Optional) Set number of relaxation sweeps after coarse-grid correction.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetNumPostRelax(HYPRE_SStructSolver solver,
                                    HYPRE_Int           num_post_relax);

/**
 * (Optional) Set the amount of logging to do.
 **/
HYPRE_Int
HYPRE_SStructMaxwellSetLogging(HYPRE_SStructSolver solver,
                               HYPRE_Int           logging);

/**
 * Return the number of iterations taken.
 **/
HYPRE_Int
HYPRE_SStructMaxwellGetNumIterations(HYPRE_SStructSolver  solver,
                                     HYPRE_Int           *num_iterations);

/**
 * Return the norm of the final relative residual.
 **/
HYPRE_Int
HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(HYPRE_SStructSolver solver,
                                                 double             *norm);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct PCG Solver
 * 
 * These routines should be used in conjunction with the generic interface in
 * \Ref{PCG Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int
HYPRE_SStructPCGCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructPCGDestroy(HYPRE_SStructSolver solver);

HYPRE_Int
HYPRE_SStructPCGSetup(HYPRE_SStructSolver solver,
                      HYPRE_SStructMatrix A,
                      HYPRE_SStructVector b,
                      HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructPCGSolve(HYPRE_SStructSolver solver,
                      HYPRE_SStructMatrix A,
                      HYPRE_SStructVector b,
                      HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructPCGSetTol(HYPRE_SStructSolver solver,
                       double              tol);

HYPRE_Int
HYPRE_SStructPCGSetAbsoluteTol(HYPRE_SStructSolver solver,
                               double              tol);

HYPRE_Int
HYPRE_SStructPCGSetMaxIter(HYPRE_SStructSolver solver,
                           HYPRE_Int           max_iter);

HYPRE_Int
HYPRE_SStructPCGSetTwoNorm(HYPRE_SStructSolver solver,
                           HYPRE_Int           two_norm);

HYPRE_Int
HYPRE_SStructPCGSetRelChange(HYPRE_SStructSolver solver,
                             HYPRE_Int           rel_change);

HYPRE_Int
HYPRE_SStructPCGSetPrecond(HYPRE_SStructSolver          solver,
                           HYPRE_PtrToSStructSolverFcn  precond,
                           HYPRE_PtrToSStructSolverFcn  precond_setup,
                           void                        *precond_solver);

HYPRE_Int
HYPRE_SStructPCGSetLogging(HYPRE_SStructSolver solver,
                           HYPRE_Int           logging);

HYPRE_Int
HYPRE_SStructPCGSetPrintLevel(HYPRE_SStructSolver solver,
                              HYPRE_Int           level);

HYPRE_Int
HYPRE_SStructPCGGetNumIterations(HYPRE_SStructSolver  solver,
                                 HYPRE_Int           *num_iterations);

HYPRE_Int
HYPRE_SStructPCGGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                             double              *norm);

HYPRE_Int
HYPRE_SStructPCGGetResidual(HYPRE_SStructSolver   solver,
                            void                **residual);

/**
 * Setup routine for diagonal preconditioning.
 **/
HYPRE_Int
HYPRE_SStructDiagScaleSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector y,
                            HYPRE_SStructVector x);

/**
 * Solve routine for diagonal preconditioning.
 **/
HYPRE_Int
HYPRE_SStructDiagScale(HYPRE_SStructSolver solver,
                       HYPRE_SStructMatrix A,
                       HYPRE_SStructVector y,
                       HYPRE_SStructVector x);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct GMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{GMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int
HYPRE_SStructGMRESCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructGMRESDestroy(HYPRE_SStructSolver solver);

HYPRE_Int
HYPRE_SStructGMRESSetup(HYPRE_SStructSolver solver,
                        HYPRE_SStructMatrix A,
                        HYPRE_SStructVector b,
                        HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructGMRESSolve(HYPRE_SStructSolver solver,
                        HYPRE_SStructMatrix A,
                        HYPRE_SStructVector b,
                        HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructGMRESSetTol(HYPRE_SStructSolver solver,
                         double              tol);

HYPRE_Int
HYPRE_SStructGMRESSetAbsoluteTol(HYPRE_SStructSolver solver,
                                 double              tol);

/*
 * RE-VISIT
 **/
HYPRE_Int
HYPRE_SStructGMRESSetMinIter(HYPRE_SStructSolver solver,
                             HYPRE_Int           min_iter);

HYPRE_Int
HYPRE_SStructGMRESSetMaxIter(HYPRE_SStructSolver solver,
                             HYPRE_Int           max_iter);

HYPRE_Int
HYPRE_SStructGMRESSetKDim(HYPRE_SStructSolver solver,
                          HYPRE_Int           k_dim);

/*
 * RE-VISIT
 **/
HYPRE_Int
HYPRE_SStructGMRESSetStopCrit(HYPRE_SStructSolver solver,
                              HYPRE_Int           stop_crit);

HYPRE_Int
HYPRE_SStructGMRESSetPrecond(HYPRE_SStructSolver          solver,
                             HYPRE_PtrToSStructSolverFcn  precond,
                             HYPRE_PtrToSStructSolverFcn  precond_setup,
                             void                        *precond_solver);

HYPRE_Int
HYPRE_SStructGMRESSetLogging(HYPRE_SStructSolver solver,
                             HYPRE_Int           logging);

HYPRE_Int
HYPRE_SStructGMRESSetPrintLevel(HYPRE_SStructSolver solver,
                                HYPRE_Int           print_level);

HYPRE_Int
HYPRE_SStructGMRESGetNumIterations(HYPRE_SStructSolver  solver,
                                   HYPRE_Int           *num_iterations);

HYPRE_Int
HYPRE_SStructGMRESGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                               double              *norm);

HYPRE_Int
HYPRE_SStructGMRESGetResidual(HYPRE_SStructSolver   solver,
                              void                **residual);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct FlexGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{FlexGMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int
HYPRE_SStructFlexGMRESCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructFlexGMRESDestroy(HYPRE_SStructSolver solver);

HYPRE_Int
HYPRE_SStructFlexGMRESSetup(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructFlexGMRESSolve(HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructFlexGMRESSetTol(HYPRE_SStructSolver solver,
                             double              tol);

HYPRE_Int
HYPRE_SStructFlexGMRESSetAbsoluteTol(HYPRE_SStructSolver solver,
                                     double              tol);

/*
 * RE-VISIT
 **/
HYPRE_Int
HYPRE_SStructFlexGMRESSetMinIter(HYPRE_SStructSolver solver,
                                 HYPRE_Int           min_iter);

HYPRE_Int
HYPRE_SStructFlexGMRESSetMaxIter(HYPRE_SStructSolver solver,
                                 HYPRE_Int           max_iter);

HYPRE_Int
HYPRE_SStructFlexGMRESSetKDim(HYPRE_SStructSolver solver,
                              HYPRE_Int           k_dim);

HYPRE_Int
HYPRE_SStructFlexGMRESSetPrecond(HYPRE_SStructSolver          solver,
                                 HYPRE_PtrToSStructSolverFcn  precond,
                                 HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void                        *precond_solver);

HYPRE_Int
HYPRE_SStructFlexGMRESSetLogging(HYPRE_SStructSolver solver,
                                 HYPRE_Int           logging);

HYPRE_Int
HYPRE_SStructFlexGMRESSetPrintLevel(HYPRE_SStructSolver solver,
                                    HYPRE_Int           print_level);

HYPRE_Int
HYPRE_SStructFlexGMRESGetNumIterations(HYPRE_SStructSolver  solver,
                                       HYPRE_Int           *num_iterations);

HYPRE_Int
HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                   double              *norm);

HYPRE_Int
HYPRE_SStructFlexGMRESGetResidual(HYPRE_SStructSolver   solver,
                                  void                **residual);

HYPRE_Int
HYPRE_SStructFlexGMRESSetModifyPC(HYPRE_SStructSolver    solver,
                                  HYPRE_PtrToModifyPCFcn modify_pc);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct LGMRES Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{LGMRES Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int
HYPRE_SStructLGMRESCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructLGMRESDestroy(HYPRE_SStructSolver solver);

HYPRE_Int
HYPRE_SStructLGMRESSetup(HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x);
   
HYPRE_Int
HYPRE_SStructLGMRESSolve(HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructLGMRESSetTol(HYPRE_SStructSolver solver,
                          double              tol);


HYPRE_Int
HYPRE_SStructLGMRESSetAbsoluteTol(HYPRE_SStructSolver solver,
                                  double              tol);

/*
 * RE-VISIT
 **/
HYPRE_Int
HYPRE_SStructLGMRESSetMinIter(HYPRE_SStructSolver solver,
                              HYPRE_Int           min_iter);

HYPRE_Int
HYPRE_SStructLGMRESSetMaxIter(HYPRE_SStructSolver solver,
                              HYPRE_Int           max_iter);

HYPRE_Int
HYPRE_SStructLGMRESSetKDim(HYPRE_SStructSolver solver,
                           HYPRE_Int           k_dim);
HYPRE_Int
HYPRE_SStructLGMRESSetAugDim(HYPRE_SStructSolver solver,
                             HYPRE_Int           aug_dim);

HYPRE_Int
HYPRE_SStructLGMRESSetPrecond(HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void                        *precond_solver);

HYPRE_Int
HYPRE_SStructLGMRESSetLogging(HYPRE_SStructSolver solver,
                              HYPRE_Int           logging);

HYPRE_Int
HYPRE_SStructLGMRESSetPrintLevel(HYPRE_SStructSolver solver,
                                 HYPRE_Int           print_level);

HYPRE_Int
HYPRE_SStructLGMRESGetNumIterations(HYPRE_SStructSolver  solver,
                                    HYPRE_Int           *num_iterations);

HYPRE_Int
HYPRE_SStructLGMRESGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                double              *norm);

HYPRE_Int
HYPRE_SStructLGMRESGetResidual(HYPRE_SStructSolver   solver,
                               void                **residual);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name SStruct BiCGSTAB Solver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{BiCGSTAB Solver}.
 **/
/*@{*/

/**
 * Create a solver object.
 **/
HYPRE_Int
HYPRE_SStructBiCGSTABCreate(MPI_Comm             comm,
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
HYPRE_Int
HYPRE_SStructBiCGSTABDestroy(HYPRE_SStructSolver solver);

HYPRE_Int
HYPRE_SStructBiCGSTABSetup(HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructBiCGSTABSolve(HYPRE_SStructSolver solver,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector b,
                           HYPRE_SStructVector x);

HYPRE_Int
HYPRE_SStructBiCGSTABSetTol(HYPRE_SStructSolver solver,
                            double              tol);

HYPRE_Int
HYPRE_SStructBiCGSTABSetAbsoluteTol(HYPRE_SStructSolver solver,
                                    double              tol);
/*
 * RE-VISIT
 **/
HYPRE_Int
HYPRE_SStructBiCGSTABSetMinIter(HYPRE_SStructSolver solver,
                                HYPRE_Int           min_iter);

HYPRE_Int
HYPRE_SStructBiCGSTABSetMaxIter(HYPRE_SStructSolver solver,
                                HYPRE_Int           max_iter);

/*
 * RE-VISIT
 **/
HYPRE_Int
HYPRE_SStructBiCGSTABSetStopCrit(HYPRE_SStructSolver solver,
                                 HYPRE_Int           stop_crit);

HYPRE_Int
HYPRE_SStructBiCGSTABSetPrecond(HYPRE_SStructSolver          solver,
                                HYPRE_PtrToSStructSolverFcn  precond,
                                HYPRE_PtrToSStructSolverFcn  precond_setup,
                                void                        *precond_solver);

HYPRE_Int
HYPRE_SStructBiCGSTABSetLogging(HYPRE_SStructSolver solver,
                                HYPRE_Int           logging);

HYPRE_Int
HYPRE_SStructBiCGSTABSetPrintLevel(HYPRE_SStructSolver solver,
                                   HYPRE_Int           level);

HYPRE_Int
HYPRE_SStructBiCGSTABGetNumIterations(HYPRE_SStructSolver  solver,
                                      HYPRE_Int           *num_iterations);

HYPRE_Int
HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm(HYPRE_SStructSolver  solver,
                                                  double              *norm);

HYPRE_Int
HYPRE_SStructBiCGSTABGetResidual(HYPRE_SStructSolver   solver,
                                 void                **residual);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* These includes shouldn't be here. (RDF) */
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "_hypre_sstruct_mv.h"

/**
 * @name SStruct LOBPCG Eigensolver
 *
 * These routines should be used in conjunction with the generic interface in
 * \Ref{LOBPCG Eigensolver}.
 **/
/*@{*/

/**
  * Load interface interpreter.  Vector part loaded with hypre_SStructKrylov
  * functions and multivector part loaded with mv_TempMultiVector functions.
  **/
HYPRE_Int
HYPRE_SStructSetupInterpreter(mv_InterfaceInterpreter *i);

/**
  * Load Matvec interpreter with hypre_SStructKrylov functions.
  **/
HYPRE_Int
HYPRE_SStructSetupMatvec(HYPRE_MatvecFunctions *mv);

/* The next routines should not be here (lower-case prefix). (RDF) */

/*
 * Set hypre_SStructPVector to random values.
 **/
HYPRE_Int
hypre_SStructPVectorSetRandomValues(hypre_SStructPVector *pvector, HYPRE_Int seed);

/*
 * Set hypre_SStructVector to random values.
 **/
HYPRE_Int
hypre_SStructVectorSetRandomValues(hypre_SStructVector *vector, HYPRE_Int seed);

/*
 * Same as hypre_SStructVectorSetRandomValues except uses void pointer.
 **/
HYPRE_Int
hypre_SStructSetRandomValues(void *v, HYPRE_Int seed);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/*@}*/

#ifdef __cplusplus
}
#endif

#endif

