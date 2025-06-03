
/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * Header file of multiprecision function prototypes.
 * This is needed for mixed-precision algorithm development.
 *****************************************************************************/

#ifndef HYPRE_STRUCT_LS_MUP_HEADER
#define HYPRE_STRUCT_LS_MUP_HEADER

#if defined (HYPRE_MIXED_PRECISION)

HYPRE_Int hypre_StructCoarsen_flt  ( hypre_StructGrid *fgrid, hypre_Index index, hypre_Index stride,
                                HYPRE_Int prune, hypre_StructGrid **cgrid_ptr );
HYPRE_Int hypre_StructCoarsen_dbl  ( hypre_StructGrid *fgrid, hypre_Index index, hypre_Index stride,
                                HYPRE_Int prune, hypre_StructGrid **cgrid_ptr );
HYPRE_Int hypre_StructCoarsen_long_dbl  ( hypre_StructGrid *fgrid, hypre_Index index, hypre_Index stride,
                                HYPRE_Int prune, hypre_StructGrid **cgrid_ptr );
HYPRE_Int hypre_StructMapCoarseToFine_flt  ( hypre_Index cindex, hypre_Index index, hypre_Index stride,
                                        hypre_Index findex );
HYPRE_Int hypre_StructMapCoarseToFine_dbl  ( hypre_Index cindex, hypre_Index index, hypre_Index stride,
                                        hypre_Index findex );
HYPRE_Int hypre_StructMapCoarseToFine_long_dbl  ( hypre_Index cindex, hypre_Index index, hypre_Index stride,
                                        hypre_Index findex );
HYPRE_Int hypre_StructMapFineToCoarse_flt  ( hypre_Index findex, hypre_Index index, hypre_Index stride,
                                        hypre_Index cindex );
HYPRE_Int hypre_StructMapFineToCoarse_dbl  ( hypre_Index findex, hypre_Index index, hypre_Index stride,
                                        hypre_Index cindex );
HYPRE_Int hypre_StructMapFineToCoarse_long_dbl  ( hypre_Index findex, hypre_Index index, hypre_Index stride,
                                        hypre_Index cindex );
HYPRE_Int hypre_CyclicReduction_flt  ( void *cyc_red_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_CyclicReduction_dbl  ( void *cyc_red_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_CyclicReduction_long_dbl  ( void *cyc_red_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
void *hypre_CyclicReductionCreate_flt  ( MPI_Comm comm );
void *hypre_CyclicReductionCreate_dbl  ( MPI_Comm comm );
void *hypre_CyclicReductionCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_CyclicReductionDestroy_flt  ( void *cyc_red_vdata );
HYPRE_Int hypre_CyclicReductionDestroy_dbl  ( void *cyc_red_vdata );
HYPRE_Int hypre_CyclicReductionDestroy_long_dbl  ( void *cyc_red_vdata );
HYPRE_Int hypre_CyclicReductionSetBase_flt  ( void *cyc_red_vdata, hypre_Index base_index,
                                         hypre_Index base_stride );
HYPRE_Int hypre_CyclicReductionSetBase_dbl  ( void *cyc_red_vdata, hypre_Index base_index,
                                         hypre_Index base_stride );
HYPRE_Int hypre_CyclicReductionSetBase_long_dbl  ( void *cyc_red_vdata, hypre_Index base_index,
                                         hypre_Index base_stride );
HYPRE_Int hypre_CyclicReductionSetCDir_flt  ( void *cyc_red_vdata, HYPRE_Int cdir );
HYPRE_Int hypre_CyclicReductionSetCDir_dbl  ( void *cyc_red_vdata, HYPRE_Int cdir );
HYPRE_Int hypre_CyclicReductionSetCDir_long_dbl  ( void *cyc_red_vdata, HYPRE_Int cdir );
HYPRE_Int hypre_CyclicReductionSetMaxLevel_flt ( void   *cyc_red_vdata, HYPRE_Int   max_level  );
HYPRE_Int hypre_CyclicReductionSetMaxLevel_dbl ( void   *cyc_red_vdata, HYPRE_Int   max_level  );
HYPRE_Int hypre_CyclicReductionSetMaxLevel_long_dbl ( void   *cyc_red_vdata, HYPRE_Int   max_level  );
HYPRE_Int hypre_CyclicReductionSetup_flt  ( void *cyc_red_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_CyclicReductionSetup_dbl  ( void *cyc_red_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_CyclicReductionSetup_long_dbl  ( void *cyc_red_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
hypre_StructMatrix *hypre_CycRedCreateCoarseOp_flt  ( hypre_StructMatrix *A,
                                                 hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_CycRedCreateCoarseOp_dbl  ( hypre_StructMatrix *A,
                                                 hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_CycRedCreateCoarseOp_long_dbl  ( hypre_StructMatrix *A,
                                                 hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_CycRedSetupCoarseOp_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *Ac,
                                      hypre_Index cindex, hypre_Index cstride, HYPRE_Int cdir );
HYPRE_Int hypre_CycRedSetupCoarseOp_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *Ac,
                                      hypre_Index cindex, hypre_Index cstride, HYPRE_Int cdir );
HYPRE_Int hypre_CycRedSetupCoarseOp_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *Ac,
                                      hypre_Index cindex, hypre_Index cstride, HYPRE_Int cdir );
HYPRE_Int hypre_StructDiagScale_flt ( hypre_StructMatrix   *A, hypre_StructVector   *y, hypre_StructVector   *x );
HYPRE_Int hypre_StructDiagScale_dbl ( hypre_StructMatrix   *A, hypre_StructVector   *y, hypre_StructVector   *x );
HYPRE_Int hypre_StructDiagScale_long_dbl ( hypre_StructMatrix   *A, hypre_StructVector   *y, hypre_StructVector   *x );
void *hypre_HybridCreate_flt  ( MPI_Comm comm );
void *hypre_HybridCreate_dbl  ( MPI_Comm comm );
void *hypre_HybridCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_HybridDestroy_flt  ( void *hybrid_vdata );
HYPRE_Int hypre_HybridDestroy_dbl  ( void *hybrid_vdata );
HYPRE_Int hypre_HybridDestroy_long_dbl  ( void *hybrid_vdata );
HYPRE_Int hypre_HybridGetDSCGNumIterations_flt  ( void *hybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_HybridGetDSCGNumIterations_dbl  ( void *hybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_HybridGetDSCGNumIterations_long_dbl  ( void *hybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_HybridGetFinalRelativeResidualNorm_flt  ( void *hybrid_vdata,
                                                     hypre_float *final_rel_res_norm );
HYPRE_Int hypre_HybridGetFinalRelativeResidualNorm_dbl  ( void *hybrid_vdata,
                                                     hypre_double *final_rel_res_norm );
HYPRE_Int hypre_HybridGetFinalRelativeResidualNorm_long_dbl  ( void *hybrid_vdata,
                                                     hypre_long_double *final_rel_res_norm );
HYPRE_Int hypre_HybridGetNumIterations_flt  ( void *hybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_HybridGetNumIterations_dbl  ( void *hybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_HybridGetNumIterations_long_dbl  ( void *hybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_HybridGetPCGNumIterations_flt  ( void *hybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_HybridGetPCGNumIterations_dbl  ( void *hybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_HybridGetPCGNumIterations_long_dbl  ( void *hybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_HybridGetRecomputeResidual_flt ( void *hybrid_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_HybridGetRecomputeResidual_dbl ( void *hybrid_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_HybridGetRecomputeResidual_long_dbl ( void *hybrid_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_HybridGetRecomputeResidualP_flt ( void *hybrid_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_HybridGetRecomputeResidualP_dbl ( void *hybrid_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_HybridGetRecomputeResidualP_long_dbl ( void *hybrid_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_HybridSetConvergenceTol_flt  ( void *hybrid_vdata, hypre_float cf_tol );
HYPRE_Int hypre_HybridSetConvergenceTol_dbl  ( void *hybrid_vdata, hypre_double cf_tol );
HYPRE_Int hypre_HybridSetConvergenceTol_long_dbl  ( void *hybrid_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_HybridSetDSCGMaxIter_flt  ( void *hybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_HybridSetDSCGMaxIter_dbl  ( void *hybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_HybridSetDSCGMaxIter_long_dbl  ( void *hybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_HybridSetKDim_flt  ( void *hybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_HybridSetKDim_dbl  ( void *hybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_HybridSetKDim_long_dbl  ( void *hybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_HybridSetLogging_flt  ( void *hybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_HybridSetLogging_dbl  ( void *hybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_HybridSetLogging_long_dbl  ( void *hybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_HybridSetPCGAbsoluteTolFactor_flt  ( void *hybrid_vdata, hypre_float pcg_atolf );
HYPRE_Int hypre_HybridSetPCGAbsoluteTolFactor_dbl  ( void *hybrid_vdata, hypre_double pcg_atolf );
HYPRE_Int hypre_HybridSetPCGAbsoluteTolFactor_long_dbl  ( void *hybrid_vdata, hypre_long_double pcg_atolf );
HYPRE_Int hypre_HybridSetPCGMaxIter_flt  ( void *hybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_HybridSetPCGMaxIter_dbl  ( void *hybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_HybridSetPCGMaxIter_long_dbl  ( void *hybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_HybridSetPrecond_flt  ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                    void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_HybridSetPrecond_dbl  ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                    void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_HybridSetPrecond_long_dbl  ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                    void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_HybridSetPrintLevel_flt  ( void *hybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_HybridSetPrintLevel_dbl  ( void *hybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_HybridSetPrintLevel_long_dbl  ( void *hybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_HybridSetRecomputeResidual_flt ( void *hybrid_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_HybridSetRecomputeResidual_dbl ( void *hybrid_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_HybridSetRecomputeResidual_long_dbl ( void *hybrid_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_HybridSetRecomputeResidualP_flt ( void *hybrid_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_HybridSetRecomputeResidualP_dbl ( void *hybrid_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_HybridSetRecomputeResidualP_long_dbl ( void *hybrid_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_HybridSetRelChange_flt  ( void *hybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_HybridSetRelChange_dbl  ( void *hybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_HybridSetRelChange_long_dbl  ( void *hybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_HybridSetSolverType_flt  ( void *hybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_HybridSetSolverType_dbl  ( void *hybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_HybridSetSolverType_long_dbl  ( void *hybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_HybridSetStopCrit_flt  ( void *hybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_HybridSetStopCrit_dbl  ( void *hybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_HybridSetStopCrit_long_dbl  ( void *hybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_HybridSetTol_flt  ( void *hybrid_vdata, hypre_float tol );
HYPRE_Int hypre_HybridSetTol_dbl  ( void *hybrid_vdata, hypre_double tol );
HYPRE_Int hypre_HybridSetTol_long_dbl  ( void *hybrid_vdata, hypre_long_double tol );
HYPRE_Int hypre_HybridSetTwoNorm_flt  ( void *hybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_HybridSetTwoNorm_dbl  ( void *hybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_HybridSetTwoNorm_long_dbl  ( void *hybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_HybridSetup_flt  ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_HybridSetup_dbl  ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_HybridSetup_long_dbl  ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_HybridSolve_flt  ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_HybridSolve_dbl  ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_HybridSolve_long_dbl  ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int HYPRE_StructBiCGSTABCreate_flt (MPI_Comm            comm,
                                     HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructBiCGSTABCreate_dbl (MPI_Comm            comm,
                                     HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructBiCGSTABCreate_long_dbl (MPI_Comm            comm,
                                     HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructBiCGSTABDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructBiCGSTABDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructBiCGSTABDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                           hypre_float         *norm);
HYPRE_Int HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                           hypre_double         *norm);
HYPRE_Int HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                           hypre_long_double         *norm);
HYPRE_Int HYPRE_StructBiCGSTABGetNumIterations_flt (HYPRE_StructSolver  solver,
                                               HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructBiCGSTABGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                               HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructBiCGSTABGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                               HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructBiCGSTABGetResidual_flt ( HYPRE_StructSolver   solver,
                                           void               **residual);
HYPRE_Int HYPRE_StructBiCGSTABGetResidual_dbl ( HYPRE_StructSolver   solver,
                                           void               **residual);
HYPRE_Int HYPRE_StructBiCGSTABGetResidual_long_dbl ( HYPRE_StructSolver   solver,
                                           void               **residual);
HYPRE_Int HYPRE_StructBiCGSTABSetAbsoluteTol_flt (HYPRE_StructSolver solver,
                                             hypre_float         tol);
HYPRE_Int HYPRE_StructBiCGSTABSetAbsoluteTol_dbl (HYPRE_StructSolver solver,
                                             hypre_double         tol);
HYPRE_Int HYPRE_StructBiCGSTABSetAbsoluteTol_long_dbl (HYPRE_StructSolver solver,
                                             hypre_long_double         tol);
HYPRE_Int HYPRE_StructBiCGSTABSetLogging_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int          logging);
HYPRE_Int HYPRE_StructBiCGSTABSetLogging_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          logging);
HYPRE_Int HYPRE_StructBiCGSTABSetLogging_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          logging);
HYPRE_Int HYPRE_StructBiCGSTABSetMaxIter_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructBiCGSTABSetMaxIter_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructBiCGSTABSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructBiCGSTABSetPrecond_flt (HYPRE_StructSolver         solver,
                                         HYPRE_PtrToStructSolverFcn precond,
                                         HYPRE_PtrToStructSolverFcn precond_setup,
                                         HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructBiCGSTABSetPrecond_dbl (HYPRE_StructSolver         solver,
                                         HYPRE_PtrToStructSolverFcn precond,
                                         HYPRE_PtrToStructSolverFcn precond_setup,
                                         HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructBiCGSTABSetPrecond_long_dbl (HYPRE_StructSolver         solver,
                                         HYPRE_PtrToStructSolverFcn precond,
                                         HYPRE_PtrToStructSolverFcn precond_setup,
                                         HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructBiCGSTABSetPrintLevel_flt (HYPRE_StructSolver solver,
                                            HYPRE_Int          level);
HYPRE_Int HYPRE_StructBiCGSTABSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                            HYPRE_Int          level);
HYPRE_Int HYPRE_StructBiCGSTABSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                            HYPRE_Int          level);
HYPRE_Int HYPRE_StructBiCGSTABSetTol_flt (HYPRE_StructSolver solver,
                                     hypre_float         tol);
HYPRE_Int HYPRE_StructBiCGSTABSetTol_dbl (HYPRE_StructSolver solver,
                                     hypre_double         tol);
HYPRE_Int HYPRE_StructBiCGSTABSetTol_long_dbl (HYPRE_StructSolver solver,
                                     hypre_long_double         tol);
HYPRE_Int HYPRE_StructBiCGSTABSetup_flt (HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);
HYPRE_Int HYPRE_StructBiCGSTABSetup_dbl (HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);
HYPRE_Int HYPRE_StructBiCGSTABSetup_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);
HYPRE_Int HYPRE_StructBiCGSTABSolve_flt (HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);
HYPRE_Int HYPRE_StructBiCGSTABSolve_dbl (HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);
HYPRE_Int HYPRE_StructBiCGSTABSolve_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_StructMatrix A,
                                    HYPRE_StructVector b,
                                    HYPRE_StructVector x);
HYPRE_Int HYPRE_StructCycRedCreate_flt (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructCycRedCreate_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructCycRedCreate_long_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructCycRedDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructCycRedDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructCycRedDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructCycRedSetBase_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          ndim,
                                    HYPRE_Int         *base_index,
                                    HYPRE_Int         *base_stride);
HYPRE_Int HYPRE_StructCycRedSetBase_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          ndim,
                                    HYPRE_Int         *base_index,
                                    HYPRE_Int         *base_stride);
HYPRE_Int HYPRE_StructCycRedSetBase_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          ndim,
                                    HYPRE_Int         *base_index,
                                    HYPRE_Int         *base_stride);
HYPRE_Int HYPRE_StructCycRedSetTDim_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          tdim);
HYPRE_Int HYPRE_StructCycRedSetTDim_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          tdim);
HYPRE_Int HYPRE_StructCycRedSetTDim_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          tdim);
HYPRE_Int HYPRE_StructCycRedSetup_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructCycRedSetup_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructCycRedSetup_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructCycRedSolve_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructCycRedSolve_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructCycRedSolve_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructFlexGMRESCreate_flt (MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructFlexGMRESCreate_dbl (MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructFlexGMRESCreate_long_dbl (MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructFlexGMRESDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructFlexGMRESDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructFlexGMRESDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                            hypre_float         *norm);
HYPRE_Int HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                            hypre_double         *norm);
HYPRE_Int HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                            hypre_long_double         *norm);
HYPRE_Int HYPRE_StructFlexGMRESGetNumIterations_flt (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructFlexGMRESGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructFlexGMRESGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructFlexGMRESSetAbsoluteTol_flt (HYPRE_StructSolver solver,
                                              hypre_float         tol);
HYPRE_Int HYPRE_StructFlexGMRESSetAbsoluteTol_dbl (HYPRE_StructSolver solver,
                                              hypre_double         tol);
HYPRE_Int HYPRE_StructFlexGMRESSetAbsoluteTol_long_dbl (HYPRE_StructSolver solver,
                                              hypre_long_double         tol);
HYPRE_Int HYPRE_StructFlexGMRESSetKDim_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructFlexGMRESSetKDim_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructFlexGMRESSetKDim_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructFlexGMRESSetLogging_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);
HYPRE_Int HYPRE_StructFlexGMRESSetLogging_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);
HYPRE_Int HYPRE_StructFlexGMRESSetLogging_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);
HYPRE_Int HYPRE_StructFlexGMRESSetMaxIter_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructFlexGMRESSetMaxIter_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructFlexGMRESSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructFlexGMRESSetModifyPC_flt (HYPRE_StructSolver     solver,
                                           HYPRE_PtrToModifyPCFcn modify_pc);
HYPRE_Int HYPRE_StructFlexGMRESSetModifyPC_dbl (HYPRE_StructSolver     solver,
                                           HYPRE_PtrToModifyPCFcn modify_pc);
HYPRE_Int HYPRE_StructFlexGMRESSetModifyPC_long_dbl (HYPRE_StructSolver     solver,
                                           HYPRE_PtrToModifyPCFcn modify_pc);
HYPRE_Int HYPRE_StructFlexGMRESSetPrecond_flt (HYPRE_StructSolver         solver,
                                          HYPRE_PtrToStructSolverFcn precond,
                                          HYPRE_PtrToStructSolverFcn precond_setup,
                                          HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructFlexGMRESSetPrecond_dbl (HYPRE_StructSolver         solver,
                                          HYPRE_PtrToStructSolverFcn precond,
                                          HYPRE_PtrToStructSolverFcn precond_setup,
                                          HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructFlexGMRESSetPrecond_long_dbl (HYPRE_StructSolver         solver,
                                          HYPRE_PtrToStructSolverFcn precond,
                                          HYPRE_PtrToStructSolverFcn precond_setup,
                                          HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructFlexGMRESSetPrintLevel_flt (HYPRE_StructSolver solver,
                                             HYPRE_Int          level);
HYPRE_Int HYPRE_StructFlexGMRESSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                             HYPRE_Int          level);
HYPRE_Int HYPRE_StructFlexGMRESSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                             HYPRE_Int          level);
HYPRE_Int HYPRE_StructFlexGMRESSetTol_flt (HYPRE_StructSolver solver,
                                      hypre_float         tol);
HYPRE_Int HYPRE_StructFlexGMRESSetTol_dbl (HYPRE_StructSolver solver,
                                      hypre_double         tol);
HYPRE_Int HYPRE_StructFlexGMRESSetTol_long_dbl (HYPRE_StructSolver solver,
                                      hypre_long_double         tol);
HYPRE_Int HYPRE_StructFlexGMRESSetup_flt (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructFlexGMRESSetup_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructFlexGMRESSetup_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructFlexGMRESSolve_flt (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructFlexGMRESSolve_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructFlexGMRESSolve_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructGMRESCreate_flt (MPI_Comm            comm,
                                  HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructGMRESCreate_dbl (MPI_Comm            comm,
                                  HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructGMRESCreate_long_dbl (MPI_Comm            comm,
                                  HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructGMRESDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructGMRESDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructGMRESDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructGMRESGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                        hypre_float         *norm);
HYPRE_Int HYPRE_StructGMRESGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                        hypre_double         *norm);
HYPRE_Int HYPRE_StructGMRESGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                        hypre_long_double         *norm);
HYPRE_Int HYPRE_StructGMRESGetNumIterations_flt (HYPRE_StructSolver  solver,
                                            HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructGMRESGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                            HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructGMRESGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                            HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructGMRESSetAbsoluteTol_flt (HYPRE_StructSolver solver,
                                          hypre_float         tol);
HYPRE_Int HYPRE_StructGMRESSetAbsoluteTol_dbl (HYPRE_StructSolver solver,
                                          hypre_double         tol);
HYPRE_Int HYPRE_StructGMRESSetAbsoluteTol_long_dbl (HYPRE_StructSolver solver,
                                          hypre_long_double         tol);
HYPRE_Int HYPRE_StructGMRESSetKDim_flt (HYPRE_StructSolver solver,
                                   HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructGMRESSetKDim_dbl (HYPRE_StructSolver solver,
                                   HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructGMRESSetKDim_long_dbl (HYPRE_StructSolver solver,
                                   HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructGMRESSetLogging_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int          logging);
HYPRE_Int HYPRE_StructGMRESSetLogging_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          logging);
HYPRE_Int HYPRE_StructGMRESSetLogging_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          logging);
HYPRE_Int HYPRE_StructGMRESSetMaxIter_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructGMRESSetMaxIter_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructGMRESSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructGMRESSetPrecond_flt (HYPRE_StructSolver         solver,
                                      HYPRE_PtrToStructSolverFcn precond,
                                      HYPRE_PtrToStructSolverFcn precond_setup,
                                      HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructGMRESSetPrecond_dbl (HYPRE_StructSolver         solver,
                                      HYPRE_PtrToStructSolverFcn precond,
                                      HYPRE_PtrToStructSolverFcn precond_setup,
                                      HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructGMRESSetPrecond_long_dbl (HYPRE_StructSolver         solver,
                                      HYPRE_PtrToStructSolverFcn precond,
                                      HYPRE_PtrToStructSolverFcn precond_setup,
                                      HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructGMRESSetPrintLevel_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int          level);
HYPRE_Int HYPRE_StructGMRESSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          level);
HYPRE_Int HYPRE_StructGMRESSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          level);
HYPRE_Int HYPRE_StructGMRESSetTol_flt (HYPRE_StructSolver solver,
                                  hypre_float         tol);
HYPRE_Int HYPRE_StructGMRESSetTol_dbl (HYPRE_StructSolver solver,
                                  hypre_double         tol);
HYPRE_Int HYPRE_StructGMRESSetTol_long_dbl (HYPRE_StructSolver solver,
                                  hypre_long_double         tol);
HYPRE_Int HYPRE_StructGMRESSetup_flt (HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);
HYPRE_Int HYPRE_StructGMRESSetup_dbl (HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);
HYPRE_Int HYPRE_StructGMRESSetup_long_dbl (HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);
HYPRE_Int HYPRE_StructGMRESSolve_flt (HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);
HYPRE_Int HYPRE_StructGMRESSolve_dbl (HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);
HYPRE_Int HYPRE_StructGMRESSolve_long_dbl (HYPRE_StructSolver solver,
                                 HYPRE_StructMatrix A,
                                 HYPRE_StructVector b,
                                 HYPRE_StructVector x);
HYPRE_Int HYPRE_StructHybridCreate_flt (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructHybridCreate_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructHybridCreate_long_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructHybridDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructHybridDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructHybridDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructHybridGetDSCGNumIterations_flt (HYPRE_StructSolver  solver,
                                                 HYPRE_Int          *ds_num_its);
HYPRE_Int HYPRE_StructHybridGetDSCGNumIterations_dbl (HYPRE_StructSolver  solver,
                                                 HYPRE_Int          *ds_num_its);
HYPRE_Int HYPRE_StructHybridGetDSCGNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                                 HYPRE_Int          *ds_num_its);
HYPRE_Int HYPRE_StructHybridGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                         hypre_float         *norm);
HYPRE_Int HYPRE_StructHybridGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                         hypre_double         *norm);
HYPRE_Int HYPRE_StructHybridGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                         hypre_long_double         *norm);
HYPRE_Int HYPRE_StructHybridGetNumIterations_flt (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_its);
HYPRE_Int HYPRE_StructHybridGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_its);
HYPRE_Int HYPRE_StructHybridGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_its);
HYPRE_Int HYPRE_StructHybridGetPCGNumIterations_flt (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *pre_num_its);
HYPRE_Int HYPRE_StructHybridGetPCGNumIterations_dbl (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *pre_num_its);
HYPRE_Int HYPRE_StructHybridGetPCGNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *pre_num_its);
HYPRE_Int HYPRE_StructHybridGetRecomputeResidual_flt( HYPRE_StructSolver  solver,
                                        HYPRE_Int          *recompute_residual );
HYPRE_Int HYPRE_StructHybridGetRecomputeResidual_dbl( HYPRE_StructSolver  solver,
                                        HYPRE_Int          *recompute_residual );
HYPRE_Int HYPRE_StructHybridGetRecomputeResidual_long_dbl( HYPRE_StructSolver  solver,
                                        HYPRE_Int          *recompute_residual );
HYPRE_Int HYPRE_StructHybridGetRecomputeResidualP_flt( HYPRE_StructSolver  solver,
                                         HYPRE_Int          *recompute_residual_p );
HYPRE_Int HYPRE_StructHybridGetRecomputeResidualP_dbl( HYPRE_StructSolver  solver,
                                         HYPRE_Int          *recompute_residual_p );
HYPRE_Int HYPRE_StructHybridGetRecomputeResidualP_long_dbl( HYPRE_StructSolver  solver,
                                         HYPRE_Int          *recompute_residual_p );
HYPRE_Int HYPRE_StructHybridSetConvergenceTol_flt (HYPRE_StructSolver solver,
                                              hypre_float         cf_tol);
HYPRE_Int HYPRE_StructHybridSetConvergenceTol_dbl (HYPRE_StructSolver solver,
                                              hypre_double         cf_tol);
HYPRE_Int HYPRE_StructHybridSetConvergenceTol_long_dbl (HYPRE_StructSolver solver,
                                              hypre_long_double         cf_tol);
HYPRE_Int HYPRE_StructHybridSetDSCGMaxIter_flt (HYPRE_StructSolver solver,
                                           HYPRE_Int          ds_max_its);
HYPRE_Int HYPRE_StructHybridSetDSCGMaxIter_dbl (HYPRE_StructSolver solver,
                                           HYPRE_Int          ds_max_its);
HYPRE_Int HYPRE_StructHybridSetDSCGMaxIter_long_dbl (HYPRE_StructSolver solver,
                                           HYPRE_Int          ds_max_its);
HYPRE_Int HYPRE_StructHybridSetKDim_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructHybridSetKDim_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructHybridSetKDim_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructHybridSetLogging_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);
HYPRE_Int HYPRE_StructHybridSetLogging_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);
HYPRE_Int HYPRE_StructHybridSetLogging_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);
HYPRE_Int HYPRE_StructHybridSetPCGAbsoluteTolFactor_flt (HYPRE_StructSolver solver,
                                                    hypre_float pcg_atolf );
HYPRE_Int HYPRE_StructHybridSetPCGAbsoluteTolFactor_dbl (HYPRE_StructSolver solver,
                                                    hypre_double pcg_atolf );
HYPRE_Int HYPRE_StructHybridSetPCGAbsoluteTolFactor_long_dbl (HYPRE_StructSolver solver,
                                                    hypre_long_double pcg_atolf );
HYPRE_Int HYPRE_StructHybridSetPCGMaxIter_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          pre_max_its);
HYPRE_Int HYPRE_StructHybridSetPCGMaxIter_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          pre_max_its);
HYPRE_Int HYPRE_StructHybridSetPCGMaxIter_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          pre_max_its);
HYPRE_Int HYPRE_StructHybridSetPrecond_flt (HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructHybridSetPrecond_dbl (HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructHybridSetPrecond_long_dbl (HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructHybridSetPrintLevel_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructHybridSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructHybridSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructHybridSetRecomputeResidual_flt ( HYPRE_StructSolver  solver,
                                        HYPRE_Int           recompute_residual );
HYPRE_Int HYPRE_StructHybridSetRecomputeResidual_dbl( HYPRE_StructSolver  solver,
                                        HYPRE_Int           recompute_residual );
HYPRE_Int HYPRE_StructHybridSetRecomputeResidual_long_dbl( HYPRE_StructSolver  solver,
                                        HYPRE_Int           recompute_residual );
HYPRE_Int HYPRE_StructHybridSetRecomputeResidualP_flt( HYPRE_StructSolver  solver,
                                         HYPRE_Int           recompute_residual_p );
HYPRE_Int HYPRE_StructHybridSetRecomputeResidualP_dbl( HYPRE_StructSolver  solver,
                                         HYPRE_Int           recompute_residual_p );
HYPRE_Int HYPRE_StructHybridSetRecomputeResidualP_long_dbl ( HYPRE_StructSolver  solver,
                                         HYPRE_Int           recompute_residual_p );
HYPRE_Int HYPRE_StructHybridSetRelChange_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructHybridSetRelChange_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructHybridSetRelChange_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructHybridSetSolverType_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          solver_type);
HYPRE_Int HYPRE_StructHybridSetSolverType_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          solver_type);
HYPRE_Int HYPRE_StructHybridSetSolverType_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          solver_type);
HYPRE_Int HYPRE_StructHybridSetStopCrit_flt (HYPRE_StructSolver solver,
                                        HYPRE_Int          stop_crit);
HYPRE_Int HYPRE_StructHybridSetStopCrit_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int          stop_crit);
HYPRE_Int HYPRE_StructHybridSetStopCrit_long_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int          stop_crit);
HYPRE_Int HYPRE_StructHybridSetTol_flt (HYPRE_StructSolver solver,
                                   hypre_float         tol);
HYPRE_Int HYPRE_StructHybridSetTol_dbl (HYPRE_StructSolver solver,
                                   hypre_double         tol);
HYPRE_Int HYPRE_StructHybridSetTol_long_dbl (HYPRE_StructSolver solver,
                                   hypre_long_double         tol);
HYPRE_Int HYPRE_StructHybridSetTwoNorm_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          two_norm);
HYPRE_Int HYPRE_StructHybridSetTwoNorm_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          two_norm);
HYPRE_Int HYPRE_StructHybridSetTwoNorm_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          two_norm);
HYPRE_Int HYPRE_StructHybridSetup_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructHybridSetup_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructHybridSetup_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructHybridSolve_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructHybridSolve_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructHybridSolve_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructJacobiCreate_flt (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructJacobiCreate_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructJacobiCreate_long_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructJacobiDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver, HYPRE_Real *norm);
HYPRE_Int HYPRE_StructJacobiGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver, HYPRE_Real *norm);
HYPRE_Int HYPRE_StructJacobiGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver, HYPRE_Real *norm);
HYPRE_Int HYPRE_StructPFMGCreate_flt(MPI_Comm            comm,
                                 HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPFMGCreate_dbl(MPI_Comm            comm,
                                 HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPFMGCreate(MPI_Comm            comm,
                                 HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructJacobiGetMaxIter_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int *max_iter );
HYPRE_Int HYPRE_StructJacobiGetMaxIter_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *max_iter );
HYPRE_Int HYPRE_StructJacobiGetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *max_iter );
HYPRE_Int HYPRE_StructJacobiGetNumIterations_flt (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructJacobiGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructJacobiGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructJacobiGetTol_flt (HYPRE_StructSolver solver,
                                   hypre_float *tol );
HYPRE_Int HYPRE_StructJacobiGetTol_dbl (HYPRE_StructSolver solver,
                                   hypre_double *tol );
HYPRE_Int HYPRE_StructJacobiGetTol_long_dbl (HYPRE_StructSolver solver,
                                   hypre_long_double *tol );
HYPRE_Int HYPRE_StructJacobiGetZeroGuess_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int *zeroguess );
HYPRE_Int HYPRE_StructJacobiGetZeroGuess_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int *zeroguess );
HYPRE_Int HYPRE_StructJacobiGetZeroGuess_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int *zeroguess );
HYPRE_Int HYPRE_StructJacobiSetMaxIter_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructJacobiSetMaxIter_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructJacobiSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructJacobiSetNonZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiSetNonZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiSetNonZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiSetTol_flt (HYPRE_StructSolver solver,
                                   hypre_float         tol);
HYPRE_Int HYPRE_StructJacobiSetTol_dbl (HYPRE_StructSolver solver,
                                   hypre_double         tol);
HYPRE_Int HYPRE_StructJacobiSetTol_long_dbl (HYPRE_StructSolver solver,
                                   hypre_long_double         tol);
HYPRE_Int HYPRE_StructJacobiSetup_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructJacobiSetup_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructJacobiSetup_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructJacobiSetZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiSetZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiSetZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructJacobiSolve_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructJacobiSolve_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructJacobiSolve_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructLGMRESCreate_flt (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructLGMRESCreate_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructLGMRESCreate_long_dbl (MPI_Comm            comm,
                                   HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructLGMRESDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructLGMRESDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructLGMRESDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructLGMRESGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                         hypre_float         *norm);
HYPRE_Int HYPRE_StructLGMRESGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                         hypre_double         *norm);
HYPRE_Int HYPRE_StructLGMRESGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                         hypre_long_double         *norm);
HYPRE_Int HYPRE_StructLGMRESGetNumIterations_flt (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructLGMRESGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructLGMRESGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                             HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructLGMRESSetAbsoluteTol_flt (HYPRE_StructSolver solver,
                                           hypre_float         tol);
HYPRE_Int HYPRE_StructLGMRESSetAbsoluteTol_dbl (HYPRE_StructSolver solver,
                                           hypre_double         tol);
HYPRE_Int HYPRE_StructLGMRESSetAbsoluteTol_long_dbl (HYPRE_StructSolver solver,
                                           hypre_long_double         tol);
HYPRE_Int HYPRE_StructLGMRESSetAugDim_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int          aug_dim);
HYPRE_Int HYPRE_StructLGMRESSetAugDim_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          aug_dim);
HYPRE_Int HYPRE_StructLGMRESSetAugDim_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          aug_dim);
HYPRE_Int HYPRE_StructLGMRESSetKDim_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructLGMRESSetKDim_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructLGMRESSetKDim_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          k_dim);
HYPRE_Int HYPRE_StructLGMRESSetLogging_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);
HYPRE_Int HYPRE_StructLGMRESSetLogging_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);
HYPRE_Int HYPRE_StructLGMRESSetLogging_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          logging);
HYPRE_Int HYPRE_StructLGMRESSetMaxIter_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructLGMRESSetMaxIter_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructLGMRESSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructLGMRESSetPrecond_flt (HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructLGMRESSetPrecond_dbl (HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructLGMRESSetPrecond_long_dbl (HYPRE_StructSolver         solver,
                                       HYPRE_PtrToStructSolverFcn precond,
                                       HYPRE_PtrToStructSolverFcn precond_setup,
                                       HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructLGMRESSetPrintLevel_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          level);
HYPRE_Int HYPRE_StructLGMRESSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          level);
HYPRE_Int HYPRE_StructLGMRESSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          level);
HYPRE_Int HYPRE_StructLGMRESSetTol_flt (HYPRE_StructSolver solver,
                                   hypre_float         tol);
HYPRE_Int HYPRE_StructLGMRESSetTol_dbl (HYPRE_StructSolver solver,
                                   hypre_double         tol);
HYPRE_Int HYPRE_StructLGMRESSetTol_long_dbl (HYPRE_StructSolver solver,
                                   hypre_long_double         tol);
HYPRE_Int HYPRE_StructLGMRESSetup_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructLGMRESSetup_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructLGMRESSetup_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructLGMRESSolve_flt (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructLGMRESSolve_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructLGMRESSolve_long_dbl (HYPRE_StructSolver solver,
                                  HYPRE_StructMatrix A,
                                  HYPRE_StructVector b,
                                  HYPRE_StructVector x);
HYPRE_Int HYPRE_StructDiagScale_flt (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix HA,
                                HYPRE_StructVector Hy,
                                HYPRE_StructVector Hx);
HYPRE_Int HYPRE_StructDiagScale_dbl (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix HA,
                                HYPRE_StructVector Hy,
                                HYPRE_StructVector Hx);
HYPRE_Int HYPRE_StructDiagScale_long_dbl (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix HA,
                                HYPRE_StructVector Hy,
                                HYPRE_StructVector Hx);
HYPRE_Int HYPRE_StructDiagScaleSetup_flt (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector y,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructDiagScaleSetup_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector y,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructDiagScaleSetup_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector y,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPCGCreate_flt (MPI_Comm            comm,
                                HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPCGCreate_dbl (MPI_Comm            comm,
                                HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPCGCreate_long_dbl (MPI_Comm            comm,
                                HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPCGDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPCGDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPCGDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPCGGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                      hypre_float         *norm);
HYPRE_Int HYPRE_StructPCGGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                      hypre_double         *norm);
HYPRE_Int HYPRE_StructPCGGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                      hypre_long_double         *norm);
HYPRE_Int HYPRE_StructPCGGetNumIterations_flt (HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructPCGGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructPCGGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructPCGSetAbsoluteTol_flt (HYPRE_StructSolver solver,
                                        hypre_float         tol);
HYPRE_Int HYPRE_StructPCGSetAbsoluteTol_dbl (HYPRE_StructSolver solver,
                                        hypre_double         tol);
HYPRE_Int HYPRE_StructPCGSetAbsoluteTol_long_dbl (HYPRE_StructSolver solver,
                                        hypre_long_double         tol);
HYPRE_Int HYPRE_StructPCGSetLogging_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);
HYPRE_Int HYPRE_StructPCGSetLogging_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);
HYPRE_Int HYPRE_StructPCGSetLogging_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);
HYPRE_Int HYPRE_StructPCGSetMaxIter_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructPCGSetMaxIter_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructPCGSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructPCGSetPrecond_flt (HYPRE_StructSolver         solver,
                                    HYPRE_PtrToStructSolverFcn precond,
                                    HYPRE_PtrToStructSolverFcn precond_setup,
                                    HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructPCGSetPrecond_dbl (HYPRE_StructSolver         solver,
                                    HYPRE_PtrToStructSolverFcn precond,
                                    HYPRE_PtrToStructSolverFcn precond_setup,
                                    HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructPCGSetPrecond_long_dbl (HYPRE_StructSolver         solver,
                                    HYPRE_PtrToStructSolverFcn precond,
                                    HYPRE_PtrToStructSolverFcn precond_setup,
                                    HYPRE_StructSolver         precond_solver);
HYPRE_Int HYPRE_StructPCGSetPrintLevel_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          level);
HYPRE_Int HYPRE_StructPCGSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          level);
HYPRE_Int HYPRE_StructPCGSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          level);
HYPRE_Int HYPRE_StructPCGSetRelChange_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructPCGSetRelChange_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructPCGSetRelChange_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructPCGSetTol_flt (HYPRE_StructSolver solver,
                                hypre_float         tol);
HYPRE_Int HYPRE_StructPCGSetTol_dbl (HYPRE_StructSolver solver,
                                hypre_double         tol);
HYPRE_Int HYPRE_StructPCGSetTol_long_dbl (HYPRE_StructSolver solver,
                                hypre_long_double         tol);
HYPRE_Int HYPRE_StructPCGSetTwoNorm_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          two_norm);
HYPRE_Int HYPRE_StructPCGSetTwoNorm_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          two_norm);
HYPRE_Int HYPRE_StructPCGSetTwoNorm_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          two_norm);
HYPRE_Int HYPRE_StructPCGSetup_flt (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPCGSetup_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPCGSetup_long_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPCGSolve_flt (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPCGSolve_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPCGSolve_long_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPFMGCreate_flt (MPI_Comm            comm,
                                 HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPFMGCreate_dbl (MPI_Comm            comm,
                                 HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPFMGCreate_long_dbl (MPI_Comm            comm,
                                 HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructPFMGDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                       hypre_float         *norm);
HYPRE_Int HYPRE_StructPFMGGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                       hypre_double         *norm);
HYPRE_Int HYPRE_StructPFMGGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                       hypre_long_double         *norm);
HYPRE_Int HYPRE_StructPFMGGetJacobiWeight_flt (HYPRE_StructSolver solver,
                                          hypre_float        *weight);
HYPRE_Int HYPRE_StructPFMGGetJacobiWeight_dbl (HYPRE_StructSolver solver,
                                          hypre_double        *weight);
HYPRE_Int HYPRE_StructPFMGGetJacobiWeight_long_dbl (HYPRE_StructSolver solver,
                                          hypre_long_double        *weight);
HYPRE_Int HYPRE_StructPFMGGetLogging_flt (HYPRE_StructSolver solver,
                                     HYPRE_Int *logging);
HYPRE_Int HYPRE_StructPFMGGetLogging_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int *logging);
HYPRE_Int HYPRE_StructPFMGGetLogging_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int *logging);
HYPRE_Int HYPRE_StructPFMGGetMaxIter_flt (HYPRE_StructSolver solver,
                                     HYPRE_Int *max_iter);
HYPRE_Int HYPRE_StructPFMGGetMaxIter_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int *max_iter);
HYPRE_Int HYPRE_StructPFMGGetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int *max_iter);
HYPRE_Int HYPRE_StructPFMGGetMaxLevels_flt  (HYPRE_StructSolver solver,
                                        HYPRE_Int *max_levels );
HYPRE_Int HYPRE_StructPFMGGetMaxLevels_dbl  (HYPRE_StructSolver solver,
                                        HYPRE_Int *max_levels );
HYPRE_Int HYPRE_StructPFMGGetMaxLevels_long_dbl  (HYPRE_StructSolver solver,
                                        HYPRE_Int *max_levels );
HYPRE_Int HYPRE_StructPFMGGetNumIterations_flt (HYPRE_StructSolver  solver,
                                           HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructPFMGGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                           HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructPFMGGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                           HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructPFMGGetNumPostRelax_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int *num_post_relax);
HYPRE_Int HYPRE_StructPFMGGetNumPostRelax_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int *num_post_relax);
HYPRE_Int HYPRE_StructPFMGGetNumPostRelax_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int *num_post_relax);
HYPRE_Int HYPRE_StructPFMGGetNumPreRelax_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int *num_pre_relax);
HYPRE_Int HYPRE_StructPFMGGetNumPreRelax_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int *num_pre_relax);
HYPRE_Int HYPRE_StructPFMGGetNumPreRelax_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int *num_pre_relax);
HYPRE_Int HYPRE_StructPFMGGetPrintLevel_flt (HYPRE_StructSolver solver,
                                        HYPRE_Int *print_level);
HYPRE_Int HYPRE_StructPFMGGetPrintLevel_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int *print_level);
HYPRE_Int HYPRE_StructPFMGGetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int *print_level);
HYPRE_Int HYPRE_StructPFMGGetRAPType_flt (HYPRE_StructSolver solver,
                                     HYPRE_Int *rap_type );
HYPRE_Int HYPRE_StructPFMGGetRAPType_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int *rap_type );
HYPRE_Int HYPRE_StructPFMGGetRAPType_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int *rap_type );
HYPRE_Int HYPRE_StructPFMGGetRelaxType_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int *relax_type);
HYPRE_Int HYPRE_StructPFMGGetRelaxType_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *relax_type);
HYPRE_Int HYPRE_StructPFMGGetRelaxType_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *relax_type);
HYPRE_Int HYPRE_StructPFMGGetRelChange_flt  (HYPRE_StructSolver solver,
                                        HYPRE_Int *rel_change);
HYPRE_Int HYPRE_StructPFMGGetRelChange_dbl  (HYPRE_StructSolver solver,
                                        HYPRE_Int *rel_change);
HYPRE_Int HYPRE_StructPFMGGetRelChange_long_dbl  (HYPRE_StructSolver solver,
                                        HYPRE_Int *rel_change);
HYPRE_Int HYPRE_StructPFMGGetSkipRelax_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int *skip_relax);
HYPRE_Int HYPRE_StructPFMGGetSkipRelax_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *skip_relax);
HYPRE_Int HYPRE_StructPFMGGetSkipRelax_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *skip_relax);
HYPRE_Int HYPRE_StructPFMGGetTol_flt  (HYPRE_StructSolver solver,
                                  hypre_float *tol);
HYPRE_Int HYPRE_StructPFMGGetTol_dbl  (HYPRE_StructSolver solver,
                                  hypre_double *tol);
HYPRE_Int HYPRE_StructPFMGGetTol_long_dbl  (HYPRE_StructSolver solver,
                                  hypre_long_double *tol);
HYPRE_Int HYPRE_StructPFMGGetZeroGuess_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int *zeroguess);
HYPRE_Int HYPRE_StructPFMGGetZeroGuess_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *zeroguess);
HYPRE_Int HYPRE_StructPFMGGetZeroGuess_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *zeroguess);
HYPRE_Int HYPRE_StructPFMGSetDxyz_flt (HYPRE_StructSolver  solver,
                                  hypre_float         *dxyz);
HYPRE_Int HYPRE_StructPFMGSetDxyz_dbl (HYPRE_StructSolver  solver,
                                  hypre_double         *dxyz);
HYPRE_Int HYPRE_StructPFMGSetDxyz_long_dbl (HYPRE_StructSolver  solver,
                                  hypre_long_double         *dxyz);
HYPRE_Int HYPRE_StructPFMGSetJacobiWeight_flt (HYPRE_StructSolver solver,
                                          hypre_float         weight);
HYPRE_Int HYPRE_StructPFMGSetJacobiWeight_dbl (HYPRE_StructSolver solver,
                                          hypre_double         weight);
HYPRE_Int HYPRE_StructPFMGSetJacobiWeight_long_dbl (HYPRE_StructSolver solver,
                                          hypre_long_double         weight);
HYPRE_Int HYPRE_StructPFMGSetLogging_flt (HYPRE_StructSolver solver,
                                     HYPRE_Int          logging);
HYPRE_Int HYPRE_StructPFMGSetLogging_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int          logging);
HYPRE_Int HYPRE_StructPFMGSetLogging_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int          logging);
HYPRE_Int HYPRE_StructPFMGSetMaxIter_flt (HYPRE_StructSolver solver,
                                     HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructPFMGSetMaxIter_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructPFMGSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructPFMGSetMaxLevels_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_levels);
HYPRE_Int HYPRE_StructPFMGSetMaxLevels_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_levels);
HYPRE_Int HYPRE_StructPFMGSetMaxLevels_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          max_levels);
HYPRE_Int HYPRE_StructPFMGSetNonZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGSetNonZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGSetNonZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGSetNumPostRelax_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructPFMGSetNumPostRelax_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructPFMGSetNumPostRelax_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructPFMGSetNumPreRelax_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructPFMGSetNumPreRelax_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructPFMGSetNumPreRelax_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructPFMGSetPrintLevel_flt (HYPRE_StructSolver solver,
                                        HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructPFMGSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructPFMGSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructPFMGSetRAPType_flt (HYPRE_StructSolver solver,
                                     HYPRE_Int          rap_type);
HYPRE_Int HYPRE_StructPFMGSetRAPType_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int          rap_type);
HYPRE_Int HYPRE_StructPFMGSetRAPType_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_Int          rap_type);
HYPRE_Int HYPRE_StructPFMGSetRelaxType_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          relax_type);
HYPRE_Int HYPRE_StructPFMGSetRelaxType_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          relax_type);
HYPRE_Int HYPRE_StructPFMGSetRelaxType_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          relax_type);
HYPRE_Int HYPRE_StructPFMGSetRelChange_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructPFMGSetRelChange_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructPFMGSetRelChange_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructPFMGSetSkipRelax_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          skip_relax);
HYPRE_Int HYPRE_StructPFMGSetSkipRelax_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          skip_relax);
HYPRE_Int HYPRE_StructPFMGSetSkipRelax_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          skip_relax);
HYPRE_Int HYPRE_StructPFMGSetTol_flt (HYPRE_StructSolver solver,
                                 hypre_float         tol);
HYPRE_Int HYPRE_StructPFMGSetTol_dbl (HYPRE_StructSolver solver,
                                 hypre_double         tol);
HYPRE_Int HYPRE_StructPFMGSetTol_long_dbl (HYPRE_StructSolver solver,
                                 hypre_long_double         tol);
HYPRE_Int HYPRE_StructPFMGSetup_flt (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPFMGSetup_dbl (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPFMGSetup_long_dbl (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPFMGSetZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGSetZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGSetZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructPFMGSolve_flt (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPFMGSolve_dbl (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);
HYPRE_Int HYPRE_StructPFMGSolve_long_dbl (HYPRE_StructSolver solver,
                                HYPRE_StructMatrix A,
                                HYPRE_StructVector b,
                                HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSMGCreate_flt (MPI_Comm            comm,
                                HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructSMGCreate_dbl (MPI_Comm            comm,
                                HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructSMGCreate_long_dbl (MPI_Comm            comm,
                                HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructSMGDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                      hypre_float         *norm);
HYPRE_Int HYPRE_StructSMGGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                      hypre_double         *norm);
HYPRE_Int HYPRE_StructSMGGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                      hypre_long_double         *norm);
HYPRE_Int HYPRE_StructSMGGetLogging_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int *logging);
HYPRE_Int HYPRE_StructSMGGetLogging_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int *logging);
HYPRE_Int HYPRE_StructSMGGetLogging_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int *logging);
HYPRE_Int HYPRE_StructSMGGetMaxIter_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int *max_iter);
HYPRE_Int HYPRE_StructSMGGetMaxIter_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int *max_iter);
HYPRE_Int HYPRE_StructSMGGetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int *max_iter);
HYPRE_Int HYPRE_StructSMGGetMemoryUse_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int *memory_use);
HYPRE_Int HYPRE_StructSMGGetMemoryUse_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int *memory_use);
HYPRE_Int HYPRE_StructSMGGetMemoryUse_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int *memory_use);
HYPRE_Int HYPRE_StructSMGGetNumIterations_flt (HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructSMGGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructSMGGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                          HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructSMGGetNumPostRelax_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int *num_post_relax);
HYPRE_Int HYPRE_StructSMGGetNumPostRelax_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int *num_post_relax);
HYPRE_Int HYPRE_StructSMGGetNumPostRelax_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int *num_post_relax);
HYPRE_Int HYPRE_StructSMGGetNumPreRelax_flt (HYPRE_StructSolver solver,
                                        HYPRE_Int *num_pre_relax);
HYPRE_Int HYPRE_StructSMGGetNumPreRelax_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int *num_pre_relax);
HYPRE_Int HYPRE_StructSMGGetNumPreRelax_long_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int *num_pre_relax);
HYPRE_Int HYPRE_StructSMGGetPrintLevel_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int *print_level);
HYPRE_Int HYPRE_StructSMGGetPrintLevel_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *print_level);
HYPRE_Int HYPRE_StructSMGGetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int *print_level);
HYPRE_Int HYPRE_StructSMGGetRelChange_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int *rel_change);
HYPRE_Int HYPRE_StructSMGGetRelChange_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int *rel_change);
HYPRE_Int HYPRE_StructSMGGetRelChange_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int *rel_change);
HYPRE_Int HYPRE_StructSMGGetTol_flt (HYPRE_StructSolver solver,
                                hypre_float *tol);
HYPRE_Int HYPRE_StructSMGGetTol_dbl (HYPRE_StructSolver solver,
                                hypre_double *tol);
HYPRE_Int HYPRE_StructSMGGetTol_long_dbl (HYPRE_StructSolver solver,
                                hypre_long_double *tol);
HYPRE_Int HYPRE_StructSMGGetZeroGuess_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int *zeroguess);
HYPRE_Int HYPRE_StructSMGGetZeroGuess_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int *zeroguess);
HYPRE_Int HYPRE_StructSMGGetZeroGuess_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int *zeroguess);
HYPRE_Int HYPRE_StructSMGSetLogging_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);
HYPRE_Int HYPRE_StructSMGSetLogging_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);
HYPRE_Int HYPRE_StructSMGSetLogging_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          logging);
HYPRE_Int HYPRE_StructSMGSetMaxIter_flt (HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructSMGSetMaxIter_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructSMGSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                    HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructSMGSetMemoryUse_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int          memory_use);
HYPRE_Int HYPRE_StructSMGSetMemoryUse_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          memory_use);
HYPRE_Int HYPRE_StructSMGSetMemoryUse_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          memory_use);
HYPRE_Int HYPRE_StructSMGSetNonZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGSetNonZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGSetNonZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGSetNumPostRelax_flt (HYPRE_StructSolver solver,
                                         HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructSMGSetNumPostRelax_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructSMGSetNumPostRelax_long_dbl (HYPRE_StructSolver solver,
                                         HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructSMGSetNumPreRelax_flt (HYPRE_StructSolver solver,
                                        HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructSMGSetNumPreRelax_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructSMGSetNumPreRelax_long_dbl (HYPRE_StructSolver solver,
                                        HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructSMGSetPrintLevel_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructSMGSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructSMGSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          print_level);
HYPRE_Int HYPRE_StructSMGSetRelChange_flt (HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructSMGSetRelChange_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructSMGSetRelChange_long_dbl (HYPRE_StructSolver solver,
                                      HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructSMGSetTol_flt (HYPRE_StructSolver solver,
                                hypre_float         tol);
HYPRE_Int HYPRE_StructSMGSetTol_dbl (HYPRE_StructSolver solver,
                                hypre_double         tol);
HYPRE_Int HYPRE_StructSMGSetTol_long_dbl (HYPRE_StructSolver solver,
                                hypre_long_double         tol);
HYPRE_Int HYPRE_StructSMGSetup_flt (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSMGSetup_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSMGSetup_long_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSMGSetZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGSetZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGSetZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSMGSolve_flt (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSMGSolve_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSMGSolve_long_dbl (HYPRE_StructSolver solver,
                               HYPRE_StructMatrix A,
                               HYPRE_StructVector b,
                               HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSparseMSGCreate_flt (MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructSparseMSGCreate_dbl (MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructSparseMSGCreate_long_dbl (MPI_Comm            comm,
                                      HYPRE_StructSolver *solver);
HYPRE_Int HYPRE_StructSparseMSGDestroy_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGDestroy_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGDestroy_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm_flt (HYPRE_StructSolver  solver,
                                                            hypre_float         *norm);
HYPRE_Int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm_dbl (HYPRE_StructSolver  solver,
                                                            hypre_double         *norm);
HYPRE_Int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm_long_dbl (HYPRE_StructSolver  solver,
                                                            hypre_long_double         *norm);
HYPRE_Int HYPRE_StructSparseMSGGetNumIterations_flt (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructSparseMSGGetNumIterations_dbl (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructSparseMSGGetNumIterations_long_dbl (HYPRE_StructSolver  solver,
                                                HYPRE_Int          *num_iterations);
HYPRE_Int HYPRE_StructSparseMSGSetJacobiWeight_flt (HYPRE_StructSolver solver,
                                               hypre_float         weight);
HYPRE_Int HYPRE_StructSparseMSGSetJacobiWeight_dbl (HYPRE_StructSolver solver,
                                               hypre_double         weight);
HYPRE_Int HYPRE_StructSparseMSGSetJacobiWeight_long_dbl (HYPRE_StructSolver solver,
                                               hypre_long_double         weight);
HYPRE_Int HYPRE_StructSparseMSGSetJump_flt (HYPRE_StructSolver solver,
                                       HYPRE_Int          jump);
HYPRE_Int HYPRE_StructSparseMSGSetJump_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          jump);
HYPRE_Int HYPRE_StructSparseMSGSetJump_long_dbl (HYPRE_StructSolver solver,
                                       HYPRE_Int          jump);
HYPRE_Int HYPRE_StructSparseMSGSetLogging_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);
HYPRE_Int HYPRE_StructSparseMSGSetLogging_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);
HYPRE_Int HYPRE_StructSparseMSGSetLogging_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          logging);
HYPRE_Int HYPRE_StructSparseMSGSetMaxIter_flt (HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructSparseMSGSetMaxIter_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructSparseMSGSetMaxIter_long_dbl (HYPRE_StructSolver solver,
                                          HYPRE_Int          max_iter);
HYPRE_Int HYPRE_StructSparseMSGSetNonZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGSetNonZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGSetNonZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGSetNumFineRelax_flt (HYPRE_StructSolver solver,
                                               HYPRE_Int          num_fine_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumFineRelax_dbl (HYPRE_StructSolver solver,
                                               HYPRE_Int          num_fine_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumFineRelax_long_dbl (HYPRE_StructSolver solver,
                                               HYPRE_Int          num_fine_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumPostRelax_flt (HYPRE_StructSolver solver,
                                               HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumPostRelax_dbl (HYPRE_StructSolver solver,
                                               HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumPostRelax_long_dbl (HYPRE_StructSolver solver,
                                               HYPRE_Int          num_post_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumPreRelax_flt (HYPRE_StructSolver solver,
                                              HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumPreRelax_dbl (HYPRE_StructSolver solver,
                                              HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructSparseMSGSetNumPreRelax_long_dbl (HYPRE_StructSolver solver,
                                              HYPRE_Int          num_pre_relax);
HYPRE_Int HYPRE_StructSparseMSGSetPrintLevel_flt (HYPRE_StructSolver solver,
                                             HYPRE_Int   print_level);
HYPRE_Int HYPRE_StructSparseMSGSetPrintLevel_dbl (HYPRE_StructSolver solver,
                                             HYPRE_Int   print_level);
HYPRE_Int HYPRE_StructSparseMSGSetPrintLevel_long_dbl (HYPRE_StructSolver solver,
                                             HYPRE_Int   print_level);
HYPRE_Int HYPRE_StructSparseMSGSetRelaxType_flt (HYPRE_StructSolver solver,
                                            HYPRE_Int          relax_type);
HYPRE_Int HYPRE_StructSparseMSGSetRelaxType_dbl (HYPRE_StructSolver solver,
                                            HYPRE_Int          relax_type);
HYPRE_Int HYPRE_StructSparseMSGSetRelaxType_long_dbl (HYPRE_StructSolver solver,
                                            HYPRE_Int          relax_type);
HYPRE_Int HYPRE_StructSparseMSGSetRelChange_flt (HYPRE_StructSolver solver,
                                            HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructSparseMSGSetRelChange_dbl (HYPRE_StructSolver solver,
                                            HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructSparseMSGSetRelChange_long_dbl (HYPRE_StructSolver solver,
                                            HYPRE_Int          rel_change);
HYPRE_Int HYPRE_StructSparseMSGSetTol_flt (HYPRE_StructSolver solver,
                                      hypre_float         tol);
HYPRE_Int HYPRE_StructSparseMSGSetTol_dbl (HYPRE_StructSolver solver,
                                      hypre_double         tol);
HYPRE_Int HYPRE_StructSparseMSGSetTol_long_dbl (HYPRE_StructSolver solver,
                                      hypre_long_double         tol);
HYPRE_Int HYPRE_StructSparseMSGSetup_flt (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSparseMSGSetup_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSparseMSGSetup_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSparseMSGSetZeroGuess_flt (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGSetZeroGuess_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGSetZeroGuess_long_dbl (HYPRE_StructSolver solver);
HYPRE_Int HYPRE_StructSparseMSGSolve_flt (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSparseMSGSolve_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
HYPRE_Int HYPRE_StructSparseMSGSolve_long_dbl (HYPRE_StructSolver solver,
                                     HYPRE_StructMatrix A,
                                     HYPRE_StructVector b,
                                     HYPRE_StructVector x);
void *hypre_JacobiCreate_flt  ( MPI_Comm comm );
void *hypre_JacobiCreate_dbl  ( MPI_Comm comm );
void *hypre_JacobiCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_JacobiDestroy_flt  ( void *jacobi_vdata );
HYPRE_Int hypre_JacobiDestroy_dbl  ( void *jacobi_vdata );
HYPRE_Int hypre_JacobiDestroy_long_dbl  ( void *jacobi_vdata );
HYPRE_Int hypre_JacobiGetFinalRelativeResidualNorm_flt  ( void *jacobi_vdata, hypre_float *norm );
HYPRE_Int hypre_JacobiGetFinalRelativeResidualNorm_dbl  ( void *jacobi_vdata, hypre_double *norm );
HYPRE_Int hypre_JacobiGetFinalRelativeResidualNorm_long_dbl  ( void *jacobi_vdata, hypre_long_double *norm );
HYPRE_Int hypre_JacobiGetMaxIter_flt  ( void *jacobi_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_JacobiGetMaxIter_dbl  ( void *jacobi_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_JacobiGetMaxIter_long_dbl  ( void *jacobi_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_JacobiGetNumIterations_flt  ( void *jacobi_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_JacobiGetNumIterations_dbl  ( void *jacobi_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_JacobiGetNumIterations_long_dbl  ( void *jacobi_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_JacobiGetTol_flt  ( void *jacobi_vdata, hypre_float *tol );
HYPRE_Int hypre_JacobiGetTol_dbl  ( void *jacobi_vdata, hypre_double *tol );
HYPRE_Int hypre_JacobiGetTol_long_dbl  ( void *jacobi_vdata, hypre_long_double *tol );
HYPRE_Int hypre_JacobiGetZeroGuess_flt  ( void *jacobi_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_JacobiGetZeroGuess_dbl  ( void *jacobi_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_JacobiGetZeroGuess_long_dbl  ( void *jacobi_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_JacobiSetMaxIter_flt  ( void *jacobi_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_JacobiSetMaxIter_dbl  ( void *jacobi_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_JacobiSetMaxIter_long_dbl  ( void *jacobi_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_JacobiSetTempVec_flt  ( void *jacobi_vdata, hypre_StructVector *t );
HYPRE_Int hypre_JacobiSetTempVec_dbl  ( void *jacobi_vdata, hypre_StructVector *t );
HYPRE_Int hypre_JacobiSetTempVec_long_dbl  ( void *jacobi_vdata, hypre_StructVector *t );
HYPRE_Int hypre_JacobiSetTol_flt  ( void *jacobi_vdata, hypre_float tol );
HYPRE_Int hypre_JacobiSetTol_dbl  ( void *jacobi_vdata, hypre_double tol );
HYPRE_Int hypre_JacobiSetTol_long_dbl  ( void *jacobi_vdata, hypre_long_double tol );
HYPRE_Int hypre_JacobiSetup_flt  ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_JacobiSetup_dbl  ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_JacobiSetup_long_dbl  ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_JacobiSetZeroGuess_flt  ( void *jacobi_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_JacobiSetZeroGuess_dbl  ( void *jacobi_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_JacobiSetZeroGuess_long_dbl  ( void *jacobi_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_JacobiSolve_flt  ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_JacobiSolve_dbl  ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_JacobiSolve_long_dbl  ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_StructKrylovAxpy_flt  ( hypre_float alpha, void *x, void *y );
HYPRE_Int hypre_StructKrylovAxpy_dbl  ( hypre_double alpha, void *x, void *y );
HYPRE_Int hypre_StructKrylovAxpy_long_dbl  ( hypre_long_double alpha, void *x, void *y );
void *hypre_StructKrylovCAlloc_flt  ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
void *hypre_StructKrylovCAlloc_dbl  ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
void *hypre_StructKrylovCAlloc_long_dbl  ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
HYPRE_Int hypre_StructKrylovClearVector_flt  ( void *x );
HYPRE_Int hypre_StructKrylovClearVector_dbl  ( void *x );
HYPRE_Int hypre_StructKrylovClearVector_long_dbl  ( void *x );
HYPRE_Int hypre_StructKrylovCommInfo_flt  ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
HYPRE_Int hypre_StructKrylovCommInfo_dbl  ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
HYPRE_Int hypre_StructKrylovCommInfo_long_dbl  ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
HYPRE_Int hypre_StructKrylovCopyVector_flt  ( void *x, void *y );
HYPRE_Int hypre_StructKrylovCopyVector_dbl  ( void *x, void *y );
HYPRE_Int hypre_StructKrylovCopyVector_long_dbl  ( void *x, void *y );
void *hypre_StructKrylovCreateVector_flt  ( void *vvector );
void *hypre_StructKrylovCreateVector_dbl  ( void *vvector );
void *hypre_StructKrylovCreateVector_long_dbl  ( void *vvector );
void *hypre_StructKrylovCreateVectorArray_flt  ( HYPRE_Int n, void *vvector );
void *hypre_StructKrylovCreateVectorArray_dbl  ( HYPRE_Int n, void *vvector );
void *hypre_StructKrylovCreateVectorArray_long_dbl  ( HYPRE_Int n, void *vvector );
HYPRE_Int hypre_StructKrylovDestroyVector_flt  ( void *vvector );
HYPRE_Int hypre_StructKrylovDestroyVector_dbl  ( void *vvector );
HYPRE_Int hypre_StructKrylovDestroyVector_long_dbl  ( void *vvector );
HYPRE_Int hypre_StructKrylovFree_flt  ( void *ptr );
HYPRE_Int hypre_StructKrylovFree_dbl  ( void *ptr );
HYPRE_Int hypre_StructKrylovFree_long_dbl  ( void *ptr );
HYPRE_Int hypre_StructKrylovIdentity_flt  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_StructKrylovIdentity_dbl  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_StructKrylovIdentity_long_dbl  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_StructKrylovIdentitySetup_flt  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_StructKrylovIdentitySetup_dbl  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_StructKrylovIdentitySetup_long_dbl  ( void *vdata, void *A, void *b, void *x );
hypre_float hypre_StructKrylovInnerProd_flt  ( void *x, void *y );
hypre_double hypre_StructKrylovInnerProd_dbl  ( void *x, void *y );
hypre_long_double hypre_StructKrylovInnerProd_long_dbl  ( void *x, void *y );
HYPRE_Int hypre_StructKrylovMatvec_flt  ( void *matvec_data, hypre_float alpha, void *A, void *x,
                                     hypre_float beta, void *y );
HYPRE_Int hypre_StructKrylovMatvec_dbl  ( void *matvec_data, hypre_double alpha, void *A, void *x,
                                     hypre_double beta, void *y );
HYPRE_Int hypre_StructKrylovMatvec_long_dbl  ( void *matvec_data, hypre_long_double alpha, void *A, void *x,
                                     hypre_long_double beta, void *y );
void *hypre_StructKrylovMatvecCreate_flt  ( void *A, void *x );
void *hypre_StructKrylovMatvecCreate_dbl  ( void *A, void *x );
void *hypre_StructKrylovMatvecCreate_long_dbl  ( void *A, void *x );
HYPRE_Int hypre_StructKrylovMatvecDestroy_flt  ( void *matvec_data );
HYPRE_Int hypre_StructKrylovMatvecDestroy_dbl  ( void *matvec_data );
HYPRE_Int hypre_StructKrylovMatvecDestroy_long_dbl  ( void *matvec_data );
HYPRE_Int hypre_StructKrylovScaleVector_flt  ( hypre_float alpha, void *x );
HYPRE_Int hypre_StructKrylovScaleVector_dbl  ( hypre_double alpha, void *x );
HYPRE_Int hypre_StructKrylovScaleVector_long_dbl  ( hypre_long_double alpha, void *x );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
hypre_StructMatrix *hypre_PFMG2CreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMG2CreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMG2CreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1_flt  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1_long_dbl  ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
hypre_StructMatrix *hypre_PFMG3CreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMG3CreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMG3CreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
void *hypre_PFMGCreate_flt  ( MPI_Comm comm );
void *hypre_PFMGCreate_dbl  ( MPI_Comm comm );
void *hypre_PFMGCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_PFMGDestroy_flt  ( void *pfmg_vdata );
HYPRE_Int hypre_PFMGDestroy_dbl  ( void *pfmg_vdata );
HYPRE_Int hypre_PFMGDestroy_long_dbl  ( void *pfmg_vdata );
HYPRE_Int hypre_PFMGGetFinalRelativeResidualNorm_flt  ( void *pfmg_vdata,
                                                   hypre_float *relative_residual_norm );
HYPRE_Int hypre_PFMGGetFinalRelativeResidualNorm_dbl  ( void *pfmg_vdata,
                                                   hypre_double *relative_residual_norm );
HYPRE_Int hypre_PFMGGetFinalRelativeResidualNorm_long_dbl  ( void *pfmg_vdata,
                                                   hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_PFMGGetJacobiWeight_flt  ( void *pfmg_vdata, hypre_float *weight );
HYPRE_Int hypre_PFMGGetJacobiWeight_dbl  ( void *pfmg_vdata, hypre_double *weight );
HYPRE_Int hypre_PFMGGetJacobiWeight_long_dbl  ( void *pfmg_vdata, hypre_long_double *weight );
HYPRE_Int hypre_PFMGGetLogging_flt  ( void *pfmg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_PFMGGetLogging_dbl  ( void *pfmg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_PFMGGetLogging_long_dbl  ( void *pfmg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_PFMGGetMaxIter_flt  ( void *pfmg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PFMGGetMaxIter_dbl  ( void *pfmg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PFMGGetMaxIter_long_dbl  ( void *pfmg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PFMGGetMaxLevels_flt  ( void *pfmg_vdata, HYPRE_Int *max_levels );
HYPRE_Int hypre_PFMGGetMaxLevels_dbl  ( void *pfmg_vdata, HYPRE_Int *max_levels );
HYPRE_Int hypre_PFMGGetMaxLevels_long_dbl  ( void *pfmg_vdata, HYPRE_Int *max_levels );
HYPRE_Int hypre_PFMGGetNumIterations_flt  ( void *pfmg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PFMGGetNumIterations_dbl  ( void *pfmg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PFMGGetNumIterations_long_dbl  ( void *pfmg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PFMGGetNumPostRelax_flt  ( void *pfmg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_PFMGGetNumPostRelax_dbl  ( void *pfmg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_PFMGGetNumPostRelax_long_dbl  ( void *pfmg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_PFMGGetNumPreRelax_flt  ( void *pfmg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_PFMGGetNumPreRelax_dbl  ( void *pfmg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_PFMGGetNumPreRelax_long_dbl  ( void *pfmg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_PFMGGetPrintLevel_flt  ( void *pfmg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_PFMGGetPrintLevel_dbl  ( void *pfmg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_PFMGGetPrintLevel_long_dbl  ( void *pfmg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_PFMGGetRAPType_flt  ( void *pfmg_vdata, HYPRE_Int *rap_type );
HYPRE_Int hypre_PFMGGetRAPType_dbl  ( void *pfmg_vdata, HYPRE_Int *rap_type );
HYPRE_Int hypre_PFMGGetRAPType_long_dbl  ( void *pfmg_vdata, HYPRE_Int *rap_type );
HYPRE_Int hypre_PFMGGetRelaxType_flt  ( void *pfmg_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_PFMGGetRelaxType_dbl  ( void *pfmg_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_PFMGGetRelaxType_long_dbl  ( void *pfmg_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_PFMGGetRelChange_flt  ( void *pfmg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PFMGGetRelChange_dbl  ( void *pfmg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PFMGGetRelChange_long_dbl  ( void *pfmg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PFMGGetSkipRelax_flt  ( void *pfmg_vdata, HYPRE_Int *skip_relax );
HYPRE_Int hypre_PFMGGetSkipRelax_dbl  ( void *pfmg_vdata, HYPRE_Int *skip_relax );
HYPRE_Int hypre_PFMGGetSkipRelax_long_dbl  ( void *pfmg_vdata, HYPRE_Int *skip_relax );
HYPRE_Int hypre_PFMGGetTol_flt  ( void *pfmg_vdata, hypre_float *tol );
HYPRE_Int hypre_PFMGGetTol_dbl  ( void *pfmg_vdata, hypre_double *tol );
HYPRE_Int hypre_PFMGGetTol_long_dbl  ( void *pfmg_vdata, hypre_long_double *tol );
HYPRE_Int hypre_PFMGGetZeroGuess_flt  ( void *pfmg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PFMGGetZeroGuess_dbl  ( void *pfmg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PFMGGetZeroGuess_long_dbl  ( void *pfmg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PFMGPrintLogging_flt  ( void *pfmg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PFMGPrintLogging_dbl  ( void *pfmg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PFMGPrintLogging_long_dbl  ( void *pfmg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PFMGSetDxyz_flt  ( void *pfmg_vdata, hypre_float *dxyz );
HYPRE_Int hypre_PFMGSetDxyz_dbl  ( void *pfmg_vdata, hypre_double *dxyz );
HYPRE_Int hypre_PFMGSetDxyz_long_dbl  ( void *pfmg_vdata, hypre_long_double *dxyz );
HYPRE_Int hypre_PFMGSetJacobiWeight_flt  ( void *pfmg_vdata, hypre_float weight );
HYPRE_Int hypre_PFMGSetJacobiWeight_dbl  ( void *pfmg_vdata, hypre_double weight );
HYPRE_Int hypre_PFMGSetJacobiWeight_long_dbl  ( void *pfmg_vdata, hypre_long_double weight );
HYPRE_Int hypre_PFMGSetLogging_flt  ( void *pfmg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_PFMGSetLogging_dbl  ( void *pfmg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_PFMGSetLogging_long_dbl  ( void *pfmg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_PFMGSetMaxIter_flt  ( void *pfmg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGSetMaxIter_dbl  ( void *pfmg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGSetMaxIter_long_dbl  ( void *pfmg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGSetMaxLevels_flt  ( void *pfmg_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_PFMGSetMaxLevels_dbl  ( void *pfmg_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_PFMGSetMaxLevels_long_dbl  ( void *pfmg_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_PFMGSetNumPostRelax_flt  ( void *pfmg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_PFMGSetNumPostRelax_dbl  ( void *pfmg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_PFMGSetNumPostRelax_long_dbl  ( void *pfmg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_PFMGSetNumPreRelax_flt  ( void *pfmg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_PFMGSetNumPreRelax_dbl  ( void *pfmg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_PFMGSetNumPreRelax_long_dbl  ( void *pfmg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_PFMGSetPrintLevel_flt  ( void *pfmg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_PFMGSetPrintLevel_dbl  ( void *pfmg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_PFMGSetPrintLevel_long_dbl  ( void *pfmg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_PFMGSetRAPType_flt  ( void *pfmg_vdata, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetRAPType_dbl  ( void *pfmg_vdata, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetRAPType_long_dbl  ( void *pfmg_vdata, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetRelaxType_flt  ( void *pfmg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGSetRelaxType_dbl  ( void *pfmg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGSetRelaxType_long_dbl  ( void *pfmg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGSetRelChange_flt  ( void *pfmg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PFMGSetRelChange_dbl  ( void *pfmg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PFMGSetRelChange_long_dbl  ( void *pfmg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PFMGSetSkipRelax_flt  ( void *pfmg_vdata, HYPRE_Int skip_relax );
HYPRE_Int hypre_PFMGSetSkipRelax_dbl  ( void *pfmg_vdata, HYPRE_Int skip_relax );
HYPRE_Int hypre_PFMGSetSkipRelax_long_dbl  ( void *pfmg_vdata, HYPRE_Int skip_relax );
HYPRE_Int hypre_PFMGSetTol_flt  ( void *pfmg_vdata, hypre_float tol );
HYPRE_Int hypre_PFMGSetTol_dbl  ( void *pfmg_vdata, hypre_double tol );
HYPRE_Int hypre_PFMGSetTol_long_dbl  ( void *pfmg_vdata, hypre_long_double tol );
HYPRE_Int hypre_PFMGSetZeroGuess_flt  ( void *pfmg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGSetZeroGuess_dbl  ( void *pfmg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGSetZeroGuess_long_dbl  ( void *pfmg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGRelax_flt  ( void *pfmg_relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelax_dbl  ( void *pfmg_relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelax_long_dbl  ( void *pfmg_relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
void *hypre_PFMGRelaxCreate_flt  ( MPI_Comm comm );
void *hypre_PFMGRelaxCreate_dbl  ( MPI_Comm comm );
void *hypre_PFMGRelaxCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_PFMGRelaxDestroy_flt  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxDestroy_dbl  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxDestroy_long_dbl  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetJacobiWeight_flt  ( void *pfmg_relax_vdata, hypre_float weight );
HYPRE_Int hypre_PFMGRelaxSetJacobiWeight_dbl  ( void *pfmg_relax_vdata, hypre_double weight );
HYPRE_Int hypre_PFMGRelaxSetJacobiWeight_long_dbl  ( void *pfmg_relax_vdata, hypre_long_double weight );
HYPRE_Int hypre_PFMGRelaxSetMaxIter_flt  ( void *pfmg_relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGRelaxSetMaxIter_dbl  ( void *pfmg_relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGRelaxSetMaxIter_long_dbl  ( void *pfmg_relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGRelaxSetPostRelax_flt  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetPostRelax_dbl  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetPostRelax_long_dbl  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetPreRelax_flt  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetPreRelax_dbl  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetPreRelax_long_dbl  ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetTempVec_flt  ( void *pfmg_relax_vdata, hypre_StructVector *t );
HYPRE_Int hypre_PFMGRelaxSetTempVec_dbl  ( void *pfmg_relax_vdata, hypre_StructVector *t );
HYPRE_Int hypre_PFMGRelaxSetTempVec_long_dbl  ( void *pfmg_relax_vdata, hypre_StructVector *t );
HYPRE_Int hypre_PFMGRelaxSetTol_flt  ( void *pfmg_relax_vdata, hypre_float tol );
HYPRE_Int hypre_PFMGRelaxSetTol_dbl  ( void *pfmg_relax_vdata, hypre_double tol );
HYPRE_Int hypre_PFMGRelaxSetTol_long_dbl  ( void *pfmg_relax_vdata, hypre_long_double tol );
HYPRE_Int hypre_PFMGRelaxSetType_flt  ( void *pfmg_relax_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGRelaxSetType_dbl  ( void *pfmg_relax_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGRelaxSetType_long_dbl  ( void *pfmg_relax_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGRelaxSetup_flt  ( void *pfmg_relax_vdata, hypre_StructMatrix *A,
                                 hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelaxSetup_dbl  ( void *pfmg_relax_vdata, hypre_StructMatrix *A,
                                 hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelaxSetup_long_dbl  ( void *pfmg_relax_vdata, hypre_StructMatrix *A,
                                 hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelaxSetZeroGuess_flt  ( void *pfmg_relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGRelaxSetZeroGuess_dbl  ( void *pfmg_relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGRelaxSetZeroGuess_long_dbl  ( void *pfmg_relax_vdata, HYPRE_Int zero_guess );
hypre_StructMatrix *hypre_PFMGCreateInterpOp_flt  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                               HYPRE_Int cdir, HYPRE_Int rap_type );
hypre_StructMatrix *hypre_PFMGCreateInterpOp_dbl  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                               HYPRE_Int cdir, HYPRE_Int rap_type );
hypre_StructMatrix *hypre_PFMGCreateInterpOp_long_dbl  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                               HYPRE_Int cdir, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp_flt  ( hypre_StructMatrix *A, HYPRE_Int cdir, hypre_Index findex,
                                    hypre_Index stride, hypre_StructMatrix *P, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp_dbl  ( hypre_StructMatrix *A, HYPRE_Int cdir, hypre_Index findex,
                                    hypre_Index stride, hypre_StructMatrix *P, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp_long_dbl  ( hypre_StructMatrix *A, HYPRE_Int cdir, hypre_Index findex,
                                    hypre_Index stride, hypre_StructMatrix *P, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS15_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                             hypre_float *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS15_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                             hypre_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS15_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                             hypre_long_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS19_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                             hypre_float *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS19_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                             hypre_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS19_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                             hypre_long_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS27_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                             hypre_float *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS27_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                             hypre_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS27_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                             hypre_long_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS5_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                            hypre_float *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS5_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                            hypre_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS5_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                            hypre_long_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS7_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                            hypre_float *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS7_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                            hypre_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS7_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                            hypre_long_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS9_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                            hypre_float *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS9_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                            hypre_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS9_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                            hypre_long_double *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC1_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                        hypre_float *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC1_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                        hypre_double *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC1_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                        hypre_long_double *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC2_flt  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_float *Pp0,
                                        hypre_float *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC2_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_double *Pp0,
                                        hypre_double *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC2_long_dbl  ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, hypre_long_double *Pp0,
                                        hypre_long_double *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGComputeDxyz_flt  ( hypre_StructMatrix *A, hypre_float *dxyz, hypre_float *mean,
                                  hypre_float *deviation);
HYPRE_Int hypre_PFMGComputeDxyz_dbl  ( hypre_StructMatrix *A, hypre_double *dxyz, hypre_double *mean,
                                  hypre_double *deviation);
HYPRE_Int hypre_PFMGComputeDxyz_long_dbl  ( hypre_StructMatrix *A, hypre_long_double *dxyz, hypre_long_double *mean,
                                  hypre_long_double *deviation);
HYPRE_Int hypre_PFMGComputeDxyz_CS_flt   ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_float *cxyz,
                                      hypre_float *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_CS_dbl   ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_double *cxyz,
                                      hypre_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_CS_long_dbl   ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_long_double *cxyz,
                                      hypre_long_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS19_flt ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_float *cxyz,
                                      hypre_float *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS19_dbl ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_double *cxyz,
                                      hypre_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS19_long_dbl ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_long_double *cxyz,
                                      hypre_long_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS27_flt ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_float *cxyz,
                                      hypre_float *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS27_dbl ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_double *cxyz,
                                      hypre_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS27_long_dbl ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_long_double *cxyz,
                                      hypre_long_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS5_flt  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_float *cxyz,
                                      hypre_float *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS5_dbl  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_double *cxyz,
                                      hypre_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS5_long_dbl  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_long_double *cxyz,
                                      hypre_long_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS7_flt  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_float *cxyz,
                                      hypre_float *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS7_dbl  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_double *cxyz,
                                      hypre_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS7_long_dbl  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_long_double *cxyz,
                                      hypre_long_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS9_flt  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_float *cxyz,
                                      hypre_float *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS9_dbl  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_double *cxyz,
                                      hypre_double *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS9_long_dbl  ( HYPRE_Int bi, hypre_StructMatrix *A, hypre_long_double *cxyz,
                                      hypre_long_double *sqcxyz);
HYPRE_Int hypre_PFMGSetup_flt  ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGSetup_dbl  ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGSetup_long_dbl  ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_ZeroDiagonal_flt  ( hypre_StructMatrix *A );
HYPRE_Int hypre_ZeroDiagonal_dbl  ( hypre_StructMatrix *A );
HYPRE_Int hypre_ZeroDiagonal_long_dbl  ( hypre_StructMatrix *A );
HYPRE_Int hypre_PFMGBuildCoarseOp5_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
hypre_StructMatrix *hypre_PFMGCreateCoarseOp5_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMGCreateCoarseOp5_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMGCreateCoarseOp5_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_PFMGBuildCoarseOp7_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp7_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp7_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
hypre_StructMatrix *hypre_PFMGCreateCoarseOp7_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMGCreateCoarseOp7_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMGCreateCoarseOp7_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_PFMGCreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir, HYPRE_Int rap_type );
hypre_StructMatrix *hypre_PFMGCreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir, HYPRE_Int rap_type );
hypre_StructMatrix *hypre_PFMGCreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                 hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int rap_type,
                                 hypre_StructMatrix *Ac );
HYPRE_Int hypre_PFMGSetupRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                 hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int rap_type,
                                 hypre_StructMatrix *Ac );
HYPRE_Int hypre_PFMGSetupRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                 hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int rap_type,
                                 hypre_StructMatrix *Ac );
HYPRE_Int hypre_PFMGSolve_flt  ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGSolve_dbl  ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGSolve_long_dbl  ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PointRelax_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
HYPRE_Int hypre_PointRelax_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
HYPRE_Int hypre_PointRelax_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
HYPRE_Int hypre_PointRelax_core0_flt  ( void *relax_vdata, hypre_StructMatrix *A,
                                   HYPRE_Int constant_coefficient, hypre_Box *compute_box, hypre_float *bp, hypre_float *xp,
                                   hypre_float *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                   hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
HYPRE_Int hypre_PointRelax_core0_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                   HYPRE_Int constant_coefficient, hypre_Box *compute_box, hypre_double *bp, hypre_double *xp,
                                   hypre_double *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                   hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
HYPRE_Int hypre_PointRelax_core0_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                   HYPRE_Int constant_coefficient, hypre_Box *compute_box, hypre_long_double *bp, hypre_long_double *xp,
                                   hypre_long_double *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                   hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
HYPRE_Int hypre_PointRelax_core12_flt  ( void *relax_vdata, hypre_StructMatrix *A,
                                    HYPRE_Int constant_coefficient, hypre_Box *compute_box, hypre_float *bp, hypre_float *xp,
                                    hypre_float *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                    hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
HYPRE_Int hypre_PointRelax_core12_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                    HYPRE_Int constant_coefficient, hypre_Box *compute_box, hypre_double *bp, hypre_double *xp,
                                    hypre_double *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                    hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
HYPRE_Int hypre_PointRelax_core12_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                    HYPRE_Int constant_coefficient, hypre_Box *compute_box, hypre_long_double *bp, hypre_long_double *xp,
                                    hypre_long_double *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                    hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
void *hypre_PointRelaxCreate_flt  ( MPI_Comm comm );
void *hypre_PointRelaxCreate_dbl  ( MPI_Comm comm );
void *hypre_PointRelaxCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_PointRelaxDestroy_flt  ( void *relax_vdata );
HYPRE_Int hypre_PointRelaxDestroy_dbl  ( void *relax_vdata );
HYPRE_Int hypre_PointRelaxDestroy_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_PointRelaxGetFinalRelativeResidualNorm_flt  ( void *relax_vdata, hypre_float *norm );
HYPRE_Int hypre_PointRelaxGetFinalRelativeResidualNorm_dbl  ( void *relax_vdata, hypre_double *norm );
HYPRE_Int hypre_PointRelaxGetFinalRelativeResidualNorm_long_dbl  ( void *relax_vdata, hypre_long_double *norm );
HYPRE_Int hypre_PointRelaxGetMaxIter_flt  ( void *relax_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PointRelaxGetMaxIter_dbl  ( void *relax_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PointRelaxGetMaxIter_long_dbl  ( void *relax_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PointRelaxGetNumIterations_flt  ( void *relax_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PointRelaxGetNumIterations_dbl  ( void *relax_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PointRelaxGetNumIterations_long_dbl  ( void *relax_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PointRelaxGetTol_flt  ( void *relax_vdata, hypre_float *tol );
HYPRE_Int hypre_PointRelaxGetTol_dbl  ( void *relax_vdata, hypre_double *tol );
HYPRE_Int hypre_PointRelaxGetTol_long_dbl  ( void *relax_vdata, hypre_long_double *tol );
HYPRE_Int hypre_PointRelaxGetZeroGuess_flt  ( void *relax_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PointRelaxGetZeroGuess_dbl  ( void *relax_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PointRelaxGetZeroGuess_long_dbl  ( void *relax_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PointRelaxSetMaxIter_flt  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PointRelaxSetMaxIter_dbl  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PointRelaxSetMaxIter_long_dbl  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PointRelaxSetNumPointsets_flt  ( void *relax_vdata, HYPRE_Int num_pointsets );
HYPRE_Int hypre_PointRelaxSetNumPointsets_dbl  ( void *relax_vdata, HYPRE_Int num_pointsets );
HYPRE_Int hypre_PointRelaxSetNumPointsets_long_dbl  ( void *relax_vdata, HYPRE_Int num_pointsets );
HYPRE_Int hypre_PointRelaxSetPointset_flt  ( void *relax_vdata, HYPRE_Int pointset,
                                        HYPRE_Int pointset_size, hypre_Index pointset_stride, hypre_Index *pointset_indices );
HYPRE_Int hypre_PointRelaxSetPointset_dbl  ( void *relax_vdata, HYPRE_Int pointset,
                                        HYPRE_Int pointset_size, hypre_Index pointset_stride, hypre_Index *pointset_indices );
HYPRE_Int hypre_PointRelaxSetPointset_long_dbl  ( void *relax_vdata, HYPRE_Int pointset,
                                        HYPRE_Int pointset_size, hypre_Index pointset_stride, hypre_Index *pointset_indices );
HYPRE_Int hypre_PointRelaxSetPointsetRank_flt  ( void *relax_vdata, HYPRE_Int pointset,
                                            HYPRE_Int pointset_rank );
HYPRE_Int hypre_PointRelaxSetPointsetRank_dbl  ( void *relax_vdata, HYPRE_Int pointset,
                                            HYPRE_Int pointset_rank );
HYPRE_Int hypre_PointRelaxSetPointsetRank_long_dbl  ( void *relax_vdata, HYPRE_Int pointset,
                                            HYPRE_Int pointset_rank );
HYPRE_Int hypre_PointRelaxSetTempVec_flt  ( void *relax_vdata, hypre_StructVector *t );
HYPRE_Int hypre_PointRelaxSetTempVec_dbl  ( void *relax_vdata, hypre_StructVector *t );
HYPRE_Int hypre_PointRelaxSetTempVec_long_dbl  ( void *relax_vdata, hypre_StructVector *t );
HYPRE_Int hypre_PointRelaxSetTol_flt  ( void *relax_vdata, hypre_float tol );
HYPRE_Int hypre_PointRelaxSetTol_dbl  ( void *relax_vdata, hypre_double tol );
HYPRE_Int hypre_PointRelaxSetTol_long_dbl  ( void *relax_vdata, hypre_long_double tol );
HYPRE_Int hypre_PointRelaxSetup_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_PointRelaxSetup_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_PointRelaxSetup_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_PointRelaxSetWeight_flt  ( void *relax_vdata, hypre_float weight );
HYPRE_Int hypre_PointRelaxSetWeight_dbl  ( void *relax_vdata, hypre_double weight );
HYPRE_Int hypre_PointRelaxSetWeight_long_dbl  ( void *relax_vdata, hypre_long_double weight );
HYPRE_Int hypre_PointRelaxSetZeroGuess_flt  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PointRelaxSetZeroGuess_dbl  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PointRelaxSetZeroGuess_long_dbl  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_relax_copy_flt  ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                             hypre_StructVector *x );
HYPRE_Int hypre_relax_copy_dbl  ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                             hypre_StructVector *x );
HYPRE_Int hypre_relax_copy_long_dbl  ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                             hypre_StructVector *x );
HYPRE_Int hypre_relax_wtx_flt  ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                            hypre_StructVector *x );
HYPRE_Int hypre_relax_wtx_dbl  ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                            hypre_StructVector *x );
HYPRE_Int hypre_relax_wtx_long_dbl  ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                            hypre_StructVector *x );
HYPRE_Int hypre_RedBlackConstantCoefGS_flt  ( void *relax_vdata, hypre_StructMatrix *A,
                                         hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_RedBlackConstantCoefGS_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                         hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_RedBlackConstantCoefGS_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                         hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGS_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGS_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGS_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
void *hypre_RedBlackGSCreate_flt  ( MPI_Comm comm );
void *hypre_RedBlackGSCreate_dbl  ( MPI_Comm comm );
void *hypre_RedBlackGSCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_RedBlackGSDestroy_flt  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSDestroy_dbl  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSDestroy_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetMaxIter_flt  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_RedBlackGSSetMaxIter_dbl  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_RedBlackGSSetMaxIter_long_dbl  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_RedBlackGSSetStartBlack_flt  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetStartBlack_dbl  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetStartBlack_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetStartRed_flt  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetStartRed_dbl  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetStartRed_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetTol_flt  ( void *relax_vdata, hypre_float tol );
HYPRE_Int hypre_RedBlackGSSetTol_dbl  ( void *relax_vdata, hypre_double tol );
HYPRE_Int hypre_RedBlackGSSetTol_long_dbl  ( void *relax_vdata, hypre_long_double tol );
HYPRE_Int hypre_RedBlackGSSetup_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGSSetup_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGSSetup_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGSSetZeroGuess_flt  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_RedBlackGSSetZeroGuess_dbl  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_RedBlackGSSetZeroGuess_long_dbl  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SemiInterp_flt  ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                             hypre_StructVector *e );
HYPRE_Int hypre_SemiInterp_dbl  ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                             hypre_StructVector *e );
HYPRE_Int hypre_SemiInterp_long_dbl  ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                             hypre_StructVector *e );
void *hypre_SemiInterpCreate_flt  ( void );
void *hypre_SemiInterpCreate_dbl  ( void );
void *hypre_SemiInterpCreate_long_dbl  ( void );
HYPRE_Int hypre_SemiInterpDestroy_flt  ( void *interp_vdata );
HYPRE_Int hypre_SemiInterpDestroy_dbl  ( void *interp_vdata );
HYPRE_Int hypre_SemiInterpDestroy_long_dbl  ( void *interp_vdata );
HYPRE_Int hypre_SemiInterpSetup_flt  ( void *interp_vdata, hypre_StructMatrix *P,
                                  HYPRE_Int P_stored_as_transpose, hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex,
                                  hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SemiInterpSetup_dbl  ( void *interp_vdata, hypre_StructMatrix *P,
                                  HYPRE_Int P_stored_as_transpose, hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex,
                                  hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SemiInterpSetup_long_dbl  ( void *interp_vdata, hypre_StructMatrix *P,
                                  HYPRE_Int P_stored_as_transpose, hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex,
                                  hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_StructInterpAssemble_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                       HYPRE_Int P_stored_as_transpose, HYPRE_Int cdir, hypre_Index index, hypre_Index stride );
HYPRE_Int hypre_StructInterpAssemble_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                       HYPRE_Int P_stored_as_transpose, HYPRE_Int cdir, hypre_Index index, hypre_Index stride );
HYPRE_Int hypre_StructInterpAssemble_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                       HYPRE_Int P_stored_as_transpose, HYPRE_Int cdir, hypre_Index index, hypre_Index stride );
HYPRE_Int hypre_SemiRestrict_flt  ( void *restrict_vdata, hypre_StructMatrix *R, hypre_StructVector *r,
                               hypre_StructVector *rc );
HYPRE_Int hypre_SemiRestrict_dbl  ( void *restrict_vdata, hypre_StructMatrix *R, hypre_StructVector *r,
                               hypre_StructVector *rc );
HYPRE_Int hypre_SemiRestrict_long_dbl  ( void *restrict_vdata, hypre_StructMatrix *R, hypre_StructVector *r,
                               hypre_StructVector *rc );
void *hypre_SemiRestrictCreate_flt  ( void );
void *hypre_SemiRestrictCreate_dbl  ( void );
void *hypre_SemiRestrictCreate_long_dbl  ( void );
HYPRE_Int hypre_SemiRestrictDestroy_flt  ( void *restrict_vdata );
HYPRE_Int hypre_SemiRestrictDestroy_dbl  ( void *restrict_vdata );
HYPRE_Int hypre_SemiRestrictDestroy_long_dbl  ( void *restrict_vdata );
HYPRE_Int hypre_SemiRestrictSetup_flt  ( void *restrict_vdata, hypre_StructMatrix *R,
                                    HYPRE_Int R_stored_as_transpose, hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex,
                                    hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SemiRestrictSetup_dbl  ( void *restrict_vdata, hypre_StructMatrix *R,
                                    HYPRE_Int R_stored_as_transpose, hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex,
                                    hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SemiRestrictSetup_long_dbl  ( void *restrict_vdata, hypre_StructMatrix *R,
                                    HYPRE_Int R_stored_as_transpose, hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex,
                                    hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SemiBuildRAP_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R,
                               HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int P_stored_as_transpose,
                               hypre_StructMatrix *RAP );
HYPRE_Int hypre_SemiBuildRAP_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R,
                               HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int P_stored_as_transpose,
                               hypre_StructMatrix *RAP );
HYPRE_Int hypre_SemiBuildRAP_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R,
                               HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int P_stored_as_transpose,
                               hypre_StructMatrix *RAP );
hypre_StructMatrix *hypre_SemiCreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir,
                                            HYPRE_Int P_stored_as_transpose );
hypre_StructMatrix *hypre_SemiCreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir,
                                            HYPRE_Int P_stored_as_transpose );
hypre_StructMatrix *hypre_SemiCreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir,
                                            HYPRE_Int P_stored_as_transpose );
HYPRE_Int hypre_StructSetRandomValues_flt  ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_StructSetRandomValues_dbl  ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_StructSetRandomValues_long_dbl  ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_StructVectorSetRandomValues_flt  ( hypre_StructVector *vector, HYPRE_Int seed );
HYPRE_Int hypre_StructVectorSetRandomValues_dbl  ( hypre_StructVector *vector, HYPRE_Int seed );
HYPRE_Int hypre_StructVectorSetRandomValues_long_dbl  ( hypre_StructVector *vector, HYPRE_Int seed );
HYPRE_Int hypre_SMG2BuildRAPNoSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG2BuildRAPNoSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG2BuildRAPNoSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG2BuildRAPSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG2BuildRAPSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG2BuildRAPSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
hypre_StructMatrix *hypre_SMG2CreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
hypre_StructMatrix *hypre_SMG2CreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
hypre_StructMatrix *hypre_SMG2CreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMG2RAPPeriodicNoSym_flt  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicNoSym_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicNoSym_long_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicSym_flt  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicSym_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicSym_long_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPNoSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPNoSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPNoSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
hypre_StructMatrix *hypre_SMG3CreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
hypre_StructMatrix *hypre_SMG3CreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
hypre_StructMatrix *hypre_SMG3CreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMG3RAPPeriodicNoSym_flt  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicNoSym_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicNoSym_long_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicSym_flt  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicSym_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicSym_long_dbl  ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMGAxpy_flt  ( hypre_float alpha, hypre_StructVector *x, hypre_StructVector *y,
                          hypre_Index base_index, hypre_Index base_stride );
HYPRE_Int hypre_SMGAxpy_dbl  ( hypre_double alpha, hypre_StructVector *x, hypre_StructVector *y,
                          hypre_Index base_index, hypre_Index base_stride );
HYPRE_Int hypre_SMGAxpy_long_dbl  ( hypre_long_double alpha, hypre_StructVector *x, hypre_StructVector *y,
                          hypre_Index base_index, hypre_Index base_stride );
void *hypre_SMGCreate_flt  ( MPI_Comm comm );
void *hypre_SMGCreate_dbl  ( MPI_Comm comm );
void *hypre_SMGCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_SMGDestroy_flt  ( void *smg_vdata );
HYPRE_Int hypre_SMGDestroy_dbl  ( void *smg_vdata );
HYPRE_Int hypre_SMGDestroy_long_dbl  ( void *smg_vdata );
HYPRE_Int hypre_SMGGetFinalRelativeResidualNorm_flt  ( void *smg_vdata,
                                                  hypre_float *relative_residual_norm );
HYPRE_Int hypre_SMGGetFinalRelativeResidualNorm_dbl  ( void *smg_vdata,
                                                  hypre_double *relative_residual_norm );
HYPRE_Int hypre_SMGGetFinalRelativeResidualNorm_long_dbl  ( void *smg_vdata,
                                                  hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_SMGGetLogging_flt  ( void *smg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_SMGGetLogging_dbl  ( void *smg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_SMGGetLogging_long_dbl  ( void *smg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_SMGGetMaxIter_flt  ( void *smg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_SMGGetMaxIter_dbl  ( void *smg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_SMGGetMaxIter_long_dbl  ( void *smg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_SMGGetMemoryUse_flt  ( void *smg_vdata, HYPRE_Int *memory_use );
HYPRE_Int hypre_SMGGetMemoryUse_dbl  ( void *smg_vdata, HYPRE_Int *memory_use );
HYPRE_Int hypre_SMGGetMemoryUse_long_dbl  ( void *smg_vdata, HYPRE_Int *memory_use );
HYPRE_Int hypre_SMGGetNumIterations_flt  ( void *smg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SMGGetNumIterations_dbl  ( void *smg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SMGGetNumIterations_long_dbl  ( void *smg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SMGGetNumPostRelax_flt  ( void *smg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_SMGGetNumPostRelax_dbl  ( void *smg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_SMGGetNumPostRelax_long_dbl  ( void *smg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_SMGGetNumPreRelax_flt  ( void *smg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_SMGGetNumPreRelax_dbl  ( void *smg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_SMGGetNumPreRelax_long_dbl  ( void *smg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_SMGGetPrintLevel_flt  ( void *smg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_SMGGetPrintLevel_dbl  ( void *smg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_SMGGetPrintLevel_long_dbl  ( void *smg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_SMGGetRelChange_flt  ( void *smg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_SMGGetRelChange_dbl  ( void *smg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_SMGGetRelChange_long_dbl  ( void *smg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_SMGGetTol_flt  ( void *smg_vdata, hypre_float *tol );
HYPRE_Int hypre_SMGGetTol_dbl  ( void *smg_vdata, hypre_double *tol );
HYPRE_Int hypre_SMGGetTol_long_dbl  ( void *smg_vdata, hypre_long_double *tol );
HYPRE_Int hypre_SMGGetZeroGuess_flt  ( void *smg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_SMGGetZeroGuess_dbl  ( void *smg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_SMGGetZeroGuess_long_dbl  ( void *smg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_SMGPrintLogging_flt  ( void *smg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SMGPrintLogging_dbl  ( void *smg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SMGPrintLogging_long_dbl  ( void *smg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SMGSetBase_flt  ( void *smg_vdata, hypre_Index base_index, hypre_Index base_stride );
HYPRE_Int hypre_SMGSetBase_dbl  ( void *smg_vdata, hypre_Index base_index, hypre_Index base_stride );
HYPRE_Int hypre_SMGSetBase_long_dbl  ( void *smg_vdata, hypre_Index base_index, hypre_Index base_stride );
HYPRE_Int hypre_SMGSetLogging_flt  ( void *smg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SMGSetLogging_dbl  ( void *smg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SMGSetLogging_long_dbl  ( void *smg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SMGSetMaxIter_flt  ( void *smg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGSetMaxIter_dbl  ( void *smg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGSetMaxIter_long_dbl  ( void *smg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGSetMemoryUse_flt  ( void *smg_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGSetMemoryUse_dbl  ( void *smg_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGSetMemoryUse_long_dbl  ( void *smg_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGSetNumPostRelax_flt  ( void *smg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGSetNumPostRelax_dbl  ( void *smg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGSetNumPostRelax_long_dbl  ( void *smg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGSetNumPreRelax_flt  ( void *smg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGSetNumPreRelax_dbl  ( void *smg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGSetNumPreRelax_long_dbl  ( void *smg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGSetPrintLevel_flt  ( void *smg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SMGSetPrintLevel_dbl  ( void *smg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SMGSetPrintLevel_long_dbl  ( void *smg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SMGSetRelChange_flt  ( void *smg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SMGSetRelChange_dbl  ( void *smg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SMGSetRelChange_long_dbl  ( void *smg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SMGSetStructVectorConstantValues_flt  ( hypre_StructVector *vector, hypre_float values,
                                                   hypre_BoxArray *box_array, hypre_Index stride );
HYPRE_Int hypre_SMGSetStructVectorConstantValues_dbl  ( hypre_StructVector *vector, hypre_double values,
                                                   hypre_BoxArray *box_array, hypre_Index stride );
HYPRE_Int hypre_SMGSetStructVectorConstantValues_long_dbl  ( hypre_StructVector *vector, hypre_long_double values,
                                                   hypre_BoxArray *box_array, hypre_Index stride );
HYPRE_Int hypre_SMGSetTol_flt  ( void *smg_vdata, hypre_float tol );
HYPRE_Int hypre_SMGSetTol_dbl  ( void *smg_vdata, hypre_double tol );
HYPRE_Int hypre_SMGSetTol_long_dbl  ( void *smg_vdata, hypre_long_double tol );
HYPRE_Int hypre_SMGSetZeroGuess_flt  ( void *smg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGSetZeroGuess_dbl  ( void *smg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGSetZeroGuess_long_dbl  ( void *smg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_StructSMGSetMaxLevel_flt ( void   *smg_vdata, HYPRE_Int   max_level  );
HYPRE_Int hypre_StructSMGSetMaxLevel_dbl ( void   *smg_vdata, HYPRE_Int   max_level  );
HYPRE_Int hypre_StructSMGSetMaxLevel_long_dbl ( void   *smg_vdata, HYPRE_Int   max_level  );
HYPRE_Int hypre_SMGRelax_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SMGRelax_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SMGRelax_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
void *hypre_SMGRelaxCreate_flt  ( MPI_Comm comm );
void *hypre_SMGRelaxCreate_dbl  ( MPI_Comm comm );
void *hypre_SMGRelaxCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_SMGRelaxDestroy_flt  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroy_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroy_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyARem_flt  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyARem_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyARem_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyASol_flt  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyASol_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyASol_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyTempVec_flt  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyTempVec_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyTempVec_long_dbl  ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxSetBase_flt  ( void *relax_vdata, hypre_Index base_index,
                                  hypre_Index base_stride );
HYPRE_Int hypre_SMGRelaxSetBase_dbl  ( void *relax_vdata, hypre_Index base_index,
                                  hypre_Index base_stride );
HYPRE_Int hypre_SMGRelaxSetBase_long_dbl  ( void *relax_vdata, hypre_Index base_index,
                                  hypre_Index base_stride );
HYPRE_Int hypre_SMGRelaxSetMaxIter_flt  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGRelaxSetMaxIter_dbl  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGRelaxSetMaxIter_long_dbl  ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGRelaxSetMaxLevel_flt ( void *relax_vdata, HYPRE_Int   num_max_level );
HYPRE_Int hypre_SMGRelaxSetMaxLevel_dbl ( void *relax_vdata, HYPRE_Int   num_max_level );
HYPRE_Int hypre_SMGRelaxSetMaxLevel_long_dbl ( void *relax_vdata, HYPRE_Int   num_max_level );
HYPRE_Int hypre_SMGRelaxSetMemoryUse_flt  ( void *relax_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGRelaxSetMemoryUse_dbl  ( void *relax_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGRelaxSetMemoryUse_long_dbl  ( void *relax_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGRelaxSetNewMatrixStencil_flt  ( void *relax_vdata,
                                              hypre_StructStencil *diff_stencil );
HYPRE_Int hypre_SMGRelaxSetNewMatrixStencil_dbl  ( void *relax_vdata,
                                              hypre_StructStencil *diff_stencil );
HYPRE_Int hypre_SMGRelaxSetNewMatrixStencil_long_dbl  ( void *relax_vdata,
                                              hypre_StructStencil *diff_stencil );
HYPRE_Int hypre_SMGRelaxSetNumPostRelax_flt  ( void *relax_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGRelaxSetNumPostRelax_dbl  ( void *relax_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGRelaxSetNumPostRelax_long_dbl  ( void *relax_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGRelaxSetNumPreRelax_flt  ( void *relax_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGRelaxSetNumPreRelax_dbl  ( void *relax_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGRelaxSetNumPreRelax_long_dbl  ( void *relax_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGRelaxSetNumPreSpaces_flt  ( void *relax_vdata, HYPRE_Int num_pre_spaces );
HYPRE_Int hypre_SMGRelaxSetNumPreSpaces_dbl  ( void *relax_vdata, HYPRE_Int num_pre_spaces );
HYPRE_Int hypre_SMGRelaxSetNumPreSpaces_long_dbl  ( void *relax_vdata, HYPRE_Int num_pre_spaces );
HYPRE_Int hypre_SMGRelaxSetNumRegSpaces_flt  ( void *relax_vdata, HYPRE_Int num_reg_spaces );
HYPRE_Int hypre_SMGRelaxSetNumRegSpaces_dbl  ( void *relax_vdata, HYPRE_Int num_reg_spaces );
HYPRE_Int hypre_SMGRelaxSetNumRegSpaces_long_dbl  ( void *relax_vdata, HYPRE_Int num_reg_spaces );
HYPRE_Int hypre_SMGRelaxSetNumSpaces_flt  ( void *relax_vdata, HYPRE_Int num_spaces );
HYPRE_Int hypre_SMGRelaxSetNumSpaces_dbl  ( void *relax_vdata, HYPRE_Int num_spaces );
HYPRE_Int hypre_SMGRelaxSetNumSpaces_long_dbl  ( void *relax_vdata, HYPRE_Int num_spaces );
HYPRE_Int hypre_SMGRelaxSetPreSpaceRank_flt  ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int pre_space_rank );
HYPRE_Int hypre_SMGRelaxSetPreSpaceRank_dbl  ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int pre_space_rank );
HYPRE_Int hypre_SMGRelaxSetPreSpaceRank_long_dbl  ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int pre_space_rank );
HYPRE_Int hypre_SMGRelaxSetRegSpaceRank_flt  ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int reg_space_rank );
HYPRE_Int hypre_SMGRelaxSetRegSpaceRank_dbl  ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int reg_space_rank );
HYPRE_Int hypre_SMGRelaxSetRegSpaceRank_long_dbl  ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int reg_space_rank );
HYPRE_Int hypre_SMGRelaxSetSpace_flt  ( void *relax_vdata, HYPRE_Int i, HYPRE_Int space_index,
                                   HYPRE_Int space_stride );
HYPRE_Int hypre_SMGRelaxSetSpace_dbl  ( void *relax_vdata, HYPRE_Int i, HYPRE_Int space_index,
                                   HYPRE_Int space_stride );
HYPRE_Int hypre_SMGRelaxSetSpace_long_dbl  ( void *relax_vdata, HYPRE_Int i, HYPRE_Int space_index,
                                   HYPRE_Int space_stride );
HYPRE_Int hypre_SMGRelaxSetTempVec_flt  ( void *relax_vdata, hypre_StructVector *temp_vec );
HYPRE_Int hypre_SMGRelaxSetTempVec_dbl  ( void *relax_vdata, hypre_StructVector *temp_vec );
HYPRE_Int hypre_SMGRelaxSetTempVec_long_dbl  ( void *relax_vdata, hypre_StructVector *temp_vec );
HYPRE_Int hypre_SMGRelaxSetTol_flt  ( void *relax_vdata, hypre_float tol );
HYPRE_Int hypre_SMGRelaxSetTol_dbl  ( void *relax_vdata, hypre_double tol );
HYPRE_Int hypre_SMGRelaxSetTol_long_dbl  ( void *relax_vdata, hypre_long_double tol );
HYPRE_Int hypre_SMGRelaxSetup_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetup_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetup_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupARem_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupARem_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupARem_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupASol_flt  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupASol_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupASol_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupBaseBoxArray_flt  ( void *relax_vdata, hypre_StructMatrix *A,
                                            hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupBaseBoxArray_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                            hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupBaseBoxArray_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                            hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupTempVec_flt  ( void *relax_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupTempVec_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupTempVec_long_dbl  ( void *relax_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetZeroGuess_flt  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGRelaxSetZeroGuess_dbl  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGRelaxSetZeroGuess_long_dbl  ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGResidual_flt  ( void *residual_vdata, hypre_StructMatrix *A, hypre_StructVector *x,
                              hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidual_dbl  ( void *residual_vdata, hypre_StructMatrix *A, hypre_StructVector *x,
                              hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidual_long_dbl  ( void *residual_vdata, hypre_StructMatrix *A, hypre_StructVector *x,
                              hypre_StructVector *b, hypre_StructVector *r );
void *hypre_SMGResidualCreate_flt  ( void );
void *hypre_SMGResidualCreate_dbl  ( void );
void *hypre_SMGResidualCreate_long_dbl  ( void );
HYPRE_Int hypre_SMGResidualDestroy_flt  ( void *residual_vdata );
HYPRE_Int hypre_SMGResidualDestroy_dbl  ( void *residual_vdata );
HYPRE_Int hypre_SMGResidualDestroy_long_dbl  ( void *residual_vdata );
HYPRE_Int hypre_SMGResidualSetBase_flt  ( void *residual_vdata, hypre_Index base_index,
                                     hypre_Index base_stride );
HYPRE_Int hypre_SMGResidualSetBase_dbl  ( void *residual_vdata, hypre_Index base_index,
                                     hypre_Index base_stride );
HYPRE_Int hypre_SMGResidualSetBase_long_dbl  ( void *residual_vdata, hypre_Index base_index,
                                     hypre_Index base_stride );
HYPRE_Int hypre_SMGResidualSetup_flt  ( void *residual_vdata, hypre_StructMatrix *A,
                                   hypre_StructVector *x, hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidualSetup_dbl  ( void *residual_vdata, hypre_StructMatrix *A,
                                   hypre_StructVector *x, hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidualSetup_long_dbl  ( void *residual_vdata, hypre_StructMatrix *A,
                                   hypre_StructVector *x, hypre_StructVector *b, hypre_StructVector *r );
hypre_StructMatrix *hypre_SMGCreateInterpOp_flt  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                              HYPRE_Int cdir );
hypre_StructMatrix *hypre_SMGCreateInterpOp_dbl  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                              HYPRE_Int cdir );
hypre_StructMatrix *hypre_SMGCreateInterpOp_long_dbl  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                              HYPRE_Int cdir );
HYPRE_Int hypre_SMGSetupInterpOp_flt  ( void *relax_data, hypre_StructMatrix *A, hypre_StructVector *b,
                                   hypre_StructVector *x, hypre_StructMatrix *PT, HYPRE_Int cdir, hypre_Index cindex,
                                   hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SMGSetupInterpOp_dbl  ( void *relax_data, hypre_StructMatrix *A, hypre_StructVector *b,
                                   hypre_StructVector *x, hypre_StructMatrix *PT, HYPRE_Int cdir, hypre_Index cindex,
                                   hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SMGSetupInterpOp_long_dbl  ( void *relax_data, hypre_StructMatrix *A, hypre_StructVector *b,
                                   hypre_StructVector *x, hypre_StructMatrix *PT, HYPRE_Int cdir, hypre_Index cindex,
                                   hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SMGSetup_flt  ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SMGSetup_dbl  ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SMGSetup_long_dbl  ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
hypre_StructMatrix *hypre_SMGCreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                           hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
hypre_StructMatrix *hypre_SMGCreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                           hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
hypre_StructMatrix *hypre_SMGCreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                           hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMGSetupRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                hypre_StructMatrix *PT, hypre_StructMatrix *Ac, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMGSetupRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                hypre_StructMatrix *PT, hypre_StructMatrix *Ac, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMGSetupRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                hypre_StructMatrix *PT, hypre_StructMatrix *Ac, hypre_Index cindex, hypre_Index cstride );
hypre_StructMatrix *hypre_SMGCreateRestrictOp_flt  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                                HYPRE_Int cdir );
hypre_StructMatrix *hypre_SMGCreateRestrictOp_dbl  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                                HYPRE_Int cdir );
hypre_StructMatrix *hypre_SMGCreateRestrictOp_long_dbl  ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                                HYPRE_Int cdir );
HYPRE_Int hypre_SMGSetupRestrictOp_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *R,
                                     hypre_StructVector *temp_vec, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMGSetupRestrictOp_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *R,
                                     hypre_StructVector *temp_vec, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMGSetupRestrictOp_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *R,
                                     hypre_StructVector *temp_vec, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMGSolve_flt  ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SMGSolve_dbl  ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SMGSolve_long_dbl  ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SparseMSG2BuildRAPNoSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG2BuildRAPNoSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG2BuildRAPNoSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG2BuildRAPSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG2BuildRAPSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG2BuildRAPSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
hypre_StructMatrix *hypre_SparseMSG2CreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_SparseMSG2CreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_SparseMSG2CreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSG3BuildRAPNoSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG3BuildRAPNoSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG3BuildRAPNoSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG3BuildRAPSym_flt  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG3BuildRAPSym_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG3BuildRAPSym_long_dbl  ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
hypre_StructMatrix *hypre_SparseMSG3CreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_SparseMSG3CreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_SparseMSG3CreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSGFilter_flt  ( hypre_StructVector *visit, hypre_StructVector *e, HYPRE_Int lx,
                                  HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGFilter_dbl  ( hypre_StructVector *visit, hypre_StructVector *e, HYPRE_Int lx,
                                  HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGFilter_long_dbl  ( hypre_StructVector *visit, hypre_StructVector *e, HYPRE_Int lx,
                                  HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGFilterSetup_flt  ( hypre_StructMatrix *A, HYPRE_Int *num_grids, HYPRE_Int lx,
                                       HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump, hypre_StructVector *visitx, hypre_StructVector *visity,
                                       hypre_StructVector *visitz );
HYPRE_Int hypre_SparseMSGFilterSetup_dbl  ( hypre_StructMatrix *A, HYPRE_Int *num_grids, HYPRE_Int lx,
                                       HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump, hypre_StructVector *visitx, hypre_StructVector *visity,
                                       hypre_StructVector *visitz );
HYPRE_Int hypre_SparseMSGFilterSetup_long_dbl  ( hypre_StructMatrix *A, HYPRE_Int *num_grids, HYPRE_Int lx,
                                       HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump, hypre_StructVector *visitx, hypre_StructVector *visity,
                                       hypre_StructVector *visitz );
HYPRE_Int hypre_SparseMSGInterp_flt  ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                                  hypre_StructVector *e );
HYPRE_Int hypre_SparseMSGInterp_dbl  ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                                  hypre_StructVector *e );
HYPRE_Int hypre_SparseMSGInterp_long_dbl  ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                                  hypre_StructVector *e );
void *hypre_SparseMSGInterpCreate_flt  ( void );
void *hypre_SparseMSGInterpCreate_dbl  ( void );
void *hypre_SparseMSGInterpCreate_long_dbl  ( void );
HYPRE_Int hypre_SparseMSGInterpDestroy_flt  ( void *interp_vdata );
HYPRE_Int hypre_SparseMSGInterpDestroy_dbl  ( void *interp_vdata );
HYPRE_Int hypre_SparseMSGInterpDestroy_long_dbl  ( void *interp_vdata );
HYPRE_Int hypre_SparseMSGInterpSetup_flt  ( void *interp_vdata, hypre_StructMatrix *P,
                                       hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex, hypre_Index findex,
                                       hypre_Index stride, hypre_Index strideP );
HYPRE_Int hypre_SparseMSGInterpSetup_dbl  ( void *interp_vdata, hypre_StructMatrix *P,
                                       hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex, hypre_Index findex,
                                       hypre_Index stride, hypre_Index strideP );
HYPRE_Int hypre_SparseMSGInterpSetup_long_dbl  ( void *interp_vdata, hypre_StructMatrix *P,
                                       hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex, hypre_Index findex,
                                       hypre_Index stride, hypre_Index strideP );
void *hypre_SparseMSGCreate_flt  ( MPI_Comm comm );
void *hypre_SparseMSGCreate_dbl  ( MPI_Comm comm );
void *hypre_SparseMSGCreate_long_dbl  ( MPI_Comm comm );
HYPRE_Int hypre_SparseMSGDestroy_flt  ( void *smsg_vdata );
HYPRE_Int hypre_SparseMSGDestroy_dbl  ( void *smsg_vdata );
HYPRE_Int hypre_SparseMSGDestroy_long_dbl  ( void *smsg_vdata );
HYPRE_Int hypre_SparseMSGGetFinalRelativeResidualNorm_flt  ( void *smsg_vdata,
                                                        hypre_float *relative_residual_norm );
HYPRE_Int hypre_SparseMSGGetFinalRelativeResidualNorm_dbl  ( void *smsg_vdata,
                                                        hypre_double *relative_residual_norm );
HYPRE_Int hypre_SparseMSGGetFinalRelativeResidualNorm_long_dbl  ( void *smsg_vdata,
                                                        hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_SparseMSGGetNumIterations_flt  ( void *smsg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SparseMSGGetNumIterations_dbl  ( void *smsg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SparseMSGGetNumIterations_long_dbl  ( void *smsg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SparseMSGPrintLogging_flt  ( void *smsg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SparseMSGPrintLogging_dbl  ( void *smsg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SparseMSGPrintLogging_long_dbl  ( void *smsg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SparseMSGSetJacobiWeight_flt  ( void *smsg_vdata, hypre_float weight );
HYPRE_Int hypre_SparseMSGSetJacobiWeight_dbl  ( void *smsg_vdata, hypre_double weight );
HYPRE_Int hypre_SparseMSGSetJacobiWeight_long_dbl  ( void *smsg_vdata, hypre_long_double weight );
HYPRE_Int hypre_SparseMSGSetJump_flt  ( void *smsg_vdata, HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGSetJump_dbl  ( void *smsg_vdata, HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGSetJump_long_dbl  ( void *smsg_vdata, HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGSetLogging_flt  ( void *smsg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SparseMSGSetLogging_dbl  ( void *smsg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SparseMSGSetLogging_long_dbl  ( void *smsg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SparseMSGSetMaxIter_flt  ( void *smsg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SparseMSGSetMaxIter_dbl  ( void *smsg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SparseMSGSetMaxIter_long_dbl  ( void *smsg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SparseMSGSetNumFineRelax_flt  ( void *smsg_vdata, HYPRE_Int num_fine_relax );
HYPRE_Int hypre_SparseMSGSetNumFineRelax_dbl  ( void *smsg_vdata, HYPRE_Int num_fine_relax );
HYPRE_Int hypre_SparseMSGSetNumFineRelax_long_dbl  ( void *smsg_vdata, HYPRE_Int num_fine_relax );
HYPRE_Int hypre_SparseMSGSetNumPostRelax_flt  ( void *smsg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SparseMSGSetNumPostRelax_dbl  ( void *smsg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SparseMSGSetNumPostRelax_long_dbl  ( void *smsg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SparseMSGSetNumPreRelax_flt  ( void *smsg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SparseMSGSetNumPreRelax_dbl  ( void *smsg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SparseMSGSetNumPreRelax_long_dbl  ( void *smsg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SparseMSGSetPrintLevel_flt  ( void *smsg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SparseMSGSetPrintLevel_dbl  ( void *smsg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SparseMSGSetPrintLevel_long_dbl  ( void *smsg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SparseMSGSetRelaxType_flt  ( void *smsg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SparseMSGSetRelaxType_dbl  ( void *smsg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SparseMSGSetRelaxType_long_dbl  ( void *smsg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SparseMSGSetRelChange_flt  ( void *smsg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SparseMSGSetRelChange_dbl  ( void *smsg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SparseMSGSetRelChange_long_dbl  ( void *smsg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SparseMSGSetTol_flt  ( void *smsg_vdata, hypre_float tol );
HYPRE_Int hypre_SparseMSGSetTol_dbl  ( void *smsg_vdata, hypre_double tol );
HYPRE_Int hypre_SparseMSGSetTol_long_dbl  ( void *smsg_vdata, hypre_long_double tol );
HYPRE_Int hypre_SparseMSGSetZeroGuess_flt  ( void *smsg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SparseMSGSetZeroGuess_dbl  ( void *smsg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SparseMSGSetZeroGuess_long_dbl  ( void *smsg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SparseMSGRestrict_flt  ( void *restrict_vdata, hypre_StructMatrix *R,
                                    hypre_StructVector *r, hypre_StructVector *rc );
HYPRE_Int hypre_SparseMSGRestrict_dbl  ( void *restrict_vdata, hypre_StructMatrix *R,
                                    hypre_StructVector *r, hypre_StructVector *rc );
HYPRE_Int hypre_SparseMSGRestrict_long_dbl  ( void *restrict_vdata, hypre_StructMatrix *R,
                                    hypre_StructVector *r, hypre_StructVector *rc );
void *hypre_SparseMSGRestrictCreate_flt  ( void );
void *hypre_SparseMSGRestrictCreate_dbl  ( void );
void *hypre_SparseMSGRestrictCreate_long_dbl  ( void );
HYPRE_Int hypre_SparseMSGRestrictDestroy_flt  ( void *restrict_vdata );
HYPRE_Int hypre_SparseMSGRestrictDestroy_dbl  ( void *restrict_vdata );
HYPRE_Int hypre_SparseMSGRestrictDestroy_long_dbl  ( void *restrict_vdata );
HYPRE_Int hypre_SparseMSGRestrictSetup_flt  ( void *restrict_vdata, hypre_StructMatrix *R,
                                         hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex, hypre_Index findex,
                                         hypre_Index stride, hypre_Index strideR );
HYPRE_Int hypre_SparseMSGRestrictSetup_dbl  ( void *restrict_vdata, hypre_StructMatrix *R,
                                         hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex, hypre_Index findex,
                                         hypre_Index stride, hypre_Index strideR );
HYPRE_Int hypre_SparseMSGRestrictSetup_long_dbl  ( void *restrict_vdata, hypre_StructMatrix *R,
                                         hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex, hypre_Index findex,
                                         hypre_Index stride, hypre_Index strideR );
HYPRE_Int hypre_SparseMSGSetup_flt  ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );
HYPRE_Int hypre_SparseMSGSetup_dbl  ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );
HYPRE_Int hypre_SparseMSGSetup_long_dbl  ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );
hypre_StructMatrix *hypre_SparseMSGCreateRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                 hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_SparseMSGCreateRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                 hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
hypre_StructMatrix *hypre_SparseMSGCreateRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                 hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSGSetupRAPOp_flt  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                      hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                      hypre_Index stridePR, hypre_StructMatrix *Ac );
HYPRE_Int hypre_SparseMSGSetupRAPOp_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                      hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                      hypre_Index stridePR, hypre_StructMatrix *Ac );
HYPRE_Int hypre_SparseMSGSetupRAPOp_long_dbl  ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                      hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                      hypre_Index stridePR, hypre_StructMatrix *Ac );
HYPRE_Int hypre_SparseMSGSolve_flt  ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );
HYPRE_Int hypre_SparseMSGSolve_dbl  ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );
HYPRE_Int hypre_SparseMSGSolve_long_dbl  ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );

#endif

#endif
