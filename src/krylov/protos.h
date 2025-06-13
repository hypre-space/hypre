/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* bicgstab.c */
void *hypre_BiCGSTABCreate ( hypre_BiCGSTABFunctions *bicgstab_functions );
HYPRE_Int hypre_BiCGSTABDestroy ( void *bicgstab_vdata );
HYPRE_Int hypre_BiCGSTABSetup ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSolve ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSetTol ( void *bicgstab_vdata, HYPRE_Real tol );
HYPRE_Int hypre_BiCGSTABSetAbsoluteTol ( void *bicgstab_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_BiCGSTABSetConvergenceFactorTol ( void *bicgstab_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_BiCGSTABSetMinIter ( void *bicgstab_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_BiCGSTABSetMaxIter ( void *bicgstab_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_BiCGSTABSetStopCrit ( void *bicgstab_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_BiCGSTABSetPrecond ( void *bicgstab_vdata, HYPRE_Int (*precond )(void*, void*,
                                                                                 void*,
                                                                                 void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_BiCGSTABGetPrecond ( void *bicgstab_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_BiCGSTABSetPrecondMatrix( void  *bicgstab_vdata,  void  *precond_matrix );
HYPRE_Int hypre_BiCGSTABGetPrecondMatrix( void  *bicgstab_vdata,  HYPRE_Matrix *precond_matrix_ptr ) ;
HYPRE_Int hypre_BiCGSTABSetLogging ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetHybrid ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetPrintLevel ( void *bicgstab_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_BiCGSTABGetConverged ( void *bicgstab_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_BiCGSTABGetNumIterations ( void *bicgstab_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BiCGSTABGetFinalRelativeResidualNorm ( void *bicgstab_vdata,
                                                       HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_BiCGSTABGetResidual ( void *bicgstab_vdata, void **residual );

/* cgnr.c */
void *hypre_CGNRCreate ( hypre_CGNRFunctions *cgnr_functions );
HYPRE_Int hypre_CGNRDestroy ( void *cgnr_vdata );
HYPRE_Int hypre_CGNRSetup ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSolve ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSetTol ( void *cgnr_vdata, HYPRE_Real tol );
HYPRE_Int hypre_CGNRSetMinIter ( void *cgnr_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_CGNRSetMaxIter ( void *cgnr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_CGNRSetStopCrit ( void *cgnr_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_CGNRSetPrecond ( void *cgnr_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                         void*),
                                 HYPRE_Int (*precondT )(void*, void*, void*, void*), HYPRE_Int (*precond_setup )(void*, void*, void*,
                                       void*), void *precond_data );
HYPRE_Int hypre_CGNRGetPrecond ( void *cgnr_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_CGNRSetLogging ( void *cgnr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_CGNRGetNumIterations ( void *cgnr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_CGNRGetFinalRelativeResidualNorm ( void *cgnr_vdata,
                                                   HYPRE_Real *relative_residual_norm );

/* gmres.c */
void *hypre_GMRESCreate ( hypre_GMRESFunctions *gmres_functions );
HYPRE_Int hypre_GMRESDestroy ( void *gmres_vdata );
HYPRE_Int hypre_GMRESGetResidual ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_GMRESSetup ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSolve ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSetKDim ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_GMRESGetKDim ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_GMRESSetTol ( void *gmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_GMRESGetTol ( void *gmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_GMRESSetAbsoluteTol ( void *gmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_GMRESGetAbsoluteTol ( void *gmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_GMRESSetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_GMRESGetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_GMRESSetMinIter ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_GMRESGetMinIter ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_GMRESSetMaxIter ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_GMRESGetMaxIter ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_GMRESSetRelChange ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_GMRESGetRelChange ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_GMRESSetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_GMRESGetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_GMRESSetStopCrit ( void *gmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_GMRESGetStopCrit ( void *gmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_GMRESSetPrecond ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                           void*),
HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_GMRESGetPrecond ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_GMRESSetPrecondMatrix ( void *gmres_vdata, void *precond_matrix );
HYPRE_Int hypre_GMRESGetPrecondMatrix ( void *gmres_vdata, HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_GMRESSetPrintLevel ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESGetPrintLevel ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESSetLogging ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESGetLogging ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESSetHybrid ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESGetNumIterations ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_GMRESGetConverged ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_GMRESGetFinalRelativeResidualNorm ( void *gmres_vdata,
                                                    HYPRE_Real *relative_residual_norm );

/* cogmres.c */
void *hypre_COGMRESCreate ( hypre_COGMRESFunctions *gmres_functions );
HYPRE_Int hypre_COGMRESDestroy ( void *gmres_vdata );
HYPRE_Int hypre_COGMRESGetResidual ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_COGMRESSetup ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSolve ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSetKDim ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_COGMRESGetKDim ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_COGMRESSetUnroll ( void *gmres_vdata, HYPRE_Int unroll );
HYPRE_Int hypre_COGMRESGetUnroll ( void *gmres_vdata, HYPRE_Int *unroll );
HYPRE_Int hypre_COGMRESSetCGS ( void *gmres_vdata, HYPRE_Int cgs );
HYPRE_Int hypre_COGMRESGetCGS ( void *gmres_vdata, HYPRE_Int *cgs );
HYPRE_Int hypre_COGMRESSetTol ( void *gmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_COGMRESGetTol ( void *gmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_COGMRESSetAbsoluteTol ( void *gmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_COGMRESGetAbsoluteTol ( void *gmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_COGMRESSetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_COGMRESGetConvergenceFactorTol ( void *gmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_COGMRESSetMinIter ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_COGMRESGetMinIter ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_COGMRESSetMaxIter ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_COGMRESGetMaxIter ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_COGMRESSetRelChange ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_COGMRESGetRelChange ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_COGMRESSetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_COGMRESGetSkipRealResidualCheck ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_COGMRESSetPrecond ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_COGMRESGetPrecond ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_COGMRESSetPrintLevel ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESGetPrintLevel ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESSetLogging ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESGetLogging ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetNumIterations ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_COGMRESGetConverged ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_COGMRESGetFinalRelativeResidualNorm ( void *gmres_vdata,
                                                      HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_COGMRESSetModifyPC ( void *fgmres_vdata, HYPRE_Int (*modify_pc )(void *precond_data,
                                                                                 HYPRE_Int iteration, HYPRE_Real rel_residual_norm));



/* flexgmres.c */
void *hypre_FlexGMRESCreate ( hypre_FlexGMRESFunctions *fgmres_functions );
HYPRE_Int hypre_FlexGMRESDestroy ( void *fgmres_vdata );
HYPRE_Int hypre_FlexGMRESGetResidual ( void *fgmres_vdata, void **residual );
HYPRE_Int hypre_FlexGMRESSetup ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSolve ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSetKDim ( void *fgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_FlexGMRESGetKDim ( void *fgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_FlexGMRESSetTol ( void *fgmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_FlexGMRESGetTol ( void *fgmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_FlexGMRESSetAbsoluteTol ( void *fgmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_FlexGMRESGetAbsoluteTol ( void *fgmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_FlexGMRESSetConvergenceFactorTol ( void *fgmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_FlexGMRESGetConvergenceFactorTol ( void *fgmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_FlexGMRESSetMinIter ( void *fgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_FlexGMRESGetMinIter ( void *fgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_FlexGMRESSetMaxIter ( void *fgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_FlexGMRESGetMaxIter ( void *fgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_FlexGMRESSetStopCrit ( void *fgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_FlexGMRESGetStopCrit ( void *fgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_FlexGMRESSetPrecond ( void *fgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                                void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_FlexGMRESGetPrecond ( void *fgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_FlexGMRESSetPrintLevel ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetPrintLevel ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESSetLogging ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetLogging ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetNumIterations ( void *fgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FlexGMRESGetConverged ( void *fgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_FlexGMRESGetFinalRelativeResidualNorm ( void *fgmres_vdata,
                                                        HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_FlexGMRESSetModifyPC ( void *fgmres_vdata,
                                       HYPRE_Int (*modify_pc )(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm));
HYPRE_Int hypre_FlexGMRESModifyPCDefault ( void *precond_data, HYPRE_Int iteration,
                                           HYPRE_Real rel_residual_norm );

/* lgmres.c */
void *hypre_LGMRESCreate ( hypre_LGMRESFunctions *lgmres_functions );
HYPRE_Int hypre_LGMRESDestroy ( void *lgmres_vdata );
HYPRE_Int hypre_LGMRESGetResidual ( void *lgmres_vdata, void **residual );
HYPRE_Int hypre_LGMRESSetup ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSolve ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSetKDim ( void *lgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_LGMRESGetKDim ( void *lgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_LGMRESSetAugDim ( void *lgmres_vdata, HYPRE_Int aug_dim );
HYPRE_Int hypre_LGMRESGetAugDim ( void *lgmres_vdata, HYPRE_Int *aug_dim );
HYPRE_Int hypre_LGMRESSetTol ( void *lgmres_vdata, HYPRE_Real tol );
HYPRE_Int hypre_LGMRESGetTol ( void *lgmres_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_LGMRESSetAbsoluteTol ( void *lgmres_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_LGMRESGetAbsoluteTol ( void *lgmres_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_LGMRESSetConvergenceFactorTol ( void *lgmres_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_LGMRESGetConvergenceFactorTol ( void *lgmres_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_LGMRESSetMinIter ( void *lgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_LGMRESGetMinIter ( void *lgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_LGMRESSetMaxIter ( void *lgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_LGMRESGetMaxIter ( void *lgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_LGMRESSetStopCrit ( void *lgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_LGMRESGetStopCrit ( void *lgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_LGMRESSetPrecond ( void *lgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_LGMRESGetPrecond ( void *lgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_LGMRESSetPrintLevel ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetPrintLevel ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESSetLogging ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetLogging ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetNumIterations ( void *lgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_LGMRESGetConverged ( void *lgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_LGMRESGetFinalRelativeResidualNorm ( void *lgmres_vdata,
                                                     HYPRE_Real *relative_residual_norm );

/* HYPRE_bicgstab.c */
HYPRE_Int HYPRE_BiCGSTABDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BiCGSTABSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_BiCGSTABSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_BiCGSTABSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_BiCGSTABSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BiCGSTABSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BiCGSTABSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_BiCGSTABSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                     HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_BiCGSTABGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_BiCGSTABSetPrecondMatrix ( HYPRE_Solver solver , HYPRE_Matrix precond_matrix );
HYPRE_Int HYPRE_BiCGSTABGetPrecondMatrix ( HYPRE_Solver solver , HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_BiCGSTABSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BiCGSTABSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BiCGSTABGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BiCGSTABGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_BiCGSTABGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_cgnr.c */
HYPRE_Int HYPRE_CGNRDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CGNRSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_CGNRSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_CGNRSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_CGNRSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_CGNRSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                 HYPRE_PtrToSolverFcn precondT, HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_CGNRGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CGNRSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_CGNRGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CGNRGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );

/* HYPRE_gmres.c */
HYPRE_Int HYPRE_GMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_GMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_GMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_GMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_GMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_GMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_GMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_GMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_GMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_GMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_GMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_GMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_GMRESSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_GMRESGetStopCrit ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_GMRESSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_GMRESGetRelChange ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_GMRESSetSkipRealResidualCheck ( HYPRE_Solver solver, HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_GMRESGetSkipRealResidualCheck ( HYPRE_Solver solver, HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_GMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                  HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_GMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_GMRESSetPrecondMatrix ( HYPRE_Solver solver , HYPRE_Matrix precond_matrix );
HYPRE_Int HYPRE_GMRESGetPrecondMatrix ( HYPRE_Solver solver , HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_GMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_GMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_GMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_GMRESGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_cogmres.c */
HYPRE_Int HYPRE_COGMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_COGMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_COGMRESSetUnroll ( HYPRE_Solver solver, HYPRE_Int unroll );
HYPRE_Int HYPRE_COGMRESGetUnroll ( HYPRE_Solver solver, HYPRE_Int *unroll );
HYPRE_Int HYPRE_COGMRESSetCGS ( HYPRE_Solver solver, HYPRE_Int cgs );
HYPRE_Int HYPRE_COGMRESGetCGS ( HYPRE_Solver solver, HYPRE_Int *cgs );
HYPRE_Int HYPRE_COGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_COGMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_COGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_COGMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_COGMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_COGMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_COGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_COGMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_COGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_COGMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_COGMRESSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_COGMRESGetRelChange ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_COGMRESSetSkipRealResidualCheck ( HYPRE_Solver solver,
                                                  HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_COGMRESGetSkipRealResidualCheck ( HYPRE_Solver solver,
                                                  HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_COGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                    HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_COGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_COGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_COGMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_COGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_COGMRESGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_flexgmres.c */
HYPRE_Int HYPRE_FlexGMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_FlexGMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_FlexGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_FlexGMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_FlexGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_FlexGMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_FlexGMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_FlexGMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_FlexGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_FlexGMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_FlexGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_FlexGMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_FlexGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                      HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_FlexGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_FlexGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_FlexGMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_FlexGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_FlexGMRESGetResidual ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_FlexGMRESSetModifyPC ( HYPRE_Solver solver, HYPRE_Int (*modify_pc )(HYPRE_Solver,
                                                                                    HYPRE_Int, HYPRE_Real ));

/* HYPRE_lgmres.c */
HYPRE_Int HYPRE_LGMRESSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_LGMRESGetKDim ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_LGMRESSetAugDim ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_LGMRESGetAugDim ( HYPRE_Solver solver, HYPRE_Int *aug_dim );
HYPRE_Int HYPRE_LGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_LGMRESGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_LGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_LGMRESGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_LGMRESSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_LGMRESGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_LGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_LGMRESGetMinIter ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_LGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_LGMRESGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_LGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                   HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_LGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_LGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESGetLogging ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_LGMRESGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_LGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_LGMRESGetResidual ( HYPRE_Solver solver, void *residual );

/* HYPRE_pcg.c */
HYPRE_Int HYPRE_PCGSetup ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSolve ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_PCGGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_PCGGetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real *a_tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTolFactor ( HYPRE_Solver solver, HYPRE_Real abstolf );
HYPRE_Int HYPRE_PCGGetAbsoluteTolFactor ( HYPRE_Solver solver, HYPRE_Real *abstolf );
HYPRE_Int HYPRE_PCGSetResidualTol ( HYPRE_Solver solver, HYPRE_Real rtol );
HYPRE_Int HYPRE_PCGGetResidualTol ( HYPRE_Solver solver, HYPRE_Real *rtol );
HYPRE_Int HYPRE_PCGSetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_PCGGetConvergenceFactorTol ( HYPRE_Solver solver, HYPRE_Real *cf_tol );
HYPRE_Int HYPRE_PCGSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_PCGGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_PCGSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_PCGGetStopCrit ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_PCGSetTwoNorm ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_PCGGetTwoNorm ( HYPRE_Solver solver, HYPRE_Int *two_norm );
HYPRE_Int HYPRE_PCGSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_PCGGetRelChange ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_PCGSetRecomputeResidual ( HYPRE_Solver solver, HYPRE_Int recompute_residual );
HYPRE_Int HYPRE_PCGGetRecomputeResidual ( HYPRE_Solver solver, HYPRE_Int *recompute_residual );
HYPRE_Int HYPRE_PCGSetRecomputeResidualP ( HYPRE_Solver solver, HYPRE_Int recompute_residual_p );
HYPRE_Int HYPRE_PCGGetRecomputeResidualP ( HYPRE_Solver solver, HYPRE_Int *recompute_residual_p );
HYPRE_Int HYPRE_PCGSetSkipBreak ( HYPRE_Solver solver, HYPRE_Int skip_break );
HYPRE_Int HYPRE_PCGGetSkipBreak ( HYPRE_Solver solver, HYPRE_Int *skip_break );
HYPRE_Int HYPRE_PCGSetFlex ( HYPRE_Solver solver, HYPRE_Int flex );
HYPRE_Int HYPRE_PCGGetFlex ( HYPRE_Solver solver, HYPRE_Int *flex );
HYPRE_Int HYPRE_PCGSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_PCGSetPrecondMatrix ( HYPRE_Solver solver , HYPRE_Matrix precond_matrix );
HYPRE_Int HYPRE_PCGSetPreconditioner ( HYPRE_Solver solver, HYPRE_Solver precond );
HYPRE_Int HYPRE_PCGGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_PCGGetPrecondMatrix ( HYPRE_Solver solver , HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_PCGSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_PCGGetLogging ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_PCGSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_PCGGetConverged ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_PCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_PCGGetResidual ( HYPRE_Solver solver, void *residual );

/* pcg.c */
void *hypre_PCGCreate ( hypre_PCGFunctions *pcg_functions );
HYPRE_Int hypre_PCGDestroy ( void *pcg_vdata );
HYPRE_Int hypre_PCGGetResidual ( void *pcg_vdata, void **residual );
HYPRE_Int hypre_PCGSetup ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSolve ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSetTol ( void *pcg_vdata, HYPRE_Real tol );
HYPRE_Int hypre_PCGGetTol ( void *pcg_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_PCGSetAbsoluteTol ( void *pcg_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_PCGGetAbsoluteTol ( void *pcg_vdata, HYPRE_Real *a_tol );
HYPRE_Int hypre_PCGSetAbsoluteTolFactor ( void *pcg_vdata, HYPRE_Real atolf );
HYPRE_Int hypre_PCGGetAbsoluteTolFactor ( void *pcg_vdata, HYPRE_Real *atolf );
HYPRE_Int hypre_PCGSetResidualTol ( void *pcg_vdata, HYPRE_Real rtol );
HYPRE_Int hypre_PCGGetResidualTol ( void *pcg_vdata, HYPRE_Real *rtol );
HYPRE_Int hypre_PCGSetConvergenceFactorTol ( void *pcg_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_PCGGetConvergenceFactorTol ( void *pcg_vdata, HYPRE_Real *cf_tol );
HYPRE_Int hypre_PCGSetMaxIter ( void *pcg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PCGGetMaxIter ( void *pcg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PCGSetTwoNorm ( void *pcg_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_PCGGetTwoNorm ( void *pcg_vdata, HYPRE_Int *two_norm );
HYPRE_Int hypre_PCGSetRelChange ( void *pcg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PCGGetRelChange ( void *pcg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PCGSetRecomputeResidual ( void *pcg_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_PCGGetRecomputeResidual ( void *pcg_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_PCGSetRecomputeResidualP ( void *pcg_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_PCGGetRecomputeResidualP ( void *pcg_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_PCGSetStopCrit ( void *pcg_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_PCGGetStopCrit ( void *pcg_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_PCGSetSkipBreak ( void *pcg_vdata, HYPRE_Int skip_break );
HYPRE_Int hypre_PCGGetSkipBreak ( void *pcg_vdata, HYPRE_Int *skip_break );
HYPRE_Int hypre_PCGSetFlex ( void *pcg_vdata, HYPRE_Int flex );
HYPRE_Int hypre_PCGGetFlex ( void *pcg_vdata, HYPRE_Int *flex );
HYPRE_Int hypre_PCGGetPrecond ( void *pcg_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_PCGGetPrecondMatrix( void  *pcg_vdata,  HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_PCGSetPrecond ( void *pcg_vdata,
                                HYPRE_Int (*precond )(void*, void*, void*, void*),
                                HYPRE_Int (*precond_setup )(void*, void*, void*, void*),
                                void *precond_data );
HYPRE_Int hypre_PCGSetPrecondMatrix( void  *pcg_vdata,  void  *precond_matrix );
HYPRE_Int hypre_PCGSetPreconditioner ( void *pcg_vdata, void *precond_data );
HYPRE_Int hypre_PCGSetPrintLevel ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGGetPrintLevel ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGSetLogging ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGGetLogging ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGSetHybrid ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGGetNumIterations ( void *pcg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PCGGetConverged ( void *pcg_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_PCGPrintLogging ( void *pcg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PCGGetFinalRelativeResidualNorm ( void *pcg_vdata,
                                                  HYPRE_Real *relative_residual_norm );
