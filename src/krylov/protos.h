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
HYPRE_Int
hypre_BiCGSTABSetPrecond ( void *bicgstab_vdata,
                           hypre_KrylovPtrToPrecond precond,
                           hypre_KrylovPtrToPrecondSetup precond_setup,
                           void *precond_data );
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
HYPRE_Int
hypre_CGNRSetPrecond ( void *cgnr_vdata,
                       hypre_KrylovPtrToPrecond precond,
                       hypre_KrylovPtrToPrecondT precondT,
                       hypre_KrylovPtrToPrecondSetup precond_setup,
                       void *precond_data );
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
HYPRE_Int
hypre_GMRESSetPrecond ( void *gmres_vdata,
                        hypre_KrylovPtrToPrecond precond,
                        hypre_KrylovPtrToPrecondSetup precond_setup,
                        void *precond_data );
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
HYPRE_Int
hypre_COGMRESSetPrecond ( void *gmres_vdata,
                          hypre_KrylovPtrToPrecond precond,
                          hypre_KrylovPtrToPrecondSetup precond_setup,
                          void *precond_data );
HYPRE_Int hypre_COGMRESGetPrecond ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_COGMRESSetPrintLevel ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESGetPrintLevel ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESSetLogging ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESGetLogging ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetNumIterations ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_COGMRESGetConverged ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_COGMRESGetFinalRelativeResidualNorm ( void *gmres_vdata,
                                                      HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_COGMRESSetModifyPC ( void *cogmres_vdata, hypre_KrylovPtrToModifyPC modify_pc );

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
HYPRE_Int
hypre_FlexGMRESSetPrecond ( void *fgmres_vdata,
                            hypre_KrylovPtrToPrecond precond,
                            hypre_KrylovPtrToPrecondSetup precond_setup,
                            void *precond_data );
HYPRE_Int hypre_FlexGMRESGetPrecond ( void *fgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_FlexGMRESSetPrintLevel ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetPrintLevel ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESSetLogging ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESGetLogging ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetNumIterations ( void *fgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FlexGMRESGetConverged ( void *fgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_FlexGMRESGetFinalRelativeResidualNorm ( void *fgmres_vdata,
                                                        HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_FlexGMRESSetModifyPC ( void *fgmres_vdata, hypre_KrylovPtrToModifyPC modify_pc );
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
HYPRE_Int
hypre_LGMRESSetPrecond ( void *lgmres_vdata,
                         hypre_KrylovPtrToPrecond precond,
                         hypre_KrylovPtrToPrecondSetup precond_setup,
                         void *precond_data );
HYPRE_Int hypre_LGMRESGetPrecond ( void *lgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_LGMRESSetPrintLevel ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetPrintLevel ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESSetLogging ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESGetLogging ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetNumIterations ( void *lgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_LGMRESGetConverged ( void *lgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_LGMRESGetFinalRelativeResidualNorm ( void *lgmres_vdata,
                                                     HYPRE_Real *relative_residual_norm );

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
HYPRE_Int
hypre_PCGSetPrecond ( void *pcg_vdata,
                      hypre_KrylovPtrToPrecond precond,
                      hypre_KrylovPtrToPrecondSetup precond_setup,
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

