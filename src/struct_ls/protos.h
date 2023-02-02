/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* coarsen.c */
HYPRE_Int hypre_StructMapFineToCoarse ( hypre_Index findex, hypre_Index index, hypre_Index stride,
                                        hypre_Index cindex );
HYPRE_Int hypre_StructMapCoarseToFine ( hypre_Index cindex, hypre_Index index, hypre_Index stride,
                                        hypre_Index findex );
HYPRE_Int hypre_StructCoarsen ( hypre_StructGrid *fgrid, hypre_Index index, hypre_Index stride,
                                HYPRE_Int prune, hypre_StructGrid **cgrid_ptr );

/* cyclic_reduction.c */
void *hypre_CyclicReductionCreate ( MPI_Comm comm );
hypre_StructMatrix *hypre_CycRedCreateCoarseOp ( hypre_StructMatrix *A,
                                                 hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_CycRedSetupCoarseOp ( hypre_StructMatrix *A, hypre_StructMatrix *Ac,
                                      hypre_Index cindex, hypre_Index cstride, HYPRE_Int cdir );
HYPRE_Int hypre_CyclicReductionSetup ( void *cyc_red_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_CyclicReduction ( void *cyc_red_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_CyclicReductionSetCDir ( void *cyc_red_vdata, HYPRE_Int cdir );
HYPRE_Int hypre_CyclicReductionSetBase ( void *cyc_red_vdata, hypre_Index base_index,
                                         hypre_Index base_stride );
HYPRE_Int hypre_CyclicReductionDestroy ( void *cyc_red_vdata );
HYPRE_Int hypre_CyclicReductionSetMaxLevel( void   *cyc_red_vdata, HYPRE_Int   max_level  );

/* general.c */
HYPRE_Int hypre_Log2 ( HYPRE_Int p );

/* hybrid.c */
void *hypre_HybridCreate ( MPI_Comm comm );
HYPRE_Int hypre_HybridDestroy ( void *hybrid_vdata );
HYPRE_Int hypre_HybridSetTol ( void *hybrid_vdata, HYPRE_Real tol );
HYPRE_Int hypre_HybridSetConvergenceTol ( void *hybrid_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_HybridSetDSCGMaxIter ( void *hybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_HybridSetPCGMaxIter ( void *hybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_HybridSetPCGAbsoluteTolFactor ( void *hybrid_vdata, HYPRE_Real pcg_atolf );
HYPRE_Int hypre_HybridSetTwoNorm ( void *hybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_HybridSetStopCrit ( void *hybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_HybridSetRelChange ( void *hybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_HybridSetSolverType ( void *hybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_HybridSetRecomputeResidual( void *hybrid_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_HybridGetRecomputeResidual( void *hybrid_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_HybridSetRecomputeResidualP( void *hybrid_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_HybridGetRecomputeResidualP( void *hybrid_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_HybridSetKDim ( void *hybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_HybridSetPrecond ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                    void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_HybridSetLogging ( void *hybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_HybridSetPrintLevel ( void *hybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_HybridGetNumIterations ( void *hybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_HybridGetDSCGNumIterations ( void *hybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_HybridGetPCGNumIterations ( void *hybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_HybridGetFinalRelativeResidualNorm ( void *hybrid_vdata,
                                                     HYPRE_Real *final_rel_res_norm );
HYPRE_Int hypre_HybridSetup ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_HybridSolve ( void *hybrid_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );

/* HYPRE_struct_int.c */
HYPRE_Int hypre_StructVectorSetRandomValues ( hypre_StructVector *vector, HYPRE_Int seed );
HYPRE_Int hypre_StructSetRandomValues ( void *v, HYPRE_Int seed );

/* HYPRE_struct_pfmg.c */
HYPRE_Int hypre_PFMGSetDeviceLevel( void *pfmg_vdata, HYPRE_Int   device_level  );

/* jacobi.c */
void *hypre_JacobiCreate ( MPI_Comm comm );
HYPRE_Int hypre_JacobiDestroy ( void *jacobi_vdata );
HYPRE_Int hypre_JacobiSetup ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_JacobiSolve ( void *jacobi_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                              hypre_StructVector *x );
HYPRE_Int hypre_JacobiSetTol ( void *jacobi_vdata, HYPRE_Real tol );
HYPRE_Int hypre_JacobiGetTol ( void *jacobi_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_JacobiSetMaxIter ( void *jacobi_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_JacobiGetMaxIter ( void *jacobi_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_JacobiSetZeroGuess ( void *jacobi_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_JacobiGetZeroGuess ( void *jacobi_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_JacobiGetNumIterations ( void *jacobi_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_JacobiSetTempVec ( void *jacobi_vdata, hypre_StructVector *t );
HYPRE_Int hypre_JacobiGetFinalRelativeResidualNorm ( void *jacobi_vdata, HYPRE_Real *norm );

/* pcg_struct.c */
void *hypre_StructKrylovCAlloc ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
HYPRE_Int hypre_StructKrylovFree ( void *ptr );
void *hypre_StructKrylovCreateVector ( void *vvector );
void *hypre_StructKrylovCreateVectorArray ( HYPRE_Int n, void *vvector );
HYPRE_Int hypre_StructKrylovDestroyVector ( void *vvector );
void *hypre_StructKrylovMatvecCreate ( void *A, void *x );
HYPRE_Int hypre_StructKrylovMatvec ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x,
                                     HYPRE_Complex beta, void *y );
HYPRE_Int hypre_StructKrylovMatvecDestroy ( void *matvec_data );
HYPRE_Real hypre_StructKrylovInnerProd ( void *x, void *y );
HYPRE_Int hypre_StructKrylovCopyVector ( void *x, void *y );
HYPRE_Int hypre_StructKrylovClearVector ( void *x );
HYPRE_Int hypre_StructKrylovScaleVector ( HYPRE_Complex alpha, void *x );
HYPRE_Int hypre_StructKrylovAxpy ( HYPRE_Complex alpha, void *x, void *y );
HYPRE_Int hypre_StructKrylovIdentitySetup ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_StructKrylovIdentity ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_StructKrylovCommInfo ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );

/* pfmg2_setup_rap.c */
hypre_StructMatrix *hypre_PFMG2CreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_PFMG2BuildRAPSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                   hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                   hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                     hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                     hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );

/* pfmg3_setup_rap.c */
hypre_StructMatrix *hypre_PFMG3CreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                             hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_PFMG3BuildRAPSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                   hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                   hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                    hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                    hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0 ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1 ( HYPRE_Int ci, HYPRE_Int fi,
                                                      hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir,
                                                      hypre_Index cindex, hypre_Index cstride, hypre_StructMatrix *RAP );

/* pfmg.c */
void *hypre_PFMGCreate ( MPI_Comm comm );
HYPRE_Int hypre_PFMGDestroy ( void *pfmg_vdata );
HYPRE_Int hypre_PFMGSetTol ( void *pfmg_vdata, HYPRE_Real tol );
HYPRE_Int hypre_PFMGGetTol ( void *pfmg_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_PFMGSetMaxIter ( void *pfmg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGGetMaxIter ( void *pfmg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PFMGSetMaxLevels ( void *pfmg_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_PFMGGetMaxLevels ( void *pfmg_vdata, HYPRE_Int *max_levels );
HYPRE_Int hypre_PFMGSetRelChange ( void *pfmg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PFMGGetRelChange ( void *pfmg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PFMGSetZeroGuess ( void *pfmg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGGetZeroGuess ( void *pfmg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PFMGSetRelaxType ( void *pfmg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGGetRelaxType ( void *pfmg_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_PFMGSetJacobiWeight ( void *pfmg_vdata, HYPRE_Real weight );
HYPRE_Int hypre_PFMGGetJacobiWeight ( void *pfmg_vdata, HYPRE_Real *weight );
HYPRE_Int hypre_PFMGSetRAPType ( void *pfmg_vdata, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGGetRAPType ( void *pfmg_vdata, HYPRE_Int *rap_type );
HYPRE_Int hypre_PFMGSetNumPreRelax ( void *pfmg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_PFMGGetNumPreRelax ( void *pfmg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_PFMGSetNumPostRelax ( void *pfmg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_PFMGGetNumPostRelax ( void *pfmg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_PFMGSetSkipRelax ( void *pfmg_vdata, HYPRE_Int skip_relax );
HYPRE_Int hypre_PFMGGetSkipRelax ( void *pfmg_vdata, HYPRE_Int *skip_relax );
HYPRE_Int hypre_PFMGSetDxyz ( void *pfmg_vdata, HYPRE_Real *dxyz );
HYPRE_Int hypre_PFMGSetLogging ( void *pfmg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_PFMGGetLogging ( void *pfmg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_PFMGSetPrintLevel ( void *pfmg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_PFMGGetPrintLevel ( void *pfmg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_PFMGGetNumIterations ( void *pfmg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PFMGPrintLogging ( void *pfmg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PFMGGetFinalRelativeResidualNorm ( void *pfmg_vdata,
                                                   HYPRE_Real *relative_residual_norm );

/* pfmg_relax.c */
void *hypre_PFMGRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_PFMGRelaxDestroy ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelax ( void *pfmg_relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelaxSetup ( void *pfmg_relax_vdata, hypre_StructMatrix *A,
                                 hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelaxSetType ( void *pfmg_relax_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGRelaxSetJacobiWeight ( void *pfmg_relax_vdata, HYPRE_Real weight );
HYPRE_Int hypre_PFMGRelaxSetPreRelax ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetPostRelax ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetTol ( void *pfmg_relax_vdata, HYPRE_Real tol );
HYPRE_Int hypre_PFMGRelaxSetMaxIter ( void *pfmg_relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGRelaxSetZeroGuess ( void *pfmg_relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGRelaxSetTempVec ( void *pfmg_relax_vdata, hypre_StructVector *t );

/* pfmg_setup.c */
HYPRE_Int hypre_PFMGSetup ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );
HYPRE_Int hypre_PFMGComputeDxyz ( hypre_StructMatrix *A, HYPRE_Real *dxyz, HYPRE_Real *mean,
                                  HYPRE_Real *deviation);
HYPRE_Int hypre_PFMGComputeDxyz_CS  ( HYPRE_Int bi, hypre_StructMatrix *A, HYPRE_Real *cxyz,
                                      HYPRE_Real *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS5 ( HYPRE_Int bi, hypre_StructMatrix *A, HYPRE_Real *cxyz,
                                      HYPRE_Real *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS9 ( HYPRE_Int bi, hypre_StructMatrix *A, HYPRE_Real *cxyz,
                                      HYPRE_Real *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS7 ( HYPRE_Int bi, hypre_StructMatrix *A, HYPRE_Real *cxyz,
                                      HYPRE_Real *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS19( HYPRE_Int bi, hypre_StructMatrix *A, HYPRE_Real *cxyz,
                                      HYPRE_Real *sqcxyz);
HYPRE_Int hypre_PFMGComputeDxyz_SS27( HYPRE_Int bi, hypre_StructMatrix *A, HYPRE_Real *cxyz,
                                      HYPRE_Real *sqcxyz);
HYPRE_Int hypre_ZeroDiagonal ( hypre_StructMatrix *A );

/* pfmg_setup_interp.c */
hypre_StructMatrix *hypre_PFMGCreateInterpOp ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                               HYPRE_Int cdir, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp ( hypre_StructMatrix *A, HYPRE_Int cdir, hypre_Index findex,
                                    hypre_Index stride, hypre_StructMatrix *P, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                        HYPRE_Real *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC1 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                        HYPRE_Real *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC2 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                        HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                        hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                        HYPRE_Real *Pp1, HYPRE_Int rap_type, HYPRE_Int si0, HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS5 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                            HYPRE_Real *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS9 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                            HYPRE_Real *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS7 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                            HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                            hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                            HYPRE_Real *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );

HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS15 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                             HYPRE_Real *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );

HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS19 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                             HYPRE_Real *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );

HYPRE_Int hypre_PFMGSetupInterpOp_CC0_SS27 ( HYPRE_Int i, hypre_StructMatrix *A, hypre_Box *A_dbox,
                                             HYPRE_Int cdir, hypre_Index stride, hypre_Index stridec, hypre_Index start, hypre_IndexRef startc,
                                             hypre_Index loop_size, hypre_Box *P_dbox, HYPRE_Int Pstenc0, HYPRE_Int Pstenc1, HYPRE_Real *Pp0,
                                             HYPRE_Real *Pp1, HYPRE_Int rap_type, hypre_Index *P_stencil_shape );
/* pfmg_setup_rap5.c */
hypre_StructMatrix *hypre_PFMGCreateCoarseOp5 ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_PFMGBuildCoarseOp5 ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_onebox_CC0 ( HYPRE_Int fi, HYPRE_Int ci, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex,
                                                hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_onebox_CC1 ( HYPRE_Int fi, HYPRE_Int ci, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex,
                                                hypre_Index cstride, hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_onebox_CC2 ( HYPRE_Int fi, HYPRE_Int ci, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex,
                                                hypre_Index cstride, hypre_StructMatrix *RAP );

/* pfmg_setup_rap7.c */
hypre_StructMatrix *hypre_PFMGCreateCoarseOp7 ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_PFMGBuildCoarseOp7 ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                     hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                     hypre_StructMatrix *RAP );

/* pfmg_setup_rap.c */
hypre_StructMatrix *hypre_PFMGCreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir, HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                 hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int rap_type,
                                 hypre_StructMatrix *Ac );

/* pfmg_solve.c */
HYPRE_Int hypre_PFMGSolve ( void *pfmg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                            hypre_StructVector *x );

/* point_relax.c */
void *hypre_PointRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_PointRelaxDestroy ( void *relax_vdata );
HYPRE_Int hypre_PointRelaxSetup ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_PointRelax ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
HYPRE_Int hypre_PointRelax_core0 ( void *relax_vdata, hypre_StructMatrix *A,
                                   HYPRE_Int constant_coefficient, hypre_Box *compute_box, HYPRE_Real *bp, HYPRE_Real *xp,
                                   HYPRE_Real *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                   hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
HYPRE_Int hypre_PointRelax_core12 ( void *relax_vdata, hypre_StructMatrix *A,
                                    HYPRE_Int constant_coefficient, hypre_Box *compute_box, HYPRE_Real *bp, HYPRE_Real *xp,
                                    HYPRE_Real *tp, HYPRE_Int boxarray_id, hypre_Box *A_data_box, hypre_Box *b_data_box,
                                    hypre_Box *x_data_box, hypre_Box *t_data_box, hypre_IndexRef stride );
HYPRE_Int hypre_PointRelaxSetTol ( void *relax_vdata, HYPRE_Real tol );
HYPRE_Int hypre_PointRelaxGetTol ( void *relax_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_PointRelaxSetMaxIter ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PointRelaxGetMaxIter ( void *relax_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PointRelaxSetZeroGuess ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_PointRelaxGetZeroGuess ( void *relax_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_PointRelaxGetNumIterations ( void *relax_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PointRelaxSetWeight ( void *relax_vdata, HYPRE_Real weight );
HYPRE_Int hypre_PointRelaxSetNumPointsets ( void *relax_vdata, HYPRE_Int num_pointsets );
HYPRE_Int hypre_PointRelaxSetPointset ( void *relax_vdata, HYPRE_Int pointset,
                                        HYPRE_Int pointset_size, hypre_Index pointset_stride, hypre_Index *pointset_indices );
HYPRE_Int hypre_PointRelaxSetPointsetRank ( void *relax_vdata, HYPRE_Int pointset,
                                            HYPRE_Int pointset_rank );
HYPRE_Int hypre_PointRelaxSetTempVec ( void *relax_vdata, hypre_StructVector *t );
HYPRE_Int hypre_PointRelaxGetFinalRelativeResidualNorm ( void *relax_vdata, HYPRE_Real *norm );
HYPRE_Int hypre_relax_wtx ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                            hypre_StructVector *x );
HYPRE_Int hypre_relax_copy ( void *relax_vdata, HYPRE_Int pointset, hypre_StructVector *t,
                             hypre_StructVector *x );

/* red_black_constantcoef_gs.c */
HYPRE_Int hypre_RedBlackConstantCoefGS ( void *relax_vdata, hypre_StructMatrix *A,
                                         hypre_StructVector *b, hypre_StructVector *x );

/* red_black_gs.c */
void *hypre_RedBlackGSCreate ( MPI_Comm comm );
HYPRE_Int hypre_RedBlackGSDestroy ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetup ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                  hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGS ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                             hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGSSetTol ( void *relax_vdata, HYPRE_Real tol );
HYPRE_Int hypre_RedBlackGSSetMaxIter ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_RedBlackGSSetZeroGuess ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_RedBlackGSSetStartRed ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetStartBlack ( void *relax_vdata );

/* semi.c */
HYPRE_Int hypre_StructInterpAssemble ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                       HYPRE_Int P_stored_as_transpose, HYPRE_Int cdir, hypre_Index index, hypre_Index stride );

/* semi_interp.c */
void *hypre_SemiInterpCreate ( void );
HYPRE_Int hypre_SemiInterpSetup ( void *interp_vdata, hypre_StructMatrix *P,
                                  HYPRE_Int P_stored_as_transpose, hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex,
                                  hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SemiInterp ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                             hypre_StructVector *e );
HYPRE_Int hypre_SemiInterpDestroy ( void *interp_vdata );

/* semi_restrict.c */
void *hypre_SemiRestrictCreate ( void );
HYPRE_Int hypre_SemiRestrictSetup ( void *restrict_vdata, hypre_StructMatrix *R,
                                    HYPRE_Int R_stored_as_transpose, hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex,
                                    hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SemiRestrict ( void *restrict_vdata, hypre_StructMatrix *R, hypre_StructVector *r,
                               hypre_StructVector *rc );
HYPRE_Int hypre_SemiRestrictDestroy ( void *restrict_vdata );

/* semi_setup_rap.c */
hypre_StructMatrix *hypre_SemiCreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir,
                                            HYPRE_Int P_stored_as_transpose );
HYPRE_Int hypre_SemiBuildRAP ( hypre_StructMatrix *A, hypre_StructMatrix *P, hypre_StructMatrix *R,
                               HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride, HYPRE_Int P_stored_as_transpose,
                               hypre_StructMatrix *RAP );

/* smg2_setup_rap.c */
hypre_StructMatrix *hypre_SMG2CreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMG2BuildRAPSym ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG2BuildRAPNoSym ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicSym ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicNoSym ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );

/* smg3_setup_rap.c */
hypre_StructMatrix *hypre_SMG3CreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                            hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMG3BuildRAPSym ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                  hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPNoSym ( hypre_StructMatrix *A, hypre_StructMatrix *PT,
                                    hypre_StructMatrix *R, hypre_StructMatrix *RAP, hypre_Index cindex, hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicSym ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                     hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicNoSym ( hypre_StructMatrix *RAP, hypre_Index cindex,
                                       hypre_Index cstride );

/* smg_axpy.c */
HYPRE_Int hypre_SMGAxpy ( HYPRE_Real alpha, hypre_StructVector *x, hypre_StructVector *y,
                          hypre_Index base_index, hypre_Index base_stride );

/* smg.c */
void *hypre_SMGCreate ( MPI_Comm comm );
HYPRE_Int hypre_SMGDestroy ( void *smg_vdata );
HYPRE_Int hypre_SMGSetMemoryUse ( void *smg_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGGetMemoryUse ( void *smg_vdata, HYPRE_Int *memory_use );
HYPRE_Int hypre_SMGSetTol ( void *smg_vdata, HYPRE_Real tol );
HYPRE_Int hypre_SMGGetTol ( void *smg_vdata, HYPRE_Real *tol );
HYPRE_Int hypre_SMGSetMaxIter ( void *smg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGGetMaxIter ( void *smg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_SMGSetRelChange ( void *smg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SMGGetRelChange ( void *smg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_SMGSetZeroGuess ( void *smg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGGetZeroGuess ( void *smg_vdata, HYPRE_Int *zero_guess );
HYPRE_Int hypre_SMGSetNumPreRelax ( void *smg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGGetNumPreRelax ( void *smg_vdata, HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_SMGSetNumPostRelax ( void *smg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGGetNumPostRelax ( void *smg_vdata, HYPRE_Int *num_post_relax );
HYPRE_Int hypre_SMGSetBase ( void *smg_vdata, hypre_Index base_index, hypre_Index base_stride );
HYPRE_Int hypre_SMGSetLogging ( void *smg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SMGGetLogging ( void *smg_vdata, HYPRE_Int *logging );
HYPRE_Int hypre_SMGSetPrintLevel ( void *smg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SMGGetPrintLevel ( void *smg_vdata, HYPRE_Int *print_level );
HYPRE_Int hypre_SMGGetNumIterations ( void *smg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SMGPrintLogging ( void *smg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SMGGetFinalRelativeResidualNorm ( void *smg_vdata,
                                                  HYPRE_Real *relative_residual_norm );
HYPRE_Int hypre_SMGSetStructVectorConstantValues ( hypre_StructVector *vector, HYPRE_Real values,
                                                   hypre_BoxArray *box_array, hypre_Index stride );
HYPRE_Int hypre_StructSMGSetMaxLevel( void   *smg_vdata, HYPRE_Int   max_level  );

/* smg_relax.c */
void *hypre_SMGRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_SMGRelaxDestroyTempVec ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyARem ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyASol ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroy ( void *relax_vdata );
HYPRE_Int hypre_SMGRelax ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetup ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupTempVec ( void *relax_vdata, hypre_StructMatrix *A,
                                       hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupARem ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupASol ( void *relax_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                    hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetTempVec ( void *relax_vdata, hypre_StructVector *temp_vec );
HYPRE_Int hypre_SMGRelaxSetMemoryUse ( void *relax_vdata, HYPRE_Int memory_use );
HYPRE_Int hypre_SMGRelaxSetTol ( void *relax_vdata, HYPRE_Real tol );
HYPRE_Int hypre_SMGRelaxSetMaxIter ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SMGRelaxSetZeroGuess ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGRelaxSetNumSpaces ( void *relax_vdata, HYPRE_Int num_spaces );
HYPRE_Int hypre_SMGRelaxSetNumPreSpaces ( void *relax_vdata, HYPRE_Int num_pre_spaces );
HYPRE_Int hypre_SMGRelaxSetNumRegSpaces ( void *relax_vdata, HYPRE_Int num_reg_spaces );
HYPRE_Int hypre_SMGRelaxSetSpace ( void *relax_vdata, HYPRE_Int i, HYPRE_Int space_index,
                                   HYPRE_Int space_stride );
HYPRE_Int hypre_SMGRelaxSetRegSpaceRank ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int reg_space_rank );
HYPRE_Int hypre_SMGRelaxSetPreSpaceRank ( void *relax_vdata, HYPRE_Int i,
                                          HYPRE_Int pre_space_rank );
HYPRE_Int hypre_SMGRelaxSetBase ( void *relax_vdata, hypre_Index base_index,
                                  hypre_Index base_stride );
HYPRE_Int hypre_SMGRelaxSetNumPreRelax ( void *relax_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGRelaxSetNumPostRelax ( void *relax_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGRelaxSetNewMatrixStencil ( void *relax_vdata,
                                              hypre_StructStencil *diff_stencil );
HYPRE_Int hypre_SMGRelaxSetupBaseBoxArray ( void *relax_vdata, hypre_StructMatrix *A,
                                            hypre_StructVector *b, hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetMaxLevel( void *relax_vdata, HYPRE_Int   num_max_level );

/* smg_residual.c */
void *hypre_SMGResidualCreate ( void );
HYPRE_Int hypre_SMGResidualSetup ( void *residual_vdata, hypre_StructMatrix *A,
                                   hypre_StructVector *x, hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidual ( void *residual_vdata, hypre_StructMatrix *A, hypre_StructVector *x,
                              hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidualSetBase ( void *residual_vdata, hypre_Index base_index,
                                     hypre_Index base_stride );
HYPRE_Int hypre_SMGResidualDestroy ( void *residual_vdata );

/* smg_residual_unrolled.c */
void *hypre_SMGResidualCreate ( void );
HYPRE_Int hypre_SMGResidualSetup ( void *residual_vdata, hypre_StructMatrix *A,
                                   hypre_StructVector *x, hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidual ( void *residual_vdata, hypre_StructMatrix *A, hypre_StructVector *x,
                              hypre_StructVector *b, hypre_StructVector *r );
HYPRE_Int hypre_SMGResidualSetBase ( void *residual_vdata, hypre_Index base_index,
                                     hypre_Index base_stride );
HYPRE_Int hypre_SMGResidualDestroy ( void *residual_vdata );

/* smg_setup.c */
HYPRE_Int hypre_SMGSetup ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );

/* smg_setup_interp.c */
hypre_StructMatrix *hypre_SMGCreateInterpOp ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                              HYPRE_Int cdir );
HYPRE_Int hypre_SMGSetupInterpOp ( void *relax_data, hypre_StructMatrix *A, hypre_StructVector *b,
                                   hypre_StructVector *x, hypre_StructMatrix *PT, HYPRE_Int cdir, hypre_Index cindex,
                                   hypre_Index findex, hypre_Index stride );

/* smg_setup_rap.c */
hypre_StructMatrix *hypre_SMGCreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                           hypre_StructMatrix *PT, hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMGSetupRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                hypre_StructMatrix *PT, hypre_StructMatrix *Ac, hypre_Index cindex, hypre_Index cstride );

/* smg_setup_restrict.c */
hypre_StructMatrix *hypre_SMGCreateRestrictOp ( hypre_StructMatrix *A, hypre_StructGrid *cgrid,
                                                HYPRE_Int cdir );
HYPRE_Int hypre_SMGSetupRestrictOp ( hypre_StructMatrix *A, hypre_StructMatrix *R,
                                     hypre_StructVector *temp_vec, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride );

/* smg_solve.c */
HYPRE_Int hypre_SMGSolve ( void *smg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                           hypre_StructVector *x );

/* sparse_msg2_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSG2CreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSG2BuildRAPSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG2BuildRAPNoSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );

/* sparse_msg3_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSG3CreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                  hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSG3BuildRAPSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                        hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                        hypre_Index stridePR, hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG3BuildRAPNoSym ( hypre_StructMatrix *A, hypre_StructMatrix *P,
                                          hypre_StructMatrix *R, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                          hypre_Index stridePR, hypre_StructMatrix *RAP );

/* sparse_msg.c */
void *hypre_SparseMSGCreate ( MPI_Comm comm );
HYPRE_Int hypre_SparseMSGDestroy ( void *smsg_vdata );
HYPRE_Int hypre_SparseMSGSetTol ( void *smsg_vdata, HYPRE_Real tol );
HYPRE_Int hypre_SparseMSGSetMaxIter ( void *smsg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SparseMSGSetJump ( void *smsg_vdata, HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGSetRelChange ( void *smsg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SparseMSGSetZeroGuess ( void *smsg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SparseMSGSetRelaxType ( void *smsg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SparseMSGSetJacobiWeight ( void *smsg_vdata, HYPRE_Real weight );
HYPRE_Int hypre_SparseMSGSetNumPreRelax ( void *smsg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SparseMSGSetNumPostRelax ( void *smsg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SparseMSGSetNumFineRelax ( void *smsg_vdata, HYPRE_Int num_fine_relax );
HYPRE_Int hypre_SparseMSGSetLogging ( void *smsg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SparseMSGSetPrintLevel ( void *smsg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SparseMSGGetNumIterations ( void *smsg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SparseMSGPrintLogging ( void *smsg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_SparseMSGGetFinalRelativeResidualNorm ( void *smsg_vdata,
                                                        HYPRE_Real *relative_residual_norm );

/* sparse_msg_filter.c */
HYPRE_Int hypre_SparseMSGFilterSetup ( hypre_StructMatrix *A, HYPRE_Int *num_grids, HYPRE_Int lx,
                                       HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump, hypre_StructVector *visitx, hypre_StructVector *visity,
                                       hypre_StructVector *visitz );
HYPRE_Int hypre_SparseMSGFilter ( hypre_StructVector *visit, hypre_StructVector *e, HYPRE_Int lx,
                                  HYPRE_Int ly, HYPRE_Int lz, HYPRE_Int jump );

/* sparse_msg_interp.c */
void *hypre_SparseMSGInterpCreate ( void );
HYPRE_Int hypre_SparseMSGInterpSetup ( void *interp_vdata, hypre_StructMatrix *P,
                                       hypre_StructVector *xc, hypre_StructVector *e, hypre_Index cindex, hypre_Index findex,
                                       hypre_Index stride, hypre_Index strideP );
HYPRE_Int hypre_SparseMSGInterp ( void *interp_vdata, hypre_StructMatrix *P, hypre_StructVector *xc,
                                  hypre_StructVector *e );
HYPRE_Int hypre_SparseMSGInterpDestroy ( void *interp_vdata );

/* sparse_msg_restrict.c */
void *hypre_SparseMSGRestrictCreate ( void );
HYPRE_Int hypre_SparseMSGRestrictSetup ( void *restrict_vdata, hypre_StructMatrix *R,
                                         hypre_StructVector *r, hypre_StructVector *rc, hypre_Index cindex, hypre_Index findex,
                                         hypre_Index stride, hypre_Index strideR );
HYPRE_Int hypre_SparseMSGRestrict ( void *restrict_vdata, hypre_StructMatrix *R,
                                    hypre_StructVector *r, hypre_StructVector *rc );
HYPRE_Int hypre_SparseMSGRestrictDestroy ( void *restrict_vdata );

/* sparse_msg_setup.c */
HYPRE_Int hypre_SparseMSGSetup ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );

/* sparse_msg_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSGCreateRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                                 hypre_StructMatrix *P, hypre_StructGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSGSetupRAPOp ( hypre_StructMatrix *R, hypre_StructMatrix *A,
                                      hypre_StructMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                      hypre_Index stridePR, hypre_StructMatrix *Ac );

/* sparse_msg_solve.c */
HYPRE_Int hypre_SparseMSGSolve ( void *smsg_vdata, hypre_StructMatrix *A, hypre_StructVector *b,
                                 hypre_StructVector *x );
