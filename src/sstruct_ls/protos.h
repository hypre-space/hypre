/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* HYPRE_sstruct_int.c */
HYPRE_Int hypre_SStructSetRandomValues ( void *v, HYPRE_Int seed );

/* krylov.c */
HYPRE_Int hypre_SStructKrylovIdentitySetup ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_SStructKrylovIdentity ( void *vdata, void *A, void *b, void *x );

/* krylov_sstruct.c */
void *hypre_SStructKrylovCAlloc ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
HYPRE_Int hypre_SStructKrylovFree ( void *ptr );
void *hypre_SStructKrylovCreateVector ( void *vvector );
void *hypre_SStructKrylovCreateVectorArray ( HYPRE_Int n, void *vvector );
HYPRE_Int hypre_SStructKrylovDestroyVector ( void *vvector );
void *hypre_SStructKrylovMatvecCreate ( void *A, void *x );
HYPRE_Int hypre_SStructKrylovMatvec ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x,
                                      HYPRE_Complex beta, void *y );
HYPRE_Int hypre_SStructKrylovMatvecDestroy ( void *matvec_data );
HYPRE_Real hypre_SStructKrylovInnerProd ( void *x, void *y );
HYPRE_Int hypre_SStructKrylovInnerProdTagged ( void *x, void *y, HYPRE_Int *num_tags_ptr,
                                               HYPRE_Complex **iprod_ptr );
HYPRE_Int hypre_SStructKrylovCopyVector ( void *x, void *y );
HYPRE_Int hypre_SStructKrylovClearVector ( void *x );
HYPRE_Int hypre_SStructKrylovScaleVector ( HYPRE_Complex alpha, void *x );
HYPRE_Int hypre_SStructKrylovAxpy ( HYPRE_Complex alpha, void *x, void *y );
HYPRE_Int hypre_SStructKrylovCommInfo ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );

/* node_relax.c */
void *hypre_NodeRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_NodeRelaxDestroy ( void *relax_vdata );
HYPRE_Int hypre_NodeRelaxSetup ( void *relax_vdata, hypre_SStructPMatrix *A,
                                 hypre_SStructPVector *b, hypre_SStructPVector *x );
HYPRE_Int hypre_NodeRelax ( void *relax_vdata, hypre_SStructPMatrix *A, hypre_SStructPVector *b,
                            hypre_SStructPVector *x );
HYPRE_Int hypre_NodeRelaxSetTol ( void *relax_vdata, HYPRE_Real tol );
HYPRE_Int hypre_NodeRelaxSetMaxIter ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_NodeRelaxSetZeroGuess ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_NodeRelaxSetWeight ( void *relax_vdata, HYPRE_Real weight );
HYPRE_Int hypre_NodeRelaxSetNumNodesets ( void *relax_vdata, HYPRE_Int num_nodesets );
HYPRE_Int hypre_NodeRelaxSetNodeset ( void *relax_vdata, HYPRE_Int nodeset, HYPRE_Int nodeset_size,
                                      hypre_Index nodeset_stride, hypre_Index *nodeset_indices );
HYPRE_Int hypre_NodeRelaxSetNodesetRank ( void *relax_vdata, HYPRE_Int nodeset,
                                          HYPRE_Int nodeset_rank );
HYPRE_Int hypre_NodeRelaxSetTempVec ( void *relax_vdata, hypre_SStructPVector *t );

/* ssamg.c */
void * hypre_SSAMGCreate ( hypre_MPI_Comm comm );
HYPRE_Int hypre_SSAMGDestroy ( void *ssamg_vdata );
HYPRE_Int hypre_SSAMGSetTol ( void *ssamg_vdata, HYPRE_Real tol );
HYPRE_Int hypre_SSAMGSetMaxIter ( void *ssamg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SSAMGSetMaxLevels ( void *ssamg_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_SSAMGSetRelChange ( void *ssamg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SSAMGSetZeroGuess ( void *ssamg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SSAMGSetNonGalerkinRAP ( void *ssamg_vdata, HYPRE_Int non_galerkin );
HYPRE_Int hypre_SSAMGSetDxyz ( void *ssamg_vdata, HYPRE_Int nparts, HYPRE_Real **dxyz );
HYPRE_Int hypre_SSAMGSetRelaxType ( void *ssamg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SSAMGSetInterpType ( void *ssamg_vdata, HYPRE_Int interp_type );
HYPRE_Int hypre_SSAMGSetRelaxWeight ( void *ssamg_vdata, HYPRE_Real relax_weight );
HYPRE_Int hypre_SSAMGSetSkipRelax ( void *ssamg_vdata, HYPRE_Int skip_relax );
HYPRE_Int hypre_SSAMGSetNumPreRelax ( void *ssamg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SSAMGSetNumPosRelax ( void *ssamg_vdata, HYPRE_Int num_pos_relax );
HYPRE_Int hypre_SSAMGSetNumCoarseRelax ( void *ssamg_vdata, HYPRE_Int num_coarse_relax );
HYPRE_Int hypre_SSAMGSetMaxCoarseSize ( void *ssamg_vdata, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_SSAMGSetCoarseSolverType ( void *ssamg_vdata, HYPRE_Int csolver_type );
HYPRE_Int hypre_SSAMGSetPrintLevel ( void *ssamg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SSAMGSetPrintFreq ( void *ssamg_vdata, HYPRE_Int print_freq );
HYPRE_Int hypre_SSAMGSetLogging ( void *ssamg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SSAMGPrintLogging ( void *ssamg_vdata );
HYPRE_Int hypre_SSAMGPrintStats ( void *ssamg_vdata );
HYPRE_Int hypre_SSAMGGetNumIterations ( void *ssamg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SSAMGGetFinalRelativeResidualNorm ( void *ssamg_vdata,
                                                    HYPRE_Real *relative_residual_norm );

/* ssamg_csolver.c */
HYPRE_Int hypre_SSAMGCoarseSolverSetup( void *ssamg_vdata );
HYPRE_Int hypre_SSAMGCoarseSolve( void *ssamg_vdata );
HYPRE_Int hypre_SSAMGCoarseSolverDestroy( void *ssamg_vdata );

/* ssamg_interp.c */
hypre_SStructMatrix* hypre_SSAMGCreateInterpOp ( hypre_SStructMatrix *A, hypre_SStructGrid *cgrid,
                                                 HYPRE_Int *cdir_p );
HYPRE_Int hypre_SSAMGSetupInterpOp ( hypre_SStructMatrix  *A, HYPRE_Int *cdir_p,
                                     hypre_SStructMatrix *P, HYPRE_Int interp_type );

/* ssamg_uinterp.c */
HYPRE_Int hypre_SSAMGSetupUInterpOp ( hypre_SStructMatrix  *A, HYPRE_Int *cdir_p,
                                      hypre_SStructMatrix *P, HYPRE_Int interp_type );

/* ssamg_setup_rap.c */
HYPRE_Int hypre_SSAMGComputeRAP ( hypre_SStructMatrix *A, hypre_SStructMatrix *P,
                                  hypre_SStructGrid **cgrid, HYPRE_Int *cdir_p, HYPRE_Int non_galerkin,
                                  hypre_SStructMatrix **Ac_ptr );
HYPRE_Int hypre_SSAMGComputeRAPNonGlk ( hypre_SStructMatrix *A, hypre_SStructMatrix *P,
                                        HYPRE_Int *cdir_p, hypre_SStructMatrix **Ac_ptr );

/* ssamg_relax.c */
HYPRE_Int hypre_SSAMGRelaxCreate ( MPI_Comm comm, HYPRE_Int nparts, void **relax_vdata_ptr );
HYPRE_Int hypre_SSAMGRelaxDestroy ( void *relax_vdata );
HYPRE_Int hypre_SSAMGRelaxSetup ( void *relax_vdata, hypre_SStructMatrix *A, hypre_SStructVector *b,
                                  hypre_SStructVector *x );
HYPRE_Int hypre_SSAMGRelax ( void *relax_vdata, hypre_SStructMatrix *A, hypre_SStructVector *b,
                             hypre_SStructVector *x );
HYPRE_Int hypre_SSAMGRelaxGeneric ( void *relax_vdata, hypre_SStructMatrix *A,
                                    hypre_SStructVector *b, hypre_SStructVector *x );
HYPRE_Int hypre_SSAMGRelaxJacobi ( void *relax_vdata, hypre_SStructMatrix *A,
                                   hypre_SStructVector *b, hypre_SStructVector *x );
HYPRE_Int hypre_SSAMGRelaxSetPreRelax ( void  *relax_vdata );
HYPRE_Int hypre_SSAMGRelaxSetPostRelax ( void  *relax_vdata );
HYPRE_Int hypre_SSAMGRelaxSetTol ( void *relax_vdata, HYPRE_Real tol );
HYPRE_Int hypre_SSAMGRelaxSetWeights ( void *relax_vdata, HYPRE_Real *weights );
HYPRE_Int hypre_SSAMGRelaxSetActiveParts ( void *relax_vdata, HYPRE_Int *active_p );
HYPRE_Int hypre_SSAMGRelaxSetMatvecData ( void  *relax_vdata, void  *matvec_vdata );
HYPRE_Int hypre_SSAMGRelaxSetNumNodesets ( void *relax_vdata, HYPRE_Int num_nodesets );
HYPRE_Int hypre_SSAMGRelaxSetNodeset ( void *relax_vdata, HYPRE_Int nodeset, HYPRE_Int nodeset_size,
                                       hypre_Index nodeset_stride, hypre_Index *nodeset_indices );
HYPRE_Int hypre_SSAMGRelaxSetNodesetRank ( void *relax_vdata, HYPRE_Int nodeset,
                                           HYPRE_Int nodeset_rank );
HYPRE_Int hypre_SSAMGRelaxSetMaxIter ( void *relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SSAMGRelaxSetZeroGuess ( void *relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SSAMGRelaxSetType ( void *relax_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SSAMGRelaxSetTempVec ( void *relax_vdata, hypre_SStructVector *t );
HYPRE_Int hypre_SSAMGRelaxGetRelaxWeight ( void *relax_vdata, HYPRE_Int part, HYPRE_Real *weight );

/* ssamg_setup.c */
HYPRE_Int hypre_SSAMGSetup ( void *ssamg_vdata, hypre_SStructMatrix *A, hypre_SStructVector *b,
                             hypre_SStructVector *x );
HYPRE_Int hypre_SSAMGComputeNumCoarseRelax ( void *ssamg_vdata );
HYPRE_Int hypre_SSAMGComputeMaxLevels ( hypre_SStructGrid *grid, HYPRE_Int *max_levels );
HYPRE_Int hypre_SSAMGComputeDxyz ( hypre_SStructMatrix *A, HYPRE_Real **dxyz,
                                   HYPRE_Int *dxyz_flag );
HYPRE_Int hypre_SSAMGCoarsen ( void *ssamg_vdata, hypre_SStructGrid *grid, HYPRE_Int *dxyz_flag,
                               HYPRE_Real **dxyz );

/* ssamg_solve.c */
HYPRE_Int hypre_SSAMGSolve ( void *ssamg_vdata, hypre_SStructMatrix *A, hypre_SStructVector *b,
                             hypre_SStructVector *x );

/* sys_pfmg.c */
void *hypre_SysPFMGCreate ( MPI_Comm comm );
HYPRE_Int hypre_SysPFMGDestroy ( void *sys_pfmg_vdata );
HYPRE_Int hypre_SysPFMGSetTol ( void *sys_pfmg_vdata, HYPRE_Real tol );
HYPRE_Int hypre_SysPFMGSetMaxIter ( void *sys_pfmg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SysPFMGSetRelChange ( void *sys_pfmg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_SysPFMGSetZeroGuess ( void *sys_pfmg_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SysPFMGSetRelaxType ( void *sys_pfmg_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SysPFMGSetJacobiWeight ( void *sys_pfmg_vdata, HYPRE_Real weight );
HYPRE_Int hypre_SysPFMGSetNumPreRelax ( void *sys_pfmg_vdata, HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SysPFMGSetNumPostRelax ( void *sys_pfmg_vdata, HYPRE_Int num_post_relax );
HYPRE_Int hypre_SysPFMGSetSkipRelax ( void *sys_pfmg_vdata, HYPRE_Int skip_relax );
HYPRE_Int hypre_SysPFMGSetDxyz ( void *sys_pfmg_vdata, HYPRE_Real *dxyz );
HYPRE_Int hypre_SysPFMGSetLogging ( void *sys_pfmg_vdata, HYPRE_Int logging );
HYPRE_Int hypre_SysPFMGSetPrintLevel ( void *sys_pfmg_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_SysPFMGGetNumIterations ( void *sys_pfmg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_SysPFMGPrintLogging ( void *sys_pfmg_vdata );
HYPRE_Int hypre_SysPFMGGetFinalRelativeResidualNorm ( void *sys_pfmg_vdata,
                                                      HYPRE_Real *relative_residual_norm );

/* sys_pfmg_relax.c */
void *hypre_SysPFMGRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_SysPFMGRelaxDestroy ( void *sys_pfmg_relax_vdata );
HYPRE_Int hypre_SysPFMGRelax ( void *sys_pfmg_relax_vdata, hypre_SStructPMatrix *A,
                               hypre_SStructPVector *b, hypre_SStructPVector *x );
HYPRE_Int hypre_SysPFMGRelaxSetup ( void *sys_pfmg_relax_vdata, hypre_SStructPMatrix *A,
                                    hypre_SStructPVector *b, hypre_SStructPVector *x );
HYPRE_Int hypre_SysPFMGRelaxSetType ( void *sys_pfmg_relax_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_SysPFMGRelaxSetJacobiWeight ( void *sys_pfmg_relax_vdata, HYPRE_Real weight );
HYPRE_Int hypre_SysPFMGRelaxSetPreRelax ( void *sys_pfmg_relax_vdata );
HYPRE_Int hypre_SysPFMGRelaxSetPostRelax ( void *sys_pfmg_relax_vdata );
HYPRE_Int hypre_SysPFMGRelaxSetTol ( void *sys_pfmg_relax_vdata, HYPRE_Real tol );
HYPRE_Int hypre_SysPFMGRelaxSetMaxIter ( void *sys_pfmg_relax_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_SysPFMGRelaxSetZeroGuess ( void *sys_pfmg_relax_vdata, HYPRE_Int zero_guess );
HYPRE_Int hypre_SysPFMGRelaxSetTempVec ( void *sys_pfmg_relax_vdata, hypre_SStructPVector *t );

/* sys_pfmg_setup.c */
HYPRE_Int hypre_SysPFMGSetup ( void *sys_pfmg_vdata, hypre_SStructMatrix *A_in,
                               hypre_SStructVector *b_in, hypre_SStructVector *x_in );
HYPRE_Int hypre_SysPFMGZeroDiagonal( hypre_SStructPMatrix *A );
#if 0
HYPRE_Int hypre_SysStructCoarsen ( hypre_SStructPGrid *fgrid, hypre_Index index, hypre_Index stride,
                                   HYPRE_Int prune, hypre_SStructPGrid **cgrid_ptr );
#endif

/* sys_pfmg_setup_interp.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateInterpOp( hypre_SStructPMatrix *A, HYPRE_Int cdir,
                                                   hypre_Index stride );
HYPRE_Int hypre_SysPFMGSetupInterpOp( hypre_SStructPMatrix *P, hypre_SStructPMatrix *A,
                                      HYPRE_Int cdir );

/* sys_pfmg_setup_rap.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateRAPOp ( hypre_SStructPMatrix *R, hypre_SStructPMatrix *A,
                                                 hypre_SStructPMatrix *P, hypre_SStructPGrid *coarse_grid, HYPRE_Int cdir );
HYPRE_Int hypre_SysPFMGSetupRAPOp ( hypre_SStructPMatrix *R, hypre_SStructPMatrix *A,
                                    hypre_SStructPMatrix *P, HYPRE_Int cdir, hypre_Index cindex, hypre_Index cstride,
                                    hypre_SStructPMatrix *Ac );

/* sys_pfmg_solve.c */
HYPRE_Int hypre_SysPFMGSolve ( void *sys_pfmg_vdata, hypre_SStructMatrix *A_in,
                               hypre_SStructVector *b_in, hypre_SStructVector *x_in );

/* sys_semi_interp.c */
HYPRE_Int hypre_SysSemiInterpCreate ( void **sys_interp_vdata_ptr );
HYPRE_Int hypre_SysSemiInterpSetup ( void *sys_interp_vdata, hypre_SStructPMatrix *P,
                                     HYPRE_Int P_stored_as_transpose, hypre_SStructPVector *xc, hypre_SStructPVector *e,
                                     hypre_Index cindex, hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SysSemiInterp ( void *sys_interp_vdata, hypre_SStructPMatrix *P,
                                hypre_SStructPVector *xc, hypre_SStructPVector *e );
HYPRE_Int hypre_SysSemiInterpDestroy ( void *sys_interp_vdata );

/* sys_semi_restrict.c */
HYPRE_Int hypre_SysSemiRestrictCreate ( void **sys_restrict_vdata_ptr );
HYPRE_Int hypre_SysSemiRestrictSetup ( void *sys_restrict_vdata, hypre_SStructPMatrix *R,
                                       HYPRE_Int R_stored_as_transpose, hypre_SStructPVector *r, hypre_SStructPVector *rc,
                                       hypre_Index cindex, hypre_Index findex, hypre_Index stride );
HYPRE_Int hypre_SysSemiRestrict ( void *sys_restrict_vdata, hypre_SStructPMatrix *R,
                                  hypre_SStructPVector *r, hypre_SStructPVector *rc );
HYPRE_Int hypre_SysSemiRestrictDestroy ( void *sys_restrict_vdata );
