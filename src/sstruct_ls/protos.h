/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* HYPRE_sstruct_int.c */
HYPRE_Int hypre_SStructPVectorSetRandomValues ( hypre_SStructPVector *pvector, HYPRE_Int seed );
HYPRE_Int hypre_SStructVectorSetRandomValues ( hypre_SStructVector *vector, HYPRE_Int seed );
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
HYPRE_Int hypre_SysPFMGPrintLogging ( void *sys_pfmg_vdata, HYPRE_Int myid );
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
HYPRE_Int hypre_SysStructCoarsen ( hypre_SStructPGrid *fgrid, hypre_Index index, hypre_Index stride,
                                   HYPRE_Int prune, hypre_SStructPGrid **cgrid_ptr );

/* sys_pfmg_setup_interp.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateInterpOp ( hypre_SStructPMatrix *A,
                                                    hypre_SStructPGrid *cgrid, HYPRE_Int cdir );
HYPRE_Int hypre_SysPFMGSetupInterpOp ( hypre_SStructPMatrix *A, HYPRE_Int cdir, hypre_Index findex,
                                       hypre_Index stride, hypre_SStructPMatrix *P );

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
