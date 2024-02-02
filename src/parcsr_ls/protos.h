/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* ads.c */
void *hypre_ADSCreate ( void );
HYPRE_Int hypre_ADSDestroy ( void *solver );
HYPRE_Int hypre_ADSSetDiscreteCurl ( void *solver, hypre_ParCSRMatrix *C );
HYPRE_Int hypre_ADSSetDiscreteGradient ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_ADSSetCoordinateVectors ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_ADSSetInterpolations ( void *solver, hypre_ParCSRMatrix *RT_Pi,
                                       hypre_ParCSRMatrix *RT_Pix, hypre_ParCSRMatrix *RT_Piy,
                                       hypre_ParCSRMatrix *RT_Piz, hypre_ParCSRMatrix *ND_Pi,
                                       hypre_ParCSRMatrix *ND_Pix, hypre_ParCSRMatrix *ND_Piy,
                                       hypre_ParCSRMatrix *ND_Piz );
HYPRE_Int hypre_ADSSetMaxIter ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_ADSSetTol ( void *solver, HYPRE_Real tol );
HYPRE_Int hypre_ADSSetCycleType ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_ADSSetPrintLevel ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_ADSSetSmoothingOptions ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, HYPRE_Real A_relax_weight, HYPRE_Real A_omega );
HYPRE_Int hypre_ADSSetChebySmoothingOptions ( void *solver, HYPRE_Int A_cheby_order,
                                              HYPRE_Real A_cheby_fraction );
HYPRE_Int hypre_ADSSetAMSOptions ( void *solver, HYPRE_Int B_C_cycle_type,
                                   HYPRE_Int B_C_coarsen_type, HYPRE_Int B_C_agg_levels, HYPRE_Int B_C_relax_type,
                                   HYPRE_Real B_C_theta, HYPRE_Int B_C_interp_type, HYPRE_Int B_C_Pmax );
HYPRE_Int hypre_ADSSetAMGOptions ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                   HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, HYPRE_Real B_Pi_theta,
                                   HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_ADSComputePi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *G,
                               hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z, hypre_ParCSRMatrix *PiNDx,
                               hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_ADSComputePixyz ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C,
                                  hypre_ParCSRMatrix *G, hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z,
                                  hypre_ParCSRMatrix *PiNDx, hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz,
                                  hypre_ParCSRMatrix **Pix_ptr, hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_ADSSetup ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ADSSolve ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ADSGetNumIterations ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ADSGetFinalRelativeResidualNorm ( void *solver, HYPRE_Real *rel_resid_norm );

/* ame.c */
void *hypre_AMECreate ( void );
HYPRE_Int hypre_AMEDestroy ( void *esolver );
HYPRE_Int hypre_AMESetAMSSolver ( void *esolver, void *ams_solver );
HYPRE_Int hypre_AMESetMassMatrix ( void *esolver, hypre_ParCSRMatrix *M );
HYPRE_Int hypre_AMESetBlockSize ( void *esolver, HYPRE_Int block_size );
HYPRE_Int hypre_AMESetMaxIter ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetMaxPCGIter ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetTol ( void *esolver, HYPRE_Real tol );
HYPRE_Int hypre_AMESetRTol ( void *esolver, HYPRE_Real tol );
HYPRE_Int hypre_AMESetPrintLevel ( void *esolver, HYPRE_Int print_level );
HYPRE_Int hypre_AMESetup ( void *esolver );
HYPRE_Int hypre_AMEDiscrDivFreeComponent ( void *esolver, hypre_ParVector *b );
void hypre_AMEOperatorA ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorA ( void *data, void *x, void *y );
void hypre_AMEOperatorM ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorM ( void *data, void *x, void *y );
void hypre_AMEOperatorB ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorB ( void *data, void *x, void *y );
HYPRE_Int hypre_AMESolve ( void *esolver );
HYPRE_Int hypre_AMEGetEigenvectors ( void *esolver, HYPRE_ParVector **eigenvectors_ptr );
HYPRE_Int hypre_AMEGetEigenvalues ( void *esolver, HYPRE_Real **eigenvalues_ptr );

/* amg_hybrid.c */
void *hypre_AMGHybridCreate ( void );
HYPRE_Int hypre_AMGHybridDestroy ( void *AMGhybrid_vdata );
HYPRE_Int hypre_AMGHybridSetTol ( void *AMGhybrid_vdata, HYPRE_Real tol );
HYPRE_Int hypre_AMGHybridSetAbsoluteTol ( void *AMGhybrid_vdata, HYPRE_Real a_tol );
HYPRE_Int hypre_AMGHybridSetConvergenceTol ( void *AMGhybrid_vdata, HYPRE_Real cf_tol );
HYPRE_Int hypre_AMGHybridSetNonGalerkinTol ( void *AMGhybrid_vdata, HYPRE_Int nongalerk_num_tol,
                                             HYPRE_Real *nongalerkin_tol );
HYPRE_Int hypre_AMGHybridSetDSCGMaxIter ( void *AMGhybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_AMGHybridSetPCGMaxIter ( void *AMGhybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_AMGHybridSetSetupType ( void *AMGhybrid_vdata, HYPRE_Int setup_type );
HYPRE_Int hypre_AMGHybridSetSolverType ( void *AMGhybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_AMGHybridSetRecomputeResidual ( void *AMGhybrid_vdata,
                                                HYPRE_Int recompute_residual );
HYPRE_Int hypre_AMGHybridGetRecomputeResidual ( void *AMGhybrid_vdata,
                                                HYPRE_Int *recompute_residual );
HYPRE_Int hypre_AMGHybridSetRecomputeResidualP ( void *AMGhybrid_vdata,
                                                 HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_AMGHybridGetRecomputeResidualP ( void *AMGhybrid_vdata,
                                                 HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_AMGHybridSetKDim ( void *AMGhybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_AMGHybridSetStopCrit ( void *AMGhybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_AMGHybridSetTwoNorm ( void *AMGhybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_AMGHybridSetRelChange ( void *AMGhybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_AMGHybridSetPrecond ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                       void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_AMGHybridSetLogging ( void *AMGhybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_AMGHybridSetPrintLevel ( void *AMGhybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_AMGHybridSetStrongThreshold ( void *AMGhybrid_vdata, HYPRE_Real strong_threshold );
HYPRE_Int hypre_AMGHybridSetMaxRowSum ( void *AMGhybrid_vdata, HYPRE_Real max_row_sum );
HYPRE_Int hypre_AMGHybridSetTruncFactor ( void *AMGhybrid_vdata, HYPRE_Real trunc_factor );
HYPRE_Int hypre_AMGHybridSetPMaxElmts ( void *AMGhybrid_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_AMGHybridSetMaxLevels ( void *AMGhybrid_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_AMGHybridSetMeasureType ( void *AMGhybrid_vdata, HYPRE_Int measure_type );
HYPRE_Int hypre_AMGHybridSetCoarsenType ( void *AMGhybrid_vdata, HYPRE_Int coarsen_type );
HYPRE_Int hypre_AMGHybridSetInterpType ( void *AMGhybrid_vdata, HYPRE_Int interp_type );
HYPRE_Int hypre_AMGHybridSetCycleType ( void *AMGhybrid_vdata, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMGHybridSetNumSweeps ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps );
HYPRE_Int hypre_AMGHybridSetCycleNumSweeps ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetRelaxType ( void *AMGhybrid_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_AMGHybridSetKeepTranspose ( void *AMGhybrid_vdata, HYPRE_Int keepT );
HYPRE_Int hypre_AMGHybridSetSplittingStrategy( void *AMGhybrid_vdata,
                                               HYPRE_Int splitting_strategy );
HYPRE_Int hypre_AMGHybridSetCycleRelaxType ( void *AMGhybrid_vdata, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetRelaxOrder ( void *AMGhybrid_vdata, HYPRE_Int relax_order );
HYPRE_Int hypre_AMGHybridSetMaxCoarseSize ( void *AMGhybrid_vdata, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_AMGHybridSetMinCoarseSize ( void *AMGhybrid_vdata, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_AMGHybridSetSeqThreshold ( void *AMGhybrid_vdata, HYPRE_Int seq_threshold );
HYPRE_Int hypre_AMGHybridSetNumGridSweeps ( void *AMGhybrid_vdata, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_AMGHybridSetGridRelaxType ( void *AMGhybrid_vdata, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_AMGHybridSetGridRelaxPoints ( void *AMGhybrid_vdata,
                                              HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_AMGHybridSetRelaxWeight ( void *AMGhybrid_vdata, HYPRE_Real *relax_weight );
HYPRE_Int hypre_AMGHybridSetOmega ( void *AMGhybrid_vdata, HYPRE_Real *omega );
HYPRE_Int hypre_AMGHybridSetRelaxWt ( void *AMGhybrid_vdata, HYPRE_Real relax_wt );
HYPRE_Int hypre_AMGHybridSetLevelRelaxWt ( void *AMGhybrid_vdata, HYPRE_Real relax_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetOuterWt ( void *AMGhybrid_vdata, HYPRE_Real outer_wt );
HYPRE_Int hypre_AMGHybridSetLevelOuterWt ( void *AMGhybrid_vdata, HYPRE_Real outer_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetNumPaths ( void *AMGhybrid_vdata, HYPRE_Int num_paths );
HYPRE_Int hypre_AMGHybridSetDofFunc ( void *AMGhybrid_vdata, HYPRE_Int *dof_func );
HYPRE_Int hypre_AMGHybridSetAggNumLevels ( void *AMGhybrid_vdata, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_AMGHybridSetAggInterpType ( void *AMGhybrid_vdata, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_AMGHybridSetNumFunctions ( void *AMGhybrid_vdata, HYPRE_Int num_functions );
HYPRE_Int hypre_AMGHybridSetNodal ( void *AMGhybrid_vdata, HYPRE_Int nodal );
HYPRE_Int hypre_AMGHybridGetSetupSolveTime( void *AMGhybrid_vdata, HYPRE_Real *time );
HYPRE_Int hypre_AMGHybridGetNumIterations ( void *AMGhybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_AMGHybridGetDSCGNumIterations ( void *AMGhybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_AMGHybridGetPCGNumIterations ( void *AMGhybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_AMGHybridGetFinalRelativeResidualNorm ( void *AMGhybrid_vdata,
                                                        HYPRE_Real *final_rel_res_norm );
HYPRE_Int hypre_AMGHybridSetup ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
HYPRE_Int hypre_AMGHybridSolve ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );

/* ams.c */
HYPRE_Int hypre_ParCSRRelax ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int relax_type,
                              HYPRE_Int relax_times, HYPRE_Real *l1_norms, HYPRE_Real relax_weight, HYPRE_Real omega,
                              HYPRE_Real max_eig_est, HYPRE_Real min_eig_est, HYPRE_Int cheby_order, HYPRE_Real cheby_fraction,
                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *z );
hypre_ParVector *hypre_ParVectorInRangeOf ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInDomainOf ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParVectorBlockSplit ( hypre_ParVector *x, hypre_ParVector *x_ [3 ], HYPRE_Int dim );
HYPRE_Int hypre_ParVectorBlockGather ( hypre_ParVector *x, hypre_ParVector *x_ [3 ],
                                       HYPRE_Int dim );
HYPRE_Int hypre_BoomerAMGBlockSolve ( void *B, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                      hypre_ParVector *x );
HYPRE_Int hypre_ParCSRMatrixFixZeroRows ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRComputeL1Norms ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                       HYPRE_Int *cf_marker, HYPRE_Real **l1_norm_ptr );
HYPRE_Int hypre_ParCSRMatrixSetDiagRows ( hypre_ParCSRMatrix *A, HYPRE_Real d );
void *hypre_AMSCreate ( void );
HYPRE_Int hypre_AMSDestroy ( void *solver );
HYPRE_Int hypre_AMSSetDimension ( void *solver, HYPRE_Int dim );
HYPRE_Int hypre_AMSSetDiscreteGradient ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_AMSSetCoordinateVectors ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_AMSSetEdgeConstantVectors ( void *solver, hypre_ParVector *Gx, hypre_ParVector *Gy,
                                            hypre_ParVector *Gz );
HYPRE_Int hypre_AMSSetInterpolations ( void *solver, hypre_ParCSRMatrix *Pi,
                                       hypre_ParCSRMatrix *Pix, hypre_ParCSRMatrix *Piy, hypre_ParCSRMatrix *Piz );
HYPRE_Int hypre_AMSSetAlphaPoissonMatrix ( void *solver, hypre_ParCSRMatrix *A_Pi );
HYPRE_Int hypre_AMSSetBetaPoissonMatrix ( void *solver, hypre_ParCSRMatrix *A_G );
HYPRE_Int hypre_AMSSetInteriorNodes ( void *solver, hypre_ParVector *interior_nodes );
HYPRE_Int hypre_AMSSetProjectionFrequency ( void *solver, HYPRE_Int projection_frequency );
HYPRE_Int hypre_AMSSetMaxIter ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_AMSSetTol ( void *solver, HYPRE_Real tol );
HYPRE_Int hypre_AMSSetCycleType ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMSSetPrintLevel ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_AMSSetSmoothingOptions ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, HYPRE_Real A_relax_weight, HYPRE_Real A_omega );
HYPRE_Int hypre_AMSSetChebySmoothingOptions ( void *solver, HYPRE_Int A_cheby_order,
                                              HYPRE_Real A_cheby_fraction );
HYPRE_Int hypre_AMSSetAlphaAMGOptions ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                        HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, HYPRE_Real B_Pi_theta,
                                        HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_AMSSetAlphaAMGCoarseRelaxType ( void *solver, HYPRE_Int B_Pi_coarse_relax_type );
HYPRE_Int hypre_AMSSetBetaAMGOptions ( void *solver, HYPRE_Int B_G_coarsen_type,
                                       HYPRE_Int B_G_agg_levels, HYPRE_Int B_G_relax_type, HYPRE_Real B_G_theta, HYPRE_Int B_G_interp_type,
                                       HYPRE_Int B_G_Pmax );
HYPRE_Int hypre_AMSSetBetaAMGCoarseRelaxType ( void *solver, HYPRE_Int B_G_coarse_relax_type );
HYPRE_Int hypre_AMSComputePi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                               hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_AMSComputePixyz ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                  hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pix_ptr,
                                  hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_AMSComputeGPi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **GPi_ptr );
HYPRE_Int hypre_AMSSetup ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_AMSSolve ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ParCSRSubspacePrec ( hypre_ParCSRMatrix *A0, HYPRE_Int A0_relax_type,
                                     HYPRE_Int A0_relax_times, HYPRE_Real *A0_l1_norms, HYPRE_Real A0_relax_weight, HYPRE_Real A0_omega,
                                     HYPRE_Real A0_max_eig_est, HYPRE_Real A0_min_eig_est, HYPRE_Int A0_cheby_order,
                                     HYPRE_Real A0_cheby_fraction, hypre_ParCSRMatrix **A, HYPRE_Solver *B, HYPRE_PtrToSolverFcn *HB,
                                     hypre_ParCSRMatrix **P, hypre_ParVector **r, hypre_ParVector **g, hypre_ParVector *x,
                                     hypre_ParVector *y, hypre_ParVector *r0, hypre_ParVector *g0, char *cycle, hypre_ParVector *z );
HYPRE_Int hypre_AMSGetNumIterations ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_AMSGetFinalRelativeResidualNorm ( void *solver, HYPRE_Real *rel_resid_norm );
HYPRE_Int hypre_AMSProjectOutGradients ( void *solver, hypre_ParVector *x );
HYPRE_Int hypre_AMSConstructDiscreteGradient ( hypre_ParCSRMatrix *A, hypre_ParVector *x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, hypre_ParCSRMatrix **G_ptr );
HYPRE_Int hypre_AMSFEISetup ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                              hypre_ParVector *x, HYPRE_Int num_vert, HYPRE_Int num_local_vert, HYPRE_BigInt *vert_number,
                              HYPRE_Real *vert_coord, HYPRE_Int num_edges, HYPRE_BigInt *edge_vertex );
HYPRE_Int hypre_AMSFEIDestroy ( void *solver );
HYPRE_Int hypre_ParCSRComputeL1NormsThreads ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                              HYPRE_Int num_threads, HYPRE_Int *cf_marker, HYPRE_Real **l1_norm_ptr );

/* aux_interp.c */
HYPRE_Int hypre_alt_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_Int *OUT_marker );
HYPRE_Int hypre_big_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_BigInt offset, HYPRE_BigInt *OUT_marker );
HYPRE_Int hypre_ssort ( HYPRE_BigInt *data, HYPRE_Int n );
HYPRE_Int hypre_index_of_minimum ( HYPRE_BigInt *data, HYPRE_Int n );
void hypre_swap_int ( HYPRE_BigInt *data, HYPRE_Int a, HYPRE_Int b );
void hypre_initialize_vecs ( HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc,
                             HYPRE_BigInt *offd_ftc, HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF );
/*HYPRE_Int hypre_new_offd_nodes(HYPRE_Int **found , HYPRE_Int num_cols_A_offd , HYPRE_Int *A_ext_i , HYPRE_Int *A_ext_j, HYPRE_Int num_cols_S_offd, HYPRE_Int *col_map_offd, HYPRE_Int col_1, HYPRE_Int col_n, HYPRE_Int *Sop_i, HYPRE_Int *Sop_j, HYPRE_Int *CF_marker_offd );*/
HYPRE_Int hypre_exchange_marker(hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *IN_marker,
                                HYPRE_Int *OUT_marker);
HYPRE_Int hypre_exchange_interp_data( HYPRE_Int **CF_marker_offd, HYPRE_Int **dof_func_offd,
                                      hypre_CSRMatrix **A_ext, HYPRE_Int *full_off_procNodes, hypre_CSRMatrix **Sop,
                                      hypre_ParCSRCommPkg **extend_comm_pkg, hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                      hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                      HYPRE_Int skip_fine_or_same_sign);
void hypre_build_interp_colmap(hypre_ParCSRMatrix *P, HYPRE_Int full_off_procNodes,
                               HYPRE_Int *tmp_CF_marker_offd, HYPRE_BigInt *fine_to_coarse_offd);

/* block_tridiag.c */
void *hypre_BlockTridiagCreate ( void );
HYPRE_Int hypre_BlockTridiagDestroy ( void *data );
HYPRE_Int hypre_BlockTridiagSetup ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSolve ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSetIndexSet ( void *data, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int hypre_BlockTridiagSetAMGStrengthThreshold ( void *data, HYPRE_Real thresh );
HYPRE_Int hypre_BlockTridiagSetAMGNumSweeps ( void *data, HYPRE_Int nsweeps );
HYPRE_Int hypre_BlockTridiagSetAMGRelaxType ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BlockTridiagSetPrintLevel ( void *data, HYPRE_Int print_level );

/* driver.c */
HYPRE_Int BuildParFromFile ( HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                             HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian ( HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                              HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv ( HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                            HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParFromOneFile ( HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildRhsParFromOneFile ( HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                   HYPRE_ParCSRMatrix A, HYPRE_ParVector *b_ptr );
HYPRE_Int BuildParLaplacian9pt ( HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                 HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt ( HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                  HYPRE_ParCSRMatrix *A_ptr );

/* gen_redcs_mat.c */
HYPRE_Int hypre_seqAMGSetup ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              HYPRE_Int coarse_threshold );
HYPRE_Int hypre_seqAMGCycle ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              hypre_ParVector **Par_F_array, hypre_ParVector **Par_U_array );
HYPRE_Int hypre_GenerateSubComm ( MPI_Comm comm, HYPRE_Int participate, MPI_Comm *new_comm_ptr );
void hypre_merge_lists ( HYPRE_Int *list1, HYPRE_Int *list2, hypre_int *np1,
                         hypre_MPI_Datatype *dptr );

/* HYPRE_ads.c */
HYPRE_Int HYPRE_ADSCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ADSDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ADSSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSetDiscreteCurl ( HYPRE_Solver solver, HYPRE_ParCSRMatrix C );
HYPRE_Int HYPRE_ADSSetDiscreteGradient ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_ADSSetCoordinateVectors ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_ADSSetInterpolations ( HYPRE_Solver solver, HYPRE_ParCSRMatrix RT_Pi,
                                       HYPRE_ParCSRMatrix RT_Pix, HYPRE_ParCSRMatrix RT_Piy, HYPRE_ParCSRMatrix RT_Piz,
                                       HYPRE_ParCSRMatrix ND_Pi, HYPRE_ParCSRMatrix ND_Pix, HYPRE_ParCSRMatrix ND_Piy,
                                       HYPRE_ParCSRMatrix ND_Piz );
HYPRE_Int HYPRE_ADSSetMaxIter ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_ADSSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ADSSetCycleType ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ADSSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ADSSetSmoothingOptions ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, HYPRE_Real relax_weight, HYPRE_Real omega );
HYPRE_Int HYPRE_ADSSetChebySmoothingOptions ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              HYPRE_Real cheby_fraction );
HYPRE_Int HYPRE_ADSSetAMSOptions ( HYPRE_Solver solver, HYPRE_Int cycle_type,
                                   HYPRE_Int coarsen_type, HYPRE_Int agg_levels, HYPRE_Int relax_type, HYPRE_Real strength_threshold,
                                   HYPRE_Int interp_type, HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetAMGOptions ( HYPRE_Solver solver, HYPRE_Int coarsen_type,
                                   HYPRE_Int agg_levels, HYPRE_Int relax_type, HYPRE_Real strength_threshold, HYPRE_Int interp_type,
                                   HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *rel_resid_norm );

/* HYPRE_ame.c */
HYPRE_Int HYPRE_AMECreate ( HYPRE_Solver *esolver );
HYPRE_Int HYPRE_AMEDestroy ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESetup ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESolve ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESetAMSSolver ( HYPRE_Solver esolver, HYPRE_Solver ams_solver );
HYPRE_Int HYPRE_AMESetMassMatrix ( HYPRE_Solver esolver, HYPRE_ParCSRMatrix M );
HYPRE_Int HYPRE_AMESetBlockSize ( HYPRE_Solver esolver, HYPRE_Int block_size );
HYPRE_Int HYPRE_AMESetMaxIter ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetMaxPCGIter ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetTol ( HYPRE_Solver esolver, HYPRE_Real tol );
HYPRE_Int HYPRE_AMESetRTol ( HYPRE_Solver esolver, HYPRE_Real tol );
HYPRE_Int HYPRE_AMESetPrintLevel ( HYPRE_Solver esolver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMEGetEigenvalues ( HYPRE_Solver esolver, HYPRE_Real **eigenvalues );
HYPRE_Int HYPRE_AMEGetEigenvectors ( HYPRE_Solver esolver, HYPRE_ParVector **eigenvectors );

/* HYPRE_ams.c */
HYPRE_Int HYPRE_AMSCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_AMSDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSetDimension ( HYPRE_Solver solver, HYPRE_Int dim );
HYPRE_Int HYPRE_AMSSetDiscreteGradient ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_AMSSetCoordinateVectors ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_AMSSetEdgeConstantVectors ( HYPRE_Solver solver, HYPRE_ParVector Gx,
                                            HYPRE_ParVector Gy, HYPRE_ParVector Gz );
HYPRE_Int HYPRE_AMSSetInterpolations ( HYPRE_Solver solver, HYPRE_ParCSRMatrix Pi,
                                       HYPRE_ParCSRMatrix Pix, HYPRE_ParCSRMatrix Piy, HYPRE_ParCSRMatrix Piz );
HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_alpha );
HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_beta );
HYPRE_Int HYPRE_AMSSetInteriorNodes ( HYPRE_Solver solver, HYPRE_ParVector interior_nodes );
HYPRE_Int HYPRE_AMSSetProjectionFrequency ( HYPRE_Solver solver, HYPRE_Int projection_frequency );
HYPRE_Int HYPRE_AMSSetMaxIter ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMSSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_AMSSetCycleType ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_AMSSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMSSetSmoothingOptions ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, HYPRE_Real relax_weight, HYPRE_Real omega );
HYPRE_Int HYPRE_AMSSetChebySmoothingOptions ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              HYPRE_Real cheby_fraction );
HYPRE_Int HYPRE_AMSSetAlphaAMGOptions ( HYPRE_Solver solver, HYPRE_Int alpha_coarsen_type,
                                        HYPRE_Int alpha_agg_levels, HYPRE_Int alpha_relax_type, HYPRE_Real alpha_strength_threshold,
                                        HYPRE_Int alpha_interp_type, HYPRE_Int alpha_Pmax );
HYPRE_Int HYPRE_AMSSetAlphaAMGCoarseRelaxType ( HYPRE_Solver solver,
                                                HYPRE_Int alpha_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetBetaAMGOptions ( HYPRE_Solver solver, HYPRE_Int beta_coarsen_type,
                                       HYPRE_Int beta_agg_levels, HYPRE_Int beta_relax_type, HYPRE_Real beta_strength_threshold,
                                       HYPRE_Int beta_interp_type, HYPRE_Int beta_Pmax );
HYPRE_Int HYPRE_AMSSetBetaAMGCoarseRelaxType ( HYPRE_Solver solver,
                                               HYPRE_Int beta_coarse_relax_type );
HYPRE_Int HYPRE_AMSGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *rel_resid_norm );
HYPRE_Int HYPRE_AMSProjectOutGradients ( HYPRE_Solver solver, HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSConstructDiscreteGradient ( HYPRE_ParCSRMatrix A, HYPRE_ParVector x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, HYPRE_ParCSRMatrix *G );
HYPRE_Int HYPRE_AMSFEISetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x, HYPRE_BigInt *EdgeNodeList_, HYPRE_BigInt *NodeNumbers_, HYPRE_Int numEdges_,
                              HYPRE_Int numLocalNodes_, HYPRE_Int numNodes_, HYPRE_Real *NodalCoord_ );
HYPRE_Int HYPRE_AMSFEIDestroy ( HYPRE_Solver solver );

/* HYPRE_parcsr_amg.c */
HYPRE_Int HYPRE_BoomerAMGCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BoomerAMGDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolveT ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSetRestriction ( HYPRE_Solver solver, HYPRE_Int restr_par );
HYPRE_Int HYPRE_BoomerAMGSetIsTriangular ( HYPRE_Solver solver, HYPRE_Int is_triangular );
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR ( HYPRE_Solver solver, HYPRE_Int gmres_switch );
HYPRE_Int HYPRE_BoomerAMGSetMaxLevels ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGGetMaxLevels ( HYPRE_Solver solver, HYPRE_Int *max_levels );
HYPRE_Int HYPRE_BoomerAMGSetMaxCoarseSize ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMaxCoarseSize ( HYPRE_Solver solver, HYPRE_Int *max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMinCoarseSize ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMinCoarseSize ( HYPRE_Solver solver, HYPRE_Int *min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetSeqThreshold ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_BoomerAMGGetSeqThreshold ( HYPRE_Solver solver, HYPRE_Int *seq_threshold );
HYPRE_Int HYPRE_BoomerAMGSetRedundant ( HYPRE_Solver solver, HYPRE_Int redundant );
HYPRE_Int HYPRE_BoomerAMGGetRedundant ( HYPRE_Solver solver, HYPRE_Int *redundant );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenCutFactor( HYPRE_Solver solver, HYPRE_Int coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenCutFactor( HYPRE_Solver solver, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGSetStrongThreshold ( HYPRE_Solver solver, HYPRE_Real strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThreshold ( HYPRE_Solver solver, HYPRE_Real *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetStrongThresholdR ( HYPRE_Solver solver, HYPRE_Real strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThresholdR ( HYPRE_Solver solver, HYPRE_Real *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetFilterThresholdR ( HYPRE_Solver solver, HYPRE_Real filter_threshold );
HYPRE_Int HYPRE_BoomerAMGGetFilterThresholdR ( HYPRE_Solver solver, HYPRE_Real *filter_threshold );
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR ( HYPRE_Solver solver, HYPRE_Int gmres_switch );
HYPRE_Int HYPRE_BoomerAMGSetSabs ( HYPRE_Solver solver, HYPRE_Int Sabs );
HYPRE_Int HYPRE_BoomerAMGSetMaxRowSum ( HYPRE_Solver solver, HYPRE_Real max_row_sum );
HYPRE_Int HYPRE_BoomerAMGGetMaxRowSum ( HYPRE_Solver solver, HYPRE_Real *max_row_sum );
HYPRE_Int HYPRE_BoomerAMGSetTruncFactor ( HYPRE_Solver solver, HYPRE_Real trunc_factor );
HYPRE_Int HYPRE_BoomerAMGGetTruncFactor ( HYPRE_Solver solver, HYPRE_Real *trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetPMaxElmts ( HYPRE_Solver solver, HYPRE_Int P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGGetPMaxElmts ( HYPRE_Solver solver, HYPRE_Int *P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetJacobiTruncThreshold ( HYPRE_Solver solver,
                                                   HYPRE_Real jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGGetJacobiTruncThreshold ( HYPRE_Solver solver,
                                                   HYPRE_Real *jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGSetPostInterpType ( HYPRE_Solver solver, HYPRE_Int post_interp_type );
HYPRE_Int HYPRE_BoomerAMGGetPostInterpType ( HYPRE_Solver solver, HYPRE_Int *post_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetSCommPkgSwitch ( HYPRE_Solver solver, HYPRE_Real S_commpkg_switch );
HYPRE_Int HYPRE_BoomerAMGSetInterpType ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_BoomerAMGSetSepWeight ( HYPRE_Solver solver, HYPRE_Int sep_weight );
HYPRE_Int HYPRE_BoomerAMGSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BoomerAMGSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BoomerAMGGetMaxIter ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenType ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenType ( HYPRE_Solver solver, HYPRE_Int *coarsen_type );
HYPRE_Int HYPRE_BoomerAMGSetMeasureType ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_BoomerAMGGetMeasureType ( HYPRE_Solver solver, HYPRE_Int *measure_type );
HYPRE_Int HYPRE_BoomerAMGSetSetupType ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_BoomerAMGSetOldDefault ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGSetFCycle ( HYPRE_Solver solver, HYPRE_Int fcycle );
HYPRE_Int HYPRE_BoomerAMGGetFCycle ( HYPRE_Solver solver, HYPRE_Int *fcycle );
HYPRE_Int HYPRE_BoomerAMGSetCycleType ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_BoomerAMGGetCycleType ( HYPRE_Solver solver, HYPRE_Int *cycle_type );
HYPRE_Int HYPRE_BoomerAMGSetConvergeType ( HYPRE_Solver solver, HYPRE_Int type );
HYPRE_Int HYPRE_BoomerAMGGetConvergeType ( HYPRE_Solver solver, HYPRE_Int *type );
HYPRE_Int HYPRE_BoomerAMGSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_BoomerAMGGetTol ( HYPRE_Solver solver, HYPRE_Real *tol );
HYPRE_Int HYPRE_BoomerAMGSetNumGridSweeps ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetNumSweeps ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetCycleNumSweeps ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleNumSweeps ( HYPRE_Solver solver, HYPRE_Int *num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGInitGridRelaxation ( HYPRE_Int **num_grid_sweeps_ptr,
                                              HYPRE_Int **grid_relax_type_ptr, HYPRE_Int ***grid_relax_points_ptr, HYPRE_Int coarsen_type,
                                              HYPRE_Real **relax_weights_ptr, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxType ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_BoomerAMGSetRelaxType ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BoomerAMGSetCycleRelaxType ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleRelaxType ( HYPRE_Solver solver, HYPRE_Int *relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetRelaxOrder ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxPoints ( HYPRE_Solver solver, HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWeight ( HYPRE_Solver solver, HYPRE_Real *relax_weight );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWt ( HYPRE_Solver solver, HYPRE_Real relax_wt );
HYPRE_Int HYPRE_BoomerAMGSetLevelRelaxWt ( HYPRE_Solver solver, HYPRE_Real relax_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetOmega ( HYPRE_Solver solver, HYPRE_Real *omega );
HYPRE_Int HYPRE_BoomerAMGSetOuterWt ( HYPRE_Solver solver, HYPRE_Real outer_wt );
HYPRE_Int HYPRE_BoomerAMGSetLevelOuterWt ( HYPRE_Solver solver, HYPRE_Real outer_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetSmoothType ( HYPRE_Solver solver, HYPRE_Int smooth_type );
HYPRE_Int HYPRE_BoomerAMGGetSmoothType ( HYPRE_Solver solver, HYPRE_Int *smooth_type );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumLevels ( HYPRE_Solver solver, HYPRE_Int smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumLevels ( HYPRE_Solver solver, HYPRE_Int *smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumSweeps ( HYPRE_Solver solver, HYPRE_Int smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumSweeps ( HYPRE_Solver solver, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BoomerAMGGetLogging ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_BoomerAMGSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BoomerAMGGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *print_level );
HYPRE_Int HYPRE_BoomerAMGSetPrintFileName ( HYPRE_Solver solver, const char *print_file_name );
HYPRE_Int HYPRE_BoomerAMGSetDebugFlag ( HYPRE_Solver solver, HYPRE_Int debug_flag );
HYPRE_Int HYPRE_BoomerAMGGetDebugFlag ( HYPRE_Solver solver, HYPRE_Int *debug_flag );
HYPRE_Int HYPRE_BoomerAMGGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetCumNumIterations ( HYPRE_Solver solver, HYPRE_Int *cum_num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetResidual ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_BoomerAMGGetFinalRelativeResidualNorm ( HYPRE_Solver solver,
                                                        HYPRE_Real *rel_resid_norm );
HYPRE_Int HYPRE_BoomerAMGSetVariant ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGGetVariant ( HYPRE_Solver solver, HYPRE_Int *variant );
HYPRE_Int HYPRE_BoomerAMGSetOverlap ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_BoomerAMGGetOverlap ( HYPRE_Solver solver, HYPRE_Int *overlap );
HYPRE_Int HYPRE_BoomerAMGSetDomainType ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_BoomerAMGGetDomainType ( HYPRE_Solver solver, HYPRE_Int *domain_type );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzRlxWeight ( HYPRE_Solver solver, HYPRE_Real schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGGetSchwarzRlxWeight ( HYPRE_Solver solver,
                                               HYPRE_Real *schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzUseNonSymm ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_BoomerAMGSetSym ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_BoomerAMGSetLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetThreshold ( HYPRE_Solver solver, HYPRE_Real threshold );
HYPRE_Int HYPRE_BoomerAMGSetFilter ( HYPRE_Solver solver, HYPRE_Real filter );
HYPRE_Int HYPRE_BoomerAMGSetDropTol ( HYPRE_Solver solver, HYPRE_Real drop_tol );
HYPRE_Int HYPRE_BoomerAMGSetMaxNzPerRow ( HYPRE_Solver solver, HYPRE_Int max_nz_per_row );
HYPRE_Int HYPRE_BoomerAMGSetEuclidFile ( HYPRE_Solver solver, char *euclidfile );
HYPRE_Int HYPRE_BoomerAMGSetEuLevel ( HYPRE_Solver solver, HYPRE_Int eu_level );
HYPRE_Int HYPRE_BoomerAMGSetEuSparseA ( HYPRE_Solver solver, HYPRE_Real eu_sparse_A );
HYPRE_Int HYPRE_BoomerAMGSetEuBJ ( HYPRE_Solver solver, HYPRE_Int eu_bj );
HYPRE_Int HYPRE_BoomerAMGSetILUType( HYPRE_Solver solver, HYPRE_Int ilu_type);
HYPRE_Int HYPRE_BoomerAMGSetILULevel( HYPRE_Solver solver, HYPRE_Int ilu_lfil);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxRowNnz( HYPRE_Solver  solver, HYPRE_Int ilu_max_row_nnz);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxIter( HYPRE_Solver solver, HYPRE_Int ilu_max_iter);
HYPRE_Int HYPRE_BoomerAMGSetILUDroptol( HYPRE_Solver solver, HYPRE_Real ilu_droptol);
HYPRE_Int HYPRE_BoomerAMGSetILUTriSolve( HYPRE_Solver solver, HYPRE_Int ilu_tri_solve);
HYPRE_Int HYPRE_BoomerAMGSetILULowerJacobiIters( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_lower_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetILUUpperJacobiIters( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_upper_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetILULocalReordering( HYPRE_Solver solver, HYPRE_Int ilu_reordering_type);
HYPRE_Int HYPRE_BoomerAMGSetFSAIAlgoType ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAILocalSolveType ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxSteps ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxStepSize ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxNnzRow ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_BoomerAMGSetFSAINumLevels ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_BoomerAMGSetFSAIEigMaxIters ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_BoomerAMGSetFSAIThreshold ( HYPRE_Solver solver, HYPRE_Real threshold );
HYPRE_Int HYPRE_BoomerAMGSetFSAIKapTolerance ( HYPRE_Solver solver, HYPRE_Real kap_tolerance );
HYPRE_Int HYPRE_BoomerAMGSetNumFunctions ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_BoomerAMGGetNumFunctions ( HYPRE_Solver solver, HYPRE_Int *num_functions );
HYPRE_Int HYPRE_BoomerAMGSetNodal ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodalLevels ( HYPRE_Solver solver, HYPRE_Int nodal_levels );
HYPRE_Int HYPRE_BoomerAMGSetNodalDiag ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetKeepSameSign ( HYPRE_Solver solver, HYPRE_Int keep_same_sign );
HYPRE_Int HYPRE_BoomerAMGSetDofFunc ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_BoomerAMGSetNumPaths ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_BoomerAMGSetAggNumLevels ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetAggInterpType ( HYPRE_Solver solver, HYPRE_Int agg_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetAggTruncFactor ( HYPRE_Solver solver, HYPRE_Real agg_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAddTruncFactor ( HYPRE_Solver solver, HYPRE_Real add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetMultAddTruncFactor ( HYPRE_Solver solver, HYPRE_Real add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggP12TruncFactor ( HYPRE_Solver solver,
                                                HYPRE_Real agg_P12_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggPMaxElmts ( HYPRE_Solver solver, HYPRE_Int agg_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAddPMaxElmts ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetMultAddPMaxElmts ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxType ( HYPRE_Solver solver, HYPRE_Int add_rlx_type );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxWt ( HYPRE_Solver solver, HYPRE_Real add_rlx_wt );
HYPRE_Int HYPRE_BoomerAMGSetAggP12MaxElmts ( HYPRE_Solver solver, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetNumCRRelaxSteps ( HYPRE_Solver solver, HYPRE_Int num_CR_relax_steps );
HYPRE_Int HYPRE_BoomerAMGSetCRRate ( HYPRE_Solver solver, HYPRE_Real CR_rate );
HYPRE_Int HYPRE_BoomerAMGSetCRStrongTh ( HYPRE_Solver solver, HYPRE_Real CR_strong_th );
HYPRE_Int HYPRE_BoomerAMGSetADropTol( HYPRE_Solver solver, HYPRE_Real A_drop_tol  );
HYPRE_Int HYPRE_BoomerAMGSetADropType( HYPRE_Solver solver, HYPRE_Int A_drop_type  );
HYPRE_Int HYPRE_BoomerAMGSetISType ( HYPRE_Solver solver, HYPRE_Int IS_type );
HYPRE_Int HYPRE_BoomerAMGSetCRUseCG ( HYPRE_Solver solver, HYPRE_Int CR_use_CG );
HYPRE_Int HYPRE_BoomerAMGSetGSMG ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetNumSamples ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetCGCIts ( HYPRE_Solver solver, HYPRE_Int its );
HYPRE_Int HYPRE_BoomerAMGSetPlotGrids ( HYPRE_Solver solver, HYPRE_Int plotgrids );
HYPRE_Int HYPRE_BoomerAMGSetPlotFileName ( HYPRE_Solver solver, const char *plotfilename );
HYPRE_Int HYPRE_BoomerAMGSetCoordDim ( HYPRE_Solver solver, HYPRE_Int coorddim );
HYPRE_Int HYPRE_BoomerAMGSetCoordinates ( HYPRE_Solver solver, float *coordinates );
HYPRE_Int HYPRE_BoomerAMGGetGridHierarchy(HYPRE_Solver solver, HYPRE_Int *cgrid );
HYPRE_Int HYPRE_BoomerAMGSetChebyOrder ( HYPRE_Solver solver, HYPRE_Int order );
HYPRE_Int HYPRE_BoomerAMGSetChebyFraction ( HYPRE_Solver solver, HYPRE_Real ratio );
HYPRE_Int HYPRE_BoomerAMGSetChebyEigEst ( HYPRE_Solver solver, HYPRE_Int eig_est );
HYPRE_Int HYPRE_BoomerAMGSetChebyVariant ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSetChebyScale ( HYPRE_Solver solver, HYPRE_Int scale );
HYPRE_Int HYPRE_BoomerAMGSetInterpVectors ( HYPRE_Solver solver, HYPRE_Int num_vectors,
                                            HYPRE_ParVector *vectors );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecVariant ( HYPRE_Solver solver, HYPRE_Int num );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecQMax ( HYPRE_Solver solver, HYPRE_Int q_max );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecAbsQTrunc ( HYPRE_Solver solver, HYPRE_Real q_trunc );
HYPRE_Int HYPRE_BoomerAMGSetSmoothInterpVectors ( HYPRE_Solver solver,
                                                  HYPRE_Int smooth_interp_vectors );
HYPRE_Int HYPRE_BoomerAMGSetInterpRefine ( HYPRE_Solver solver, HYPRE_Int num_refine );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecFirstLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetAdditive ( HYPRE_Solver solver, HYPRE_Int additive );
HYPRE_Int HYPRE_BoomerAMGGetAdditive ( HYPRE_Solver solver, HYPRE_Int *additive );
HYPRE_Int HYPRE_BoomerAMGSetMultAdditive ( HYPRE_Solver solver, HYPRE_Int mult_additive );
HYPRE_Int HYPRE_BoomerAMGGetMultAdditive ( HYPRE_Solver solver, HYPRE_Int *mult_additive );
HYPRE_Int HYPRE_BoomerAMGSetSimple ( HYPRE_Solver solver, HYPRE_Int simple );
HYPRE_Int HYPRE_BoomerAMGGetSimple ( HYPRE_Solver solver, HYPRE_Int *simple );
HYPRE_Int HYPRE_BoomerAMGSetAddLastLvl ( HYPRE_Solver solver, HYPRE_Int add_last_lvl );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkinTol ( HYPRE_Solver solver, HYPRE_Real nongalerkin_tol );
HYPRE_Int HYPRE_BoomerAMGSetLevelNonGalerkinTol ( HYPRE_Solver solver, HYPRE_Real nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkTol ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                           HYPRE_Real *nongalerk_tol );
HYPRE_Int HYPRE_BoomerAMGSetRAP2 ( HYPRE_Solver solver, HYPRE_Int rap2 );
HYPRE_Int HYPRE_BoomerAMGSetModuleRAP2 ( HYPRE_Solver solver, HYPRE_Int mod_rap2 );
HYPRE_Int HYPRE_BoomerAMGSetKeepTranspose ( HYPRE_Solver solver, HYPRE_Int keepTranspose );
#ifdef HYPRE_USING_DSUPERLU
HYPRE_Int HYPRE_BoomerAMGSetDSLUThreshold ( HYPRE_Solver solver, HYPRE_Int slu_threshold );
#endif
HYPRE_Int HYPRE_BoomerAMGSetCpointsToKeep( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                           HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetCPoints( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetIsolatedFPoints( HYPRE_Solver solver, HYPRE_Int num_isolated_fpt,
                                             HYPRE_BigInt *isolated_fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetFPoints( HYPRE_Solver solver, HYPRE_Int num_fpt,
                                     HYPRE_BigInt *fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetCumNnzAP ( HYPRE_Solver solver, HYPRE_Real cum_nnz_AP );
HYPRE_Int HYPRE_BoomerAMGGetCumNnzAP ( HYPRE_Solver solver, HYPRE_Real *cum_nnz_AP );

/* HYPRE_parcsr_amgdd.c */
HYPRE_Int HYPRE_BoomerAMGDDSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDSetStartLevel ( HYPRE_Solver solver, HYPRE_Int start_level );
HYPRE_Int HYPRE_BoomerAMGDDGetStartLevel ( HYPRE_Solver solver, HYPRE_Int *start_level );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumCycles ( HYPRE_Solver solver, HYPRE_Int fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumCycles ( HYPRE_Solver solver, HYPRE_Int *fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDSetFACCycleType ( HYPRE_Solver solver, HYPRE_Int fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACCycleType ( HYPRE_Solver solver, HYPRE_Int *fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumRelax ( HYPRE_Solver solver, HYPRE_Int fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumRelax ( HYPRE_Solver solver, HYPRE_Int *fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxType ( HYPRE_Solver solver, HYPRE_Int fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxType ( HYPRE_Solver solver, HYPRE_Int *fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxWeight ( HYPRE_Solver solver, HYPRE_Real fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxWeight ( HYPRE_Solver solver, HYPRE_Real *fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDSetPadding ( HYPRE_Solver solver, HYPRE_Int padding );
HYPRE_Int HYPRE_BoomerAMGDDGetPadding ( HYPRE_Solver solver, HYPRE_Int *padding );
HYPRE_Int HYPRE_BoomerAMGDDSetNumGhostLayers ( HYPRE_Solver solver, HYPRE_Int num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDGetNumGhostLayers ( HYPRE_Solver solver, HYPRE_Int *num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDSetUserFACRelaxation( HYPRE_Solver solver,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int HYPRE_BoomerAMGDDGetAMG ( HYPRE_Solver solver, HYPRE_Solver *amg_solver );

/* HYPRE_parcsr_bicgstab.c */
HYPRE_Int HYPRE_ParCSRBiCGSTABCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                           HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm ( HYPRE_Solver solver,
                                                             HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetResidual ( HYPRE_Solver solver, HYPRE_ParVector *residual );

/* HYPRE_parcsr_block.c */
HYPRE_Int HYPRE_BlockTridiagCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BlockTridiagDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BlockTridiagSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSetIndexSet ( HYPRE_Solver solver, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int HYPRE_BlockTridiagSetAMGStrengthThreshold ( HYPRE_Solver solver, HYPRE_Real thresh );
HYPRE_Int HYPRE_BlockTridiagSetAMGNumSweeps ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BlockTridiagSetAMGRelaxType ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BlockTridiagSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );

/* HYPRE_parcsr_cgnr.c */
HYPRE_Int HYPRE_ParCSRCGNRCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCGNRDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCGNRSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRCGNRSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRCGNRSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                       HYPRE_PtrToParSolverFcn precondT, HYPRE_PtrToParSolverFcn precond_setup,
                                       HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCGNRGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCGNRSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCGNRGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );

/* HYPRE_parcsr_Euclid.c */
HYPRE_Int HYPRE_EuclidCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_EuclidDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_EuclidSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x );
HYPRE_Int HYPRE_EuclidSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector bb,
                              HYPRE_ParVector xx );
HYPRE_Int HYPRE_EuclidSetParams ( HYPRE_Solver solver, HYPRE_Int argc, char *argv []);
HYPRE_Int HYPRE_EuclidSetParamsFromFile ( HYPRE_Solver solver, char *filename );
HYPRE_Int HYPRE_EuclidSetLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_EuclidSetBJ ( HYPRE_Solver solver, HYPRE_Int bj );
HYPRE_Int HYPRE_EuclidSetStats ( HYPRE_Solver solver, HYPRE_Int eu_stats );
HYPRE_Int HYPRE_EuclidSetMem ( HYPRE_Solver solver, HYPRE_Int eu_mem );
HYPRE_Int HYPRE_EuclidSetSparseA ( HYPRE_Solver solver, HYPRE_Real sparse_A );
HYPRE_Int HYPRE_EuclidSetRowScale ( HYPRE_Solver solver, HYPRE_Int row_scale );
HYPRE_Int HYPRE_EuclidSetILUT ( HYPRE_Solver solver, HYPRE_Real ilut );

/* HYPRE_parcsr_flexgmres.c */
HYPRE_Int HYPRE_ParCSRFlexGMRESCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                            HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver,
                                                              HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetResidual ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC ( HYPRE_Solver solver,
                                             HYPRE_PtrToModifyPCFcn modify_pc );

/* HYPRE_parcsr_gmres.c */
HYPRE_Int HYPRE_ParCSRGMRESCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRGMRESSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                        HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRGMRESGetResidual ( HYPRE_Solver solver, HYPRE_ParVector *residual );


/*HYPRE_parcsr_cogmres.c*/
HYPRE_Int HYPRE_ParCSRCOGMRESCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCOGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCOGMRESSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRCOGMRESSetCGS2 ( HYPRE_Solver solver, HYPRE_Int cgs2 );
HYPRE_Int HYPRE_ParCSRCOGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                          HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCOGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCOGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRCOGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRCOGMRESGetResidual ( HYPRE_Solver solver, HYPRE_ParVector *residual );



/* HYPRE_parcsr_hybrid.c */
HYPRE_Int HYPRE_ParCSRHybridCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRHybridDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRHybridSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRHybridSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRHybridSetConvergenceTol ( HYPRE_Solver solver, HYPRE_Real cf_tol );
HYPRE_Int HYPRE_ParCSRHybridSetDSCGMaxIter ( HYPRE_Solver solver, HYPRE_Int dscg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetPCGMaxIter ( HYPRE_Solver solver, HYPRE_Int pcg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetSetupType ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_ParCSRHybridSetSolverType ( HYPRE_Solver solver, HYPRE_Int solver_type );
HYPRE_Int HYPRE_ParCSRHybridSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRHybridSetTwoNorm ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRHybridSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRHybridSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRHybridSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRHybridSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRHybridSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRHybridSetStrongThreshold ( HYPRE_Solver solver, HYPRE_Real strong_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetMaxRowSum ( HYPRE_Solver solver, HYPRE_Real max_row_sum );
HYPRE_Int HYPRE_ParCSRHybridSetTruncFactor ( HYPRE_Solver solver, HYPRE_Real trunc_factor );
HYPRE_Int HYPRE_ParCSRHybridSetPMaxElmts ( HYPRE_Solver solver, HYPRE_Int p_max );
HYPRE_Int HYPRE_ParCSRHybridSetMaxLevels ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_ParCSRHybridSetMeasureType ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_ParCSRHybridSetCoarsenType ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_ParCSRHybridSetInterpType ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_ParCSRHybridSetCycleType ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ParCSRHybridSetNumGridSweeps ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxType ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxPoints ( HYPRE_Solver solver,
                                                 HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_ParCSRHybridSetNumSweeps ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetCycleNumSweeps ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxType ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetKeepTranspose ( HYPRE_Solver solver, HYPRE_Int keepT );
HYPRE_Int HYPRE_ParCSRHybridSetCycleRelaxType ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxOrder ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_ParCSRHybridSetMaxCoarseSize ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetMinCoarseSize ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetSeqThreshold ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWt ( HYPRE_Solver solver, HYPRE_Real relax_wt );
HYPRE_Int HYPRE_ParCSRHybridSetLevelRelaxWt ( HYPRE_Solver solver, HYPRE_Real relax_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetOuterWt ( HYPRE_Solver solver, HYPRE_Real outer_wt );
HYPRE_Int HYPRE_ParCSRHybridSetLevelOuterWt ( HYPRE_Solver solver, HYPRE_Real outer_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWeight ( HYPRE_Solver solver, HYPRE_Real *relax_weight );
HYPRE_Int HYPRE_ParCSRHybridSetOmega ( HYPRE_Solver solver, HYPRE_Real *omega );
HYPRE_Int HYPRE_ParCSRHybridSetAggNumLevels ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_ParCSRHybridSetNumPaths ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_ParCSRHybridSetNumFunctions ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_ParCSRHybridSetNodal ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_ParCSRHybridSetDofFunc ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_ParCSRHybridSetNonGalerkinTol ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                                HYPRE_Real *nongalerkin_tol );
HYPRE_Int HYPRE_ParCSRHybridGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_its );
HYPRE_Int HYPRE_ParCSRHybridGetDSCGNumIterations ( HYPRE_Solver solver, HYPRE_Int *dscg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetPCGNumIterations ( HYPRE_Solver solver, HYPRE_Int *pcg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRHybridGetSetupSolveTime( HYPRE_Solver solver, HYPRE_Real *time );

/* HYPRE_parcsr_int.c */
HYPRE_Int hypre_ParSetRandomValues ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_ParPrintVector ( void *v, const char *file );
void *hypre_ParReadVector ( MPI_Comm comm, const char *file );
HYPRE_Int hypre_ParVectorSize ( void *x );
HYPRE_Int HYPRE_ParCSRMultiVectorPrint ( void *x_, const char *fileName );
void *HYPRE_ParCSRMultiVectorRead ( MPI_Comm comm, void *ii_, const char *fileName );
HYPRE_Int aux_maskCount ( HYPRE_Int n, HYPRE_Int *mask );
void aux_indexFromMask ( HYPRE_Int n, HYPRE_Int *mask, HYPRE_Int *index );
HYPRE_Int HYPRE_TempParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRSetupMatvec ( HYPRE_MatvecFunctions *mv );

/* HYPRE_parcsr_lgmres.c */
HYPRE_Int HYPRE_ParCSRLGMRESCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRLGMRESDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRLGMRESSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSetKDim ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetAugDim ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetMinIter ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRLGMRESGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRLGMRESSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRLGMRESGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRLGMRESGetResidual ( HYPRE_Solver solver, HYPRE_ParVector *residual );

/* HYPRE_parcsr_ParaSails.c */
HYPRE_Int HYPRE_ParCSRParaSailsCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRParaSailsDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRParaSailsSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSetParams ( HYPRE_Solver solver, HYPRE_Real thresh,
                                           HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParCSRParaSailsSetFilter ( HYPRE_Solver solver, HYPRE_Real filter );
HYPRE_Int HYPRE_ParCSRParaSailsGetFilter ( HYPRE_Solver solver, HYPRE_Real *filter );
HYPRE_Int HYPRE_ParCSRParaSailsSetSym ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParCSRParaSailsSetLoadbal ( HYPRE_Solver solver, HYPRE_Real loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsGetLoadbal ( HYPRE_Solver solver, HYPRE_Real *loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsSetReuse ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParCSRParaSailsSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParaSailsCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParaSailsDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParaSailsSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSetParams ( HYPRE_Solver solver, HYPRE_Real thresh, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetThresh ( HYPRE_Solver solver, HYPRE_Real thresh );
HYPRE_Int HYPRE_ParaSailsGetThresh ( HYPRE_Solver solver, HYPRE_Real *thresh );
HYPRE_Int HYPRE_ParaSailsSetNlevels ( HYPRE_Solver solver, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsGetNlevels ( HYPRE_Solver solver, HYPRE_Int *nlevels );
HYPRE_Int HYPRE_ParaSailsSetFilter ( HYPRE_Solver solver, HYPRE_Real filter );
HYPRE_Int HYPRE_ParaSailsGetFilter ( HYPRE_Solver solver, HYPRE_Real *filter );
HYPRE_Int HYPRE_ParaSailsSetSym ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParaSailsGetSym ( HYPRE_Solver solver, HYPRE_Int *sym );
HYPRE_Int HYPRE_ParaSailsSetLoadbal ( HYPRE_Solver solver, HYPRE_Real loadbal );
HYPRE_Int HYPRE_ParaSailsGetLoadbal ( HYPRE_Solver solver, HYPRE_Real *loadbal );
HYPRE_Int HYPRE_ParaSailsSetReuse ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParaSailsGetReuse ( HYPRE_Solver solver, HYPRE_Int *reuse );
HYPRE_Int HYPRE_ParaSailsSetLogging ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParaSailsGetLogging ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_ParaSailsBuildIJMatrix ( HYPRE_Solver solver, HYPRE_IJMatrix *pij_A );

/* HYPRE_parcsr_fsai.c */
HYPRE_Int HYPRE_FSAICreate ( HYPRE_Solver *solver);
HYPRE_Int HYPRE_FSAIDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_FSAISetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAISolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAISetAlgoType ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_FSAIGetAlgoType ( HYPRE_Solver solver, HYPRE_Int *algo_type );
HYPRE_Int HYPRE_FSAISetLocalSolveType ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_FSAIGetLocalSolveType ( HYPRE_Solver solver, HYPRE_Int *local_solve_type );
HYPRE_Int HYPRE_FSAISetMaxSteps ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_FSAIGetMaxSteps ( HYPRE_Solver solver, HYPRE_Int *max_steps );
HYPRE_Int HYPRE_FSAISetMaxStepSize ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_FSAIGetMaxStepSize ( HYPRE_Solver solver, HYPRE_Int *max_step_size );
HYPRE_Int HYPRE_FSAISetMaxNnzRow ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_FSAIGetMaxNnzRow ( HYPRE_Solver solver, HYPRE_Int *max_nnz_row );
HYPRE_Int HYPRE_FSAISetNumLevels ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_FSAIGetNumLevels ( HYPRE_Solver solver, HYPRE_Int *num_levels );
HYPRE_Int HYPRE_FSAISetThreshold ( HYPRE_Solver solver, HYPRE_Real threshold );
HYPRE_Int HYPRE_FSAIGetThreshold ( HYPRE_Solver solver, HYPRE_Real *threshold );
HYPRE_Int HYPRE_FSAISetKapTolerance ( HYPRE_Solver solver, HYPRE_Real kap_tolerance );
HYPRE_Int HYPRE_FSAIGetKapTolerance ( HYPRE_Solver solver, HYPRE_Real *kap_tolerance );
HYPRE_Int HYPRE_FSAISetTolerance ( HYPRE_Solver solver, HYPRE_Real tolerance );
HYPRE_Int HYPRE_FSAIGetTolerance ( HYPRE_Solver solver, HYPRE_Real *tolerance );
HYPRE_Int HYPRE_FSAISetOmega ( HYPRE_Solver solver, HYPRE_Real omega );
HYPRE_Int HYPRE_FSAIGetOmega ( HYPRE_Solver solver, HYPRE_Real *omega );
HYPRE_Int HYPRE_FSAISetMaxIterations ( HYPRE_Solver solver, HYPRE_Int max_iterations );
HYPRE_Int HYPRE_FSAIGetMaxIterations ( HYPRE_Solver solver, HYPRE_Int *max_iterations );
HYPRE_Int HYPRE_FSAISetEigMaxIters ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_FSAIGetEigMaxIters ( HYPRE_Solver solver, HYPRE_Int *eig_max_iters );
HYPRE_Int HYPRE_FSAISetZeroGuess ( HYPRE_Solver solver, HYPRE_Int zero_guess );
HYPRE_Int HYPRE_FSAIGetZeroGuess ( HYPRE_Solver solver, HYPRE_Int *zero_guess );
HYPRE_Int HYPRE_FSAISetPrintLevel ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_FSAIGetPrintLevel ( HYPRE_Solver solver, HYPRE_Int *print_level );

/* HYPRE_parcsr_pcg.c */
HYPRE_Int HYPRE_ParCSRPCGCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPCGDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPCGSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSetTol ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRPCGSetAbsoluteTol ( HYPRE_Solver solver, HYPRE_Real a_tol );
HYPRE_Int HYPRE_ParCSRPCGSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPCGSetStopCrit ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRPCGSetTwoNorm ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRPCGSetRelChange ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRPCGSetPrecond ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                      HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRPCGGetPrecond ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRPCGSetPrintLevel ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetLogging ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGGetNumIterations ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver, HYPRE_Real *norm );
HYPRE_Int HYPRE_ParCSRPCGGetResidual ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRDiagScaleSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector y,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRDiagScale ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA, HYPRE_ParVector Hy,
                                  HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );

/* HYPRE_parcsr_pilut.c */
HYPRE_Int HYPRE_ParCSRPilutCreate ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPilutDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPilutSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSetMaxIter ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPilutSetDropTolerance ( HYPRE_Solver solver, HYPRE_Real tol );
HYPRE_Int HYPRE_ParCSRPilutSetFactorRowSize ( HYPRE_Solver solver, HYPRE_Int size );

/* HYPRE_parcsr_schwarz.c */
HYPRE_Int HYPRE_SchwarzCreate ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_SchwarzDestroy ( HYPRE_Solver solver );
HYPRE_Int HYPRE_SchwarzSetup ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSolve ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSetVariant ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_SchwarzSetOverlap ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_SchwarzSetDomainType ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_SchwarzSetDomainStructure ( HYPRE_Solver solver, HYPRE_CSRMatrix domain_structure );
HYPRE_Int HYPRE_SchwarzSetNumFunctions ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_SchwarzSetNonSymm ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_SchwarzSetRelaxWeight ( HYPRE_Solver solver, HYPRE_Real relax_weight );
HYPRE_Int HYPRE_SchwarzSetDofFunc ( HYPRE_Solver solver, HYPRE_Int *dof_func );

/* par_add_cycle.c */
HYPRE_Int hypre_BoomerAMGAdditiveCycle ( void *amg_vdata );
HYPRE_Int hypre_CreateLambda ( void *amg_vdata );
HYPRE_Int hypre_CreateDinv ( void *amg_vdata );

/* par_amg.c */
void *hypre_BoomerAMGCreate ( void );
HYPRE_Int hypre_BoomerAMGDestroy ( void *data );
HYPRE_Int hypre_BoomerAMGSetRestriction ( void *data, HYPRE_Int restr_par );
HYPRE_Int hypre_BoomerAMGSetIsTriangular ( void *data, HYPRE_Int is_triangular );
HYPRE_Int hypre_BoomerAMGSetGMRESSwitchR ( void *data, HYPRE_Int gmres_switch );
HYPRE_Int hypre_BoomerAMGSetMaxLevels ( void *data, HYPRE_Int max_levels );
HYPRE_Int hypre_BoomerAMGGetMaxLevels ( void *data, HYPRE_Int *max_levels );
HYPRE_Int hypre_BoomerAMGSetMaxCoarseSize ( void *data, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMaxCoarseSize ( void *data, HYPRE_Int *max_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMinCoarseSize ( void *data, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMinCoarseSize ( void *data, HYPRE_Int *min_coarse_size );
HYPRE_Int hypre_BoomerAMGSetSeqThreshold ( void *data, HYPRE_Int seq_threshold );
HYPRE_Int hypre_BoomerAMGGetSeqThreshold ( void *data, HYPRE_Int *seq_threshold );
HYPRE_Int hypre_BoomerAMGSetCoarsenCutFactor( void *data, HYPRE_Int coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGGetCoarsenCutFactor( void *data, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGSetRedundant ( void *data, HYPRE_Int redundant );
HYPRE_Int hypre_BoomerAMGGetRedundant ( void *data, HYPRE_Int *redundant );
HYPRE_Int hypre_BoomerAMGSetStrongThreshold ( void *data, HYPRE_Real strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThreshold ( void *data, HYPRE_Real *strong_threshold );
HYPRE_Int hypre_BoomerAMGSetStrongThresholdR ( void *data, HYPRE_Real strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThresholdR ( void *data, HYPRE_Real *strong_threshold );
HYPRE_Int hypre_BoomerAMGSetFilterThresholdR ( void *data, HYPRE_Real filter_threshold );
HYPRE_Int hypre_BoomerAMGGetFilterThresholdR ( void *data, HYPRE_Real *filter_threshold );
HYPRE_Int hypre_BoomerAMGSetSabs ( void *data, HYPRE_Int Sabs );
HYPRE_Int hypre_BoomerAMGSetMaxRowSum ( void *data, HYPRE_Real max_row_sum );
HYPRE_Int hypre_BoomerAMGGetMaxRowSum ( void *data, HYPRE_Real *max_row_sum );
HYPRE_Int hypre_BoomerAMGSetTruncFactor ( void *data, HYPRE_Real trunc_factor );
HYPRE_Int hypre_BoomerAMGGetTruncFactor ( void *data, HYPRE_Real *trunc_factor );
HYPRE_Int hypre_BoomerAMGSetPMaxElmts ( void *data, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_BoomerAMGGetPMaxElmts ( void *data, HYPRE_Int *P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetJacobiTruncThreshold ( void *data, HYPRE_Real jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGGetJacobiTruncThreshold ( void *data, HYPRE_Real *jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGSetPostInterpType ( void *data, HYPRE_Int post_interp_type );
HYPRE_Int hypre_BoomerAMGGetPostInterpType ( void *data, HYPRE_Int *post_interp_type );
HYPRE_Int hypre_BoomerAMGSetInterpType ( void *data, HYPRE_Int interp_type );
HYPRE_Int hypre_BoomerAMGGetInterpType ( void *data, HYPRE_Int *interp_type );
HYPRE_Int hypre_BoomerAMGSetSepWeight ( void *data, HYPRE_Int sep_weight );
HYPRE_Int hypre_BoomerAMGSetMinIter ( void *data, HYPRE_Int min_iter );
HYPRE_Int hypre_BoomerAMGGetMinIter ( void *data, HYPRE_Int *min_iter );
HYPRE_Int hypre_BoomerAMGSetMaxIter ( void *data, HYPRE_Int max_iter );
HYPRE_Int hypre_BoomerAMGGetMaxIter ( void *data, HYPRE_Int *max_iter );
HYPRE_Int hypre_BoomerAMGSetCoarsenType ( void *data, HYPRE_Int coarsen_type );
HYPRE_Int hypre_BoomerAMGGetCoarsenType ( void *data, HYPRE_Int *coarsen_type );
HYPRE_Int hypre_BoomerAMGSetMeasureType ( void *data, HYPRE_Int measure_type );
HYPRE_Int hypre_BoomerAMGGetMeasureType ( void *data, HYPRE_Int *measure_type );
HYPRE_Int hypre_BoomerAMGSetSetupType ( void *data, HYPRE_Int setup_type );
HYPRE_Int hypre_BoomerAMGGetSetupType ( void *data, HYPRE_Int *setup_type );
HYPRE_Int hypre_BoomerAMGSetFCycle ( void *data, HYPRE_Int fcycle );
HYPRE_Int hypre_BoomerAMGGetFCycle ( void *data, HYPRE_Int *fcycle );
HYPRE_Int hypre_BoomerAMGSetCycleType ( void *data, HYPRE_Int cycle_type );
HYPRE_Int hypre_BoomerAMGGetCycleType ( void *data, HYPRE_Int *cycle_type );
HYPRE_Int hypre_BoomerAMGSetConvergeType ( void *data, HYPRE_Int type );
HYPRE_Int hypre_BoomerAMGGetConvergeType ( void *data, HYPRE_Int *type );
HYPRE_Int hypre_BoomerAMGSetTol ( void *data, HYPRE_Real tol );
HYPRE_Int hypre_BoomerAMGGetTol ( void *data, HYPRE_Real *tol );
HYPRE_Int hypre_BoomerAMGSetNumSweeps ( void *data, HYPRE_Int num_sweeps );
HYPRE_Int hypre_BoomerAMGSetCycleNumSweeps ( void *data, HYPRE_Int num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleNumSweeps ( void *data, HYPRE_Int *num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetNumGridSweeps ( void *data, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGGetNumGridSweeps ( void *data, HYPRE_Int **num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGSetRelaxType ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BoomerAMGSetCycleRelaxType ( void *data, HYPRE_Int relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleRelaxType ( void *data, HYPRE_Int *relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetRelaxOrder ( void *data, HYPRE_Int relax_order );
HYPRE_Int hypre_BoomerAMGGetRelaxOrder ( void *data, HYPRE_Int *relax_order );
HYPRE_Int hypre_BoomerAMGSetGridRelaxType ( void *data, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_BoomerAMGGetGridRelaxType ( void *data, HYPRE_Int **grid_relax_type );
HYPRE_Int hypre_BoomerAMGSetGridRelaxPoints ( void *data, HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_BoomerAMGGetGridRelaxPoints ( void *data, HYPRE_Int ***grid_relax_points );
HYPRE_Int hypre_BoomerAMGSetRelaxWeight ( void *data, HYPRE_Real *relax_weight );
HYPRE_Int hypre_BoomerAMGGetRelaxWeight ( void *data, HYPRE_Real **relax_weight );
HYPRE_Int hypre_BoomerAMGSetRelaxWt ( void *data, HYPRE_Real relax_weight );
HYPRE_Int hypre_BoomerAMGSetLevelRelaxWt ( void *data, HYPRE_Real relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelRelaxWt ( void *data, HYPRE_Real *relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetOmega ( void *data, HYPRE_Real *omega );
HYPRE_Int hypre_BoomerAMGGetOmega ( void *data, HYPRE_Real **omega );
HYPRE_Int hypre_BoomerAMGSetOuterWt ( void *data, HYPRE_Real omega );
HYPRE_Int hypre_BoomerAMGSetLevelOuterWt ( void *data, HYPRE_Real omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelOuterWt ( void *data, HYPRE_Real *omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetSmoothType ( void *data, HYPRE_Int smooth_type );
HYPRE_Int hypre_BoomerAMGGetSmoothType ( void *data, HYPRE_Int *smooth_type );
HYPRE_Int hypre_BoomerAMGSetSmoothNumLevels ( void *data, HYPRE_Int smooth_num_levels );
HYPRE_Int hypre_BoomerAMGGetSmoothNumLevels ( void *data, HYPRE_Int *smooth_num_levels );
HYPRE_Int hypre_BoomerAMGSetSmoothNumSweeps ( void *data, HYPRE_Int smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGGetSmoothNumSweeps ( void *data, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGSetLogging ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_BoomerAMGGetLogging ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_BoomerAMGSetPrintLevel ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_BoomerAMGGetPrintLevel ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_BoomerAMGSetPrintFileName ( void *data, const char *print_file_name );
HYPRE_Int hypre_BoomerAMGGetPrintFileName ( void *data, char **print_file_name );
HYPRE_Int hypre_BoomerAMGSetNumIterations ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_BoomerAMGSetDebugFlag ( void *data, HYPRE_Int debug_flag );
HYPRE_Int hypre_BoomerAMGGetDebugFlag ( void *data, HYPRE_Int *debug_flag );
HYPRE_Int hypre_BoomerAMGSetGSMG ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetNumSamples ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetCGCIts ( void *data, HYPRE_Int its );
HYPRE_Int hypre_BoomerAMGSetPlotGrids ( void *data, HYPRE_Int plotgrids );
HYPRE_Int hypre_BoomerAMGSetPlotFileName ( void *data, const char *plot_file_name );
HYPRE_Int hypre_BoomerAMGSetCoordDim ( void *data, HYPRE_Int coorddim );
HYPRE_Int hypre_BoomerAMGSetCoordinates ( void *data, float *coordinates );
HYPRE_Int hypre_BoomerAMGGetGridHierarchy(void *data, HYPRE_Int *cgrid );
HYPRE_Int hypre_BoomerAMGSetNumFunctions ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMGGetNumFunctions ( void *data, HYPRE_Int *num_functions );
HYPRE_Int hypre_BoomerAMGSetNodal ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodalLevels ( void *data, HYPRE_Int nodal_levels );
HYPRE_Int hypre_BoomerAMGSetNodalDiag ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetKeepSameSign ( void *data, HYPRE_Int keep_same_sign );
HYPRE_Int hypre_BoomerAMGSetNumPaths ( void *data, HYPRE_Int num_paths );
HYPRE_Int hypre_BoomerAMGSetAggNumLevels ( void *data, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_BoomerAMGSetAggInterpType ( void *data, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_BoomerAMGSetAggPMaxElmts ( void *data, HYPRE_Int agg_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetMultAddPMaxElmts ( void *data, HYPRE_Int add_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAddRelaxType ( void *data, HYPRE_Int add_rlx_type );
HYPRE_Int hypre_BoomerAMGSetAddRelaxWt ( void *data, HYPRE_Real add_rlx_wt );
HYPRE_Int hypre_BoomerAMGSetAggP12MaxElmts ( void *data, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggTruncFactor ( void *data, HYPRE_Real agg_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetMultAddTruncFactor ( void *data, HYPRE_Real add_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetAggP12TruncFactor ( void *data, HYPRE_Real agg_P12_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetNumCRRelaxSteps ( void *data, HYPRE_Int num_CR_relax_steps );
HYPRE_Int hypre_BoomerAMGSetCRRate ( void *data, HYPRE_Real CR_rate );
HYPRE_Int hypre_BoomerAMGSetCRStrongTh ( void *data, HYPRE_Real CR_strong_th );
HYPRE_Int hypre_BoomerAMGSetADropTol( void     *data, HYPRE_Real  A_drop_tol );
HYPRE_Int hypre_BoomerAMGSetADropType( void     *data, HYPRE_Int  A_drop_type );
HYPRE_Int hypre_BoomerAMGSetISType ( void *data, HYPRE_Int IS_type );
HYPRE_Int hypre_BoomerAMGSetCRUseCG ( void *data, HYPRE_Int CR_use_CG );
HYPRE_Int hypre_BoomerAMGSetNumPoints ( void *data, HYPRE_Int num_points );
HYPRE_Int hypre_BoomerAMGSetDofFunc ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_BoomerAMGSetPointDofMap ( void *data, HYPRE_Int *point_dof_map );
HYPRE_Int hypre_BoomerAMGSetDofPoint ( void *data, HYPRE_Int *dof_point );
HYPRE_Int hypre_BoomerAMGGetNumIterations ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BoomerAMGGetCumNumIterations ( void *data, HYPRE_Int *cum_num_iterations );
HYPRE_Int hypre_BoomerAMGGetResidual ( void *data, hypre_ParVector **resid );
HYPRE_Int hypre_BoomerAMGGetRelResidualNorm ( void *data, HYPRE_Real *rel_resid_norm );
HYPRE_Int hypre_BoomerAMGSetVariant ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGGetVariant ( void *data, HYPRE_Int *variant );
HYPRE_Int hypre_BoomerAMGSetOverlap ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_BoomerAMGGetOverlap ( void *data, HYPRE_Int *overlap );
HYPRE_Int hypre_BoomerAMGSetDomainType ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_BoomerAMGGetDomainType ( void *data, HYPRE_Int *domain_type );
HYPRE_Int hypre_BoomerAMGSetSchwarzRlxWeight ( void *data, HYPRE_Real schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGGetSchwarzRlxWeight ( void *data, HYPRE_Real *schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGSetSchwarzUseNonSymm ( void *data, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_BoomerAMGSetSym ( void *data, HYPRE_Int sym );
HYPRE_Int hypre_BoomerAMGSetLevel ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetThreshold ( void *data, HYPRE_Real thresh );
HYPRE_Int hypre_BoomerAMGSetFilter ( void *data, HYPRE_Real filter );
HYPRE_Int hypre_BoomerAMGSetDropTol ( void *data, HYPRE_Real drop_tol );
HYPRE_Int hypre_BoomerAMGSetMaxNzPerRow ( void *data, HYPRE_Int max_nz_per_row );
HYPRE_Int hypre_BoomerAMGSetEuclidFile ( void *data, char *euclidfile );
HYPRE_Int hypre_BoomerAMGSetEuLevel ( void *data, HYPRE_Int eu_level );
HYPRE_Int hypre_BoomerAMGSetEuSparseA ( void *data, HYPRE_Real eu_sparse_A );
HYPRE_Int hypre_BoomerAMGSetEuBJ ( void *data, HYPRE_Int eu_bj );
HYPRE_Int hypre_BoomerAMGSetILUType( void *data, HYPRE_Int ilu_type );
HYPRE_Int hypre_BoomerAMGSetILULevel( void *data, HYPRE_Int ilu_lfil );
HYPRE_Int hypre_BoomerAMGSetILUDroptol( void *data, HYPRE_Real ilu_droptol );
HYPRE_Int hypre_BoomerAMGSetILUTriSolve( void *data, HYPRE_Int ilu_tri_solve );
HYPRE_Int hypre_BoomerAMGSetILULowerJacobiIters( void *data, HYPRE_Int ilu_lower_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetILUUpperJacobiIters( void *data, HYPRE_Int ilu_upper_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetILUMaxIter( void *data, HYPRE_Int ilu_max_iter );
HYPRE_Int hypre_BoomerAMGSetILUMaxRowNnz( void *data, HYPRE_Int ilu_max_row_nnz );
HYPRE_Int hypre_BoomerAMGSetILULocalReordering( void *data, HYPRE_Int ilu_reordering_type );
HYPRE_Int hypre_BoomerAMGSetILUIterSetupType( void *data, HYPRE_Int ilu_iter_setup_type );
HYPRE_Int hypre_BoomerAMGSetILUIterSetupOption( void *data, HYPRE_Int ilu_iter_setup_option );
HYPRE_Int hypre_BoomerAMGSetILUIterSetupMaxIter( void *data, HYPRE_Int ilu_iter_setup_max_iter );
HYPRE_Int hypre_BoomerAMGSetILUIterSetupTolerance( void *data,
                                                   HYPRE_Real ilu_iter_setup_tolerance );
HYPRE_Int hypre_BoomerAMGSetFSAIAlgoType ( void *data, HYPRE_Int fsai_algo_type );
HYPRE_Int hypre_BoomerAMGSetFSAILocalSolveType ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxSteps ( void *data, HYPRE_Int fsai_max_steps );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxStepSize ( void *data, HYPRE_Int fsai_max_step_size );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxNnzRow ( void *data, HYPRE_Int fsai_max_nnz_row );
HYPRE_Int hypre_BoomerAMGSetFSAINumLevels ( void *data, HYPRE_Int fsai_num_levels );
HYPRE_Int hypre_BoomerAMGSetFSAIEigMaxIters ( void *data, HYPRE_Int fsai_eig_max_iters );
HYPRE_Int hypre_BoomerAMGSetFSAIThreshold ( void *data, HYPRE_Real fsai_threshold );
HYPRE_Int hypre_BoomerAMGSetFSAIKapTolerance ( void *data, HYPRE_Real fsai_kap_tolerance );
HYPRE_Int hypre_BoomerAMGSetChebyOrder ( void *data, HYPRE_Int order );
HYPRE_Int hypre_BoomerAMGSetChebyFraction ( void *data, HYPRE_Real ratio );
HYPRE_Int hypre_BoomerAMGSetChebyEigEst ( void *data, HYPRE_Int eig_est );
HYPRE_Int hypre_BoomerAMGSetChebyVariant ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetChebyScale ( void *data, HYPRE_Int scale );
HYPRE_Int hypre_BoomerAMGSetInterpVectors ( void *solver, HYPRE_Int num_vectors,
                                            hypre_ParVector **interp_vectors );
HYPRE_Int hypre_BoomerAMGSetInterpVecVariant ( void *solver, HYPRE_Int var );
HYPRE_Int hypre_BoomerAMGSetInterpVecQMax ( void *data, HYPRE_Int q_max );
HYPRE_Int hypre_BoomerAMGSetInterpVecAbsQTrunc ( void *data, HYPRE_Real q_trunc );
HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors ( void *solver, HYPRE_Int smooth_interp_vectors );
HYPRE_Int hypre_BoomerAMGSetInterpRefine ( void *data, HYPRE_Int num_refine );
HYPRE_Int hypre_BoomerAMGSetInterpVecFirstLevel ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetAdditive ( void *data, HYPRE_Int additive );
HYPRE_Int hypre_BoomerAMGGetAdditive ( void *data, HYPRE_Int *additive );
HYPRE_Int hypre_BoomerAMGSetMultAdditive ( void *data, HYPRE_Int mult_additive );
HYPRE_Int hypre_BoomerAMGGetMultAdditive ( void *data, HYPRE_Int *mult_additive );
HYPRE_Int hypre_BoomerAMGSetSimple ( void *data, HYPRE_Int simple );
HYPRE_Int hypre_BoomerAMGGetSimple ( void *data, HYPRE_Int *simple );
HYPRE_Int hypre_BoomerAMGSetAddLastLvl ( void *data, HYPRE_Int add_last_lvl );
HYPRE_Int hypre_BoomerAMGSetNonGalerkinTol ( void *data, HYPRE_Real nongalerkin_tol );
HYPRE_Int hypre_BoomerAMGSetLevelNonGalerkinTol ( void *data, HYPRE_Real nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetNonGalerkTol ( void *data, HYPRE_Int nongalerk_num_tol,
                                           HYPRE_Real *nongalerk_tol );
HYPRE_Int hypre_BoomerAMGSetRAP2 ( void *data, HYPRE_Int rap2 );
HYPRE_Int hypre_BoomerAMGSetModuleRAP2 ( void *data, HYPRE_Int mod_rap2 );
HYPRE_Int hypre_BoomerAMGSetKeepTranspose ( void *data, HYPRE_Int keepTranspose );
#ifdef HYPRE_USING_DSUPERLU
HYPRE_Int hypre_BoomerAMGSetDSLUThreshold ( void *data, HYPRE_Int slu_threshold );
#endif
HYPRE_Int hypre_BoomerAMGSetCPoints( void *data, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int  num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index );
HYPRE_Int hypre_BoomerAMGSetFPoints( void *data, HYPRE_Int isolated, HYPRE_Int num_points,
                                     HYPRE_BigInt *indices );
HYPRE_Int hypre_BoomerAMGSetCumNnzAP ( void *data, HYPRE_Real cum_nnz_AP );
HYPRE_Int hypre_BoomerAMGGetCumNnzAP ( void *data, HYPRE_Real *cum_nnz_AP );

/* par_amg_setup.c */
HYPRE_Int hypre_BoomerAMGSetup ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );

/* par_amg_solve.c */
HYPRE_Int hypre_BoomerAMGSolve ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );

/* par_amg_solveT.c */
HYPRE_Int hypre_BoomerAMGSolveT ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGCycleT ( void *amg_vdata, hypre_ParVector **F_array,
                                  hypre_ParVector **U_array );
HYPRE_Int hypre_BoomerAMGRelaxT ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                  HYPRE_Int relax_type, HYPRE_Int relax_points, HYPRE_Real relax_weight, hypre_ParVector *u,
                                  hypre_ParVector *Vtemp );

/* par_cgc_coarsen.c */
HYPRE_Int hypre_BoomerAMGCoarsenCGCb ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cgc_its, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenCGC ( hypre_ParCSRMatrix *S, HYPRE_Int numberofgrids,
                                      HYPRE_Int coarsen_type, HYPRE_Int *CF_marker );
HYPRE_Int hypre_AmgCGCPrepare ( hypre_ParCSRMatrix *S, HYPRE_Int nlocal, HYPRE_Int *CF_marker,
                                HYPRE_Int **CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_Int **vrange );
//HYPRE_Int hypre_AmgCGCPrepare ( hypre_ParCSRMatrix *S , HYPRE_Int nlocal , HYPRE_Int *CF_marker , HYPRE_BigInt **CF_marker_offd , HYPRE_Int coarsen_type , HYPRE_BigInt **vrange );
HYPRE_Int hypre_AmgCGCGraphAssemble ( hypre_ParCSRMatrix *S, HYPRE_Int *vertexrange,
                                      HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_IJMatrix *ijG );
HYPRE_Int hypre_AmgCGCChoose ( hypre_CSRMatrix *G, HYPRE_Int *vertexrange, HYPRE_Int mpisize,
                               HYPRE_Int **coarse );
HYPRE_Int hypre_AmgCGCBoundaryFix ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                    HYPRE_Int *CF_marker_offd );

/* par_cg_relax_wt.c */
HYPRE_Int hypre_BoomerAMGCGRelaxWt ( void *amg_vdata, HYPRE_Int level, HYPRE_Int num_cg_sweeps,
                                     HYPRE_Real *rlx_wt_ptr );
HYPRE_Int hypre_Bisection ( HYPRE_Int n, HYPRE_Real *diag, HYPRE_Real *offd, HYPRE_Real y,
                            HYPRE_Real z, HYPRE_Real tol, HYPRE_Int k, HYPRE_Real *ev_ptr );

/* par_cheby.c */
HYPRE_Int hypre_ParCSRRelax_Cheby_Setup ( hypre_ParCSRMatrix *A, HYPRE_Real max_eig,
                                          HYPRE_Real min_eig, HYPRE_Real fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          HYPRE_Real **coefs_ptr, HYPRE_Real **ds_ptr );
HYPRE_Int hypre_ParCSRRelax_Cheby_Solve ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                          HYPRE_Real *ds_data, HYPRE_Real *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                          hypre_ParVector *tmp_vec);

HYPRE_Int hypre_ParCSRRelax_Cheby_SolveHost ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                              HYPRE_Real *ds_data, HYPRE_Real *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                              hypre_ParVector *tmp_vec);

/* par_cheby_device.c */
HYPRE_Int hypre_ParCSRRelax_Cheby_SolveDevice ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                HYPRE_Real *ds_data, HYPRE_Real *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                                hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                                hypre_ParVector *tmp_vec);

/* par_coarsen.c */
HYPRE_Int hypre_BoomerAMGCoarsen ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int CF_init,
                                   HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenRuge ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenFalgout ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                          HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                          hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenHMIS ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMIS ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMISHost ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                           HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );

HYPRE_Int hypre_BoomerAMGCoarsenPMISDevice( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                            HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );

/* par_coarsen_device.c */
HYPRE_Int hypre_GetGlobalMeasureDevice( hypre_ParCSRMatrix *S, hypre_ParCSRCommPkg *comm_pkg,
                                        HYPRE_Int CF_init, HYPRE_Int aug_rand, HYPRE_Real *measure_diag, HYPRE_Real *measure_offd,
                                        HYPRE_Real *real_send_buf );

/* par_coarse_parms.c */
HYPRE_Int hypre_BoomerAMGCoarseParms ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                       HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                       hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGCoarseParmsHost ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                           HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                           hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGCoarseParmsDevice ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                             HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                             hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGInitDofFuncDevice( HYPRE_Int *dof_func, HYPRE_Int local_size,
                                            HYPRE_Int offset, HYPRE_Int num_functions );

/* par_coordinates.c */
float *hypre_GenerateCoordinates ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                   HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                   HYPRE_Int p, HYPRE_Int q, HYPRE_Int r, HYPRE_Int coorddim );

/* par_cr.c */
HYPRE_Int hypre_BoomerAMGCoarsenCR1 ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                      HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                      HYPRE_Int CRaddCpoints );
HYPRE_Int hypre_cr ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data, HYPRE_Int n, HYPRE_Int *cf,
                     HYPRE_Int rlx, HYPRE_Real omega, HYPRE_Real tg, HYPRE_Int mu );
HYPRE_Int hypre_GraphAdd ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index,
                           HYPRE_Int istack );
HYPRE_Int hypre_GraphRemove ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index );
HYPRE_Int hypre_IndepSetGreedy ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_IndepSetGreedyS ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_fptjaccr ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data,
                           HYPRE_Int n, HYPRE_Real *e0, HYPRE_Real omega, HYPRE_Real *e1 );
HYPRE_Int hypre_fptgscr ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data,
                          HYPRE_Int n, HYPRE_Real *e0, HYPRE_Real *e1 );
HYPRE_Int hypre_formu ( HYPRE_Int *cf, HYPRE_Int n, HYPRE_Real *e1, HYPRE_Int *A_i,
                        HYPRE_Real rho );
HYPRE_Int hypre_BoomerAMGIndepRS ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                   HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRSa ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                    HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMIS ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                     HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMISa ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMIS ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init, HYPRE_Int debug_flag,
                                     HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMISa ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGCoarsenCR ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                     HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                     HYPRE_Int num_functions, HYPRE_Int rlx_type, HYPRE_Real relax_weight, HYPRE_Real omega,
                                     HYPRE_Real theta, HYPRE_Solver smoother, hypre_ParCSRMatrix *AN, HYPRE_Int useCG,
                                     hypre_ParCSRMatrix *S );

/* par_cycle.c */
HYPRE_Int hypre_BoomerAMGCycle ( void *amg_vdata, hypre_ParVector **F_array,
                                 hypre_ParVector **U_array );

/* par_difconv.c */
HYPRE_ParCSRMatrix GenerateDifConv ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                     HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                     HYPRE_Real *value );

/* par_gsmg.c */
HYPRE_Int hypre_ParCSRMatrixFillSmooth ( HYPRE_Int nsamples, HYPRE_Real *samples,
                                         hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int num_functions, HYPRE_Int *dof_func );
HYPRE_Real hypre_ParCSRMatrixChooseThresh ( hypre_ParCSRMatrix *S );
HYPRE_Int hypre_ParCSRMatrixThreshold ( hypre_ParCSRMatrix *A, HYPRE_Real thresh );
HYPRE_Int hypre_BoomerAMGCreateSmoothVecs ( void *data, hypre_ParCSRMatrix *A, HYPRE_Int num_sweeps,
                                            HYPRE_Int level, HYPRE_Real **SmoothVecs_p );
HYPRE_Int hypre_BoomerAMGCreateSmoothDirs ( void *data, hypre_ParCSRMatrix *A,
                                            HYPRE_Real *SmoothVecs, HYPRE_Real thresh, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGNormalizeVecs ( HYPRE_Int n, HYPRE_Int num, HYPRE_Real *V );
HYPRE_Int hypre_BoomerAMGFitVectors ( HYPRE_Int ip, HYPRE_Int n, HYPRE_Int num, const HYPRE_Real *V,
                                      HYPRE_Int nc, const HYPRE_Int *ind, HYPRE_Real *val );
HYPRE_Int hypre_BoomerAMGBuildInterpLS ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int num_smooth, HYPRE_Real *SmoothVecs,
                                         hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpGSMG ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, HYPRE_Real trunc_factor, hypre_ParCSRMatrix **P_ptr );

/* par_indepset.c */
HYPRE_Int hypre_BoomerAMGIndepSetInit ( hypre_ParCSRMatrix *S, HYPRE_Real *measure_array,
                                        HYPRE_Int seq_rand );
HYPRE_Int hypre_BoomerAMGIndepSet ( hypre_ParCSRMatrix *S, HYPRE_Real *measure_array,
                                    HYPRE_Int *graph_array, HYPRE_Int graph_array_size, HYPRE_Int *graph_array_offd,
                                    HYPRE_Int graph_array_offd_size, HYPRE_Int *IS_marker, HYPRE_Int *IS_marker_offd );

HYPRE_Int hypre_BoomerAMGIndepSetInitDevice( hypre_ParCSRMatrix *S, HYPRE_Real *measure_array,
                                             HYPRE_Int aug_rand);

HYPRE_Int hypre_BoomerAMGIndepSetDevice( hypre_ParCSRMatrix *S, HYPRE_Real *measure_diag,
                                         HYPRE_Real *measure_offd, HYPRE_Int graph_diag_size, HYPRE_Int *graph_diag,
                                         HYPRE_Int *IS_marker_diag, HYPRE_Int *IS_marker_offd, hypre_ParCSRCommPkg *comm_pkg,
                                         HYPRE_Int *int_send_buf );

/* par_interp.c */
HYPRE_Int hypre_BoomerAMGBuildInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                       hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                       HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpHE ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildDirInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, HYPRE_Int interp_type,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildDirInterpDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                               hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                               HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, HYPRE_Int interp_type,
                                               hypre_ParCSRMatrix **P_ptr );

HYPRE_Int hypre_BoomerAMGInterpTruncation ( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor,
                                            HYPRE_Int max_elmts );
HYPRE_Int hypre_BoomerAMGInterpTruncationDevice( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor,
                                                 HYPRE_Int max_elmts );

HYPRE_Int hypre_BoomerAMGBuildInterpModUnk ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGTruncandBuild ( hypre_ParCSRMatrix *P, HYPRE_Real trunc_factor,
                                         HYPRE_Int max_elmts );
hypre_ParCSRMatrix *hypre_CreateC ( hypre_ParCSRMatrix *A, HYPRE_Real w );

HYPRE_Int hypre_BoomerAMGBuildInterpOnePntHost( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildInterpOnePntDevice( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                                  hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                  HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildInterpOnePnt( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);

/* par_jacobi_interp.c */
void hypre_BoomerAMGJacobiInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                   hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                   HYPRE_Int level, HYPRE_Real truncation_threshold, HYPRE_Real truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_1 ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                     hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker, HYPRE_Int level, HYPRE_Real truncation_threshold,
                                     HYPRE_Real truncation_threshold_minus, HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd,
                                     HYPRE_Real weight_AF );
void hypre_BoomerAMGTruncateInterp ( hypre_ParCSRMatrix *P, HYPRE_Real eps, HYPRE_Real dlt,
                                     HYPRE_Int *CF_marker );
HYPRE_Int hypre_ParCSRMatrix_dof_func_offd ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                             HYPRE_Int *dof_func, HYPRE_Int **dof_func_offd );

/* par_laplace_27pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian27pt ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                           HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                           HYPRE_Real *value );
HYPRE_Int hypre_map3 ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_BigInt iz, HYPRE_Int p, HYPRE_Int q,
                       HYPRE_Int r, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part,
                       HYPRE_BigInt *nz_part );

/* par_laplace_9pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian9pt ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_Int P, HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, HYPRE_Real *value );
HYPRE_BigInt hypre_map2 ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_Int p, HYPRE_Int q,
                          HYPRE_BigInt nx, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part );

/* par_laplace.c */
HYPRE_ParCSRMatrix GenerateLaplacian ( MPI_Comm comm, HYPRE_BigInt ix, HYPRE_BigInt ny,
                                       HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                       HYPRE_Real *value );
HYPRE_BigInt hypre_map ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_BigInt iz, HYPRE_Int p,
                         HYPRE_Int q, HYPRE_Int r, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt *nx_part,
                         HYPRE_BigInt *ny_part, HYPRE_BigInt *nz_part );
HYPRE_ParCSRMatrix GenerateSysLaplacian ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                               HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                               HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value );

/* par_lr_interp.c */
HYPRE_Int hypreDevice_extendWtoP ( HYPRE_Int P_nr_of_rows, HYPRE_Int W_nr_of_rows,
                                   HYPRE_Int W_nr_of_cols, HYPRE_Int *CF_marker,
                                   HYPRE_Int  W_diag_nnz, HYPRE_Int *W_diag_i,
                                   HYPRE_Int *W_diag_j, HYPRE_Complex *W_diag_data,
                                   HYPRE_Int *P_diag_i, HYPRE_Int *P_diag_j,
                                   HYPRE_Complex *P_diag_data, HYPRE_Int *W_offd_i,
                                   HYPRE_Int *P_offd_i );
HYPRE_Int hypre_BoomerAMGBuildStdInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                          HYPRE_Int max_elmts, HYPRE_Int sep_weight,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                            HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                            HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterpHost ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPICCInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                              HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                              HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFFInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                         HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                         HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFF1Interp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );

/* par_lr_interp_device.c */
HYPRE_Int hypre_BoomerAMGBuildExtInterpDevice(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions,
                                              HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                                              hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildExtPIInterpDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                 HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildExtPEInterpDevice(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions,
                                                HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                                                hypre_ParCSRMatrix  **P_ptr);

/* par_mod_lr_interp.c */
HYPRE_Int hypre_BoomerAMGBuildModExtInterp(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPIInterp(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPEInterp(hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr);

/* par_2s_interp.c */
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterpHost ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                        HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                        HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterpDevice ( hypre_ParCSRMatrix *A,
                                                          HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                          HYPRE_BigInt *num_old_cpts_global, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                    hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                    HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                    HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterpHost ( hypre_ParCSRMatrix *A,
                                                          HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                          HYPRE_BigInt *num_old_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                          HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterpDevice ( hypre_ParCSRMatrix *A,
                                                            HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                            HYPRE_BigInt *num_old_cpts_global, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                            HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                      hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                      HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                      HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );

/* par_mod_multi_interp.c */
HYPRE_Int hypre_BoomerAMGBuildModMultipass ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Real trunc_factor,
                                             HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModMultipassHost ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Real trunc_factor,
                                                 HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultipassPi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                      HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                      HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, HYPRE_Real *row_sums,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultiPi ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                  hypre_ParCSRMatrix *P, HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                  HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, HYPRE_Int num_functions,
                                  HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_BoomerAMGBuildModMultipassDevice ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                   hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Real trunc_factor,
                                                   HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                   hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultipassPiDevice ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                            HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                            HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, HYPRE_Real *row_sums,
                                            hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultiPiDevice ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                        hypre_ParCSRMatrix *P, HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                        HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, HYPRE_Int num_functions,
                                        HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd, hypre_ParCSRMatrix **Pi_ptr );

/* par_multi_interp.c */
HYPRE_Int hypre_BoomerAMGBuildMultipass ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildMultipassHost ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, HYPRE_Real trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                              hypre_ParCSRMatrix **P_ptr );

/* par_nodal_systems.c */
HYPRE_Int hypre_BoomerAMGCreateNodalA ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                        HYPRE_Int *dof_func, HYPRE_Int option, HYPRE_Int diag_option, hypre_ParCSRMatrix **AN_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCFS ( hypre_ParCSRMatrix *SN, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *CFN_marker, HYPRE_Int num_functions, HYPRE_Int nodal, HYPRE_Int keep_same_sign,
                                           hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCF ( HYPRE_Int *CFN_marker, HYPRE_Int num_functions,
                                          HYPRE_Int num_nodes, hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr );

/* par_nongalerkin.c */
HYPRE_Int hypre_GrabSubArray ( HYPRE_Int *indices, HYPRE_Int start, HYPRE_Int end,
                               HYPRE_BigInt *array, HYPRE_BigInt *output );
HYPRE_Int hypre_IntersectTwoArrays ( HYPRE_Int *x, HYPRE_Real *x_data, HYPRE_Int x_length,
                                     HYPRE_Int *y, HYPRE_Int y_length, HYPRE_Int *z, HYPRE_Real *output_x_data,
                                     HYPRE_Int *intersect_length );
HYPRE_Int hypre_IntersectTwoBigArrays ( HYPRE_BigInt *x, HYPRE_Real *x_data, HYPRE_Int x_length,
                                        HYPRE_BigInt *y, HYPRE_Int y_length, HYPRE_BigInt *z, HYPRE_Real *output_x_data,
                                        HYPRE_Int *intersect_length );
HYPRE_Int hypre_SortedCopyParCSRData ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
HYPRE_Int hypre_BoomerAMG_MyCreateS ( hypre_ParCSRMatrix *A, HYPRE_Real strength_threshold,
                                      HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSFromCFMarker(hypre_ParCSRMatrix    *A,
                                             HYPRE_Real strength_threshold, HYPRE_Real max_row_sum, HYPRE_Int *CF_marker,
                                             HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int SMRK, hypre_ParCSRMatrix    **S_ptr);
HYPRE_Int hypre_NonGalerkinIJBufferInit ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                          HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBigBufferInit ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                             HYPRE_BigInt *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferNewRow ( HYPRE_BigInt *ijbuf_rownums, HYPRE_Int *ijbuf_numcols,
                                            HYPRE_Int *ijbuf_rowcounter, HYPRE_BigInt new_row );
HYPRE_Int hypre_NonGalerkinIJBufferCompressRow ( HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter,
                                                 HYPRE_Real *ijbuf_data, HYPRE_BigInt *ijbuf_cols, HYPRE_BigInt *ijbuf_rownums,
                                                 HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompress ( HYPRE_MemoryLocation memory_location,
                                              HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_cnt,
                                              HYPRE_Int *ijbuf_rowcounter, HYPRE_Real **ijbuf_data, HYPRE_BigInt **ijbuf_cols,
                                              HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferWrite ( HYPRE_IJMatrix B, HYPRE_Int *ijbuf_cnt,
                                           HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_rowcounter, HYPRE_Real **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols,
                                           HYPRE_BigInt row_to_write, HYPRE_BigInt col_to_write, HYPRE_Real val_to_write );
HYPRE_Int hypre_NonGalerkinIJBufferEmpty ( HYPRE_IJMatrix B, HYPRE_Int ijbuf_size,
                                           HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter, HYPRE_Real **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
hypre_ParCSRMatrix * hypre_NonGalerkinSparsityPattern(hypre_ParCSRMatrix *R_IAP,
                                                      hypre_ParCSRMatrix *RAP, HYPRE_Int * CF_marker, HYPRE_Real droptol, HYPRE_Int sym_collapse,
                                                      HYPRE_Int collapse_beta );
HYPRE_Int hypre_BoomerAMGBuildNonGalerkinCoarseOperator( hypre_ParCSRMatrix **RAP_ptr,
                                                         hypre_ParCSRMatrix *AP, HYPRE_Real strong_threshold, HYPRE_Real max_row_sum,
                                                         HYPRE_Int num_functions, HYPRE_Int * dof_func_value, HYPRE_Int * CF_marker, HYPRE_Real droptol,
                                                         HYPRE_Int sym_collapse, HYPRE_Real lump_percent, HYPRE_Int collapse_beta );

/* par_rap.c */
HYPRE_Int hypre_BoomerAMGBuildCoarseOperator ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                               hypre_ParCSRMatrix *P, hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperatorKT ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keepTranspose, hypre_ParCSRMatrix **RAP_ptr );

/* par_rap_communication.c */
HYPRE_Int hypre_GetCommPkgRTFromCommPkgA ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *fine_to_coarse, HYPRE_Int *tmp_map_offd );
HYPRE_Int hypre_GenerateSendMapAndCommPkg ( MPI_Comm comm, HYPRE_Int num_sends, HYPRE_Int num_recvs,
                                            HYPRE_Int *recv_procs, HYPRE_Int *send_procs, HYPRE_Int *recv_vec_starts, hypre_ParCSRMatrix *A );

/* par_ge_device.c */
HYPRE_Int hypre_GaussElimSetupDevice ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                       HYPRE_Int solver_type );
HYPRE_Int hypre_GaussElimSolveDevice ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                       HYPRE_Int solver_type );

/* par_gauss_elim.c */
HYPRE_Int hypre_GaussElimSetup ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int solver_type );
HYPRE_Int hypre_GaussElimSolve ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int solver_type );

/* par_relax.c */
HYPRE_Int hypre_BoomerAMGRelax ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                 HYPRE_Int relax_type, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                 HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelaxHybridGaussSeidel_core( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                      HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                                      HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                                      HYPRE_Int GS_order, HYPRE_Int Symm, HYPRE_Int Skip_diag, HYPRE_Int forced_seq,
                                                      HYPRE_Int Topo_order );
HYPRE_Int hypre_BoomerAMGRelax0WeightedJacobi( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                               HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, hypre_ParVector *u,
                                               hypre_ParVector *Vtemp );

HYPRE_Int hypre_BoomerAMGRelaxHybridSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                         HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                         HYPRE_Int direction, HYPRE_Int symm, HYPRE_Int skip_diag, HYPRE_Int force_seq );

HYPRE_Int hypre_BoomerAMGRelax1GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );

HYPRE_Int hypre_BoomerAMGRelax2GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );

HYPRE_Int hypre_BoomerAMGRelax5ChaoticHybridGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );

HYPRE_Int hypre_BoomerAMGRelax3HybridGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax4HybridGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax6HybridSSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax7Jacobi( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                       HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real *l1_norms,
                                       hypre_ParVector *u, hypre_ParVector *Vtemp );

HYPRE_Int hypre_BoomerAMGRelax8HybridL1SSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                             HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                             HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax10TopoOrderedGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                                        hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax13HybridL1GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                                     HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax14HybridL1GaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                                     HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax88HybridL1SSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                              HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                              HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax89HybridL1SSOR( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                              HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real omega,
                                              HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax18WeightedL1Jacobi( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real *l1_norms,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp );

HYPRE_Int hypre_BoomerAMGRelaxKaczmarz( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Real omega,
                                        HYPRE_Real *l1_norms, hypre_ParVector *u );

HYPRE_Int hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                          HYPRE_Real relax_weight, HYPRE_Real omega,
                                                          HYPRE_Real *A_diag_diag, hypre_ParVector *u,
                                                          hypre_ParVector *r, hypre_ParVector *z,
                                                          HYPRE_Int choice );

HYPRE_Int hypre_BoomerAMGRelax11TwoStageGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     HYPRE_Real relax_weight, HYPRE_Real omega,
                                                     HYPRE_Real *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelax12TwoStageGaussSeidel( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     HYPRE_Real relax_weight, HYPRE_Real omega,
                                                     HYPRE_Real *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

HYPRE_Int hypre_BoomerAMGRelaxComputeL1Norms( hypre_ParCSRMatrix *A, HYPRE_Int relax_type,
                                              HYPRE_Int relax_order, HYPRE_Int coarsest_lvl,
                                              hypre_IntArray *CF_marker,
                                              HYPRE_Real **l1_norms_data_ptr );

/* par_relax_device.c */
HYPRE_Int hypre_BoomerAMGRelaxHybridGaussSeidelDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                       HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                       HYPRE_Real relax_weight, HYPRE_Real omega,
                                                       HYPRE_Real *l1_norms, hypre_ParVector *u,
                                                       hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                                       HYPRE_Int GS_order, HYPRE_Int Symm );

/* par_relax_interface.c */
HYPRE_Int hypre_BoomerAMGRelaxIF ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                   HYPRE_Int relax_type, HYPRE_Int relax_order, HYPRE_Int cycle_type, HYPRE_Real relax_weight,
                                   HYPRE_Real omega, HYPRE_Real *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp,
                                   hypre_ParVector *Ztemp );

/* par_relax_more.c */
HYPRE_Int hypre_ParCSRMaxEigEstimate ( hypre_ParCSRMatrix *A, HYPRE_Int scale, HYPRE_Real *max_eig,
                                       HYPRE_Real *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateHost ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                           HYPRE_Real *max_eig, HYPRE_Real *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCG ( hypre_ParCSRMatrix *A, HYPRE_Int scale, HYPRE_Int max_iter,
                                         HYPRE_Real *max_eig, HYPRE_Real *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCGHost ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                             HYPRE_Int max_iter, HYPRE_Real *max_eig, HYPRE_Real *min_eig );
HYPRE_Int hypre_ParCSRRelax_Cheby ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Real max_eig,
                                    HYPRE_Real min_eig, HYPRE_Real fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                    hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r );
HYPRE_Int hypre_BoomerAMGRelax_FCFJacobi ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, HYPRE_Real relax_weight,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_ParCSRRelax_CG ( HYPRE_Solver solver, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int num_its );
HYPRE_Int hypre_LINPACKcgtql1 ( HYPRE_Int *n, HYPRE_Real *d, HYPRE_Real *e, HYPRE_Int *ierr );
HYPRE_Real hypre_LINPACKcgpthy ( HYPRE_Real *a, HYPRE_Real *b );
HYPRE_Int hypre_ParCSRRelax_L1_Jacobi ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, HYPRE_Real relax_weight, HYPRE_Real *l1_norms,
                                        hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_LINPACKcgtql1(HYPRE_Int*, HYPRE_Real *, HYPRE_Real *, HYPRE_Int *);
HYPRE_Real hypre_LINPACKcgpthy(HYPRE_Real*, HYPRE_Real*);

/* par_relax_more_device.c */
HYPRE_Int hypre_ParCSRMaxEigEstimateDevice ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                             HYPRE_Real *max_eig, HYPRE_Real *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCGDevice ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                               HYPRE_Int max_iter, HYPRE_Real *max_eig, HYPRE_Real *min_eig );

/* par_rotate_7pt.c */
HYPRE_ParCSRMatrix GenerateRotate7pt ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_Int P,
                                       HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, HYPRE_Real alpha, HYPRE_Real eps );

/* par_scaled_matnorm.c */
HYPRE_Int hypre_ParCSRMatrixScaledNorm ( hypre_ParCSRMatrix *A, HYPRE_Real *scnorm );

/* par_schwarz.c */
void *hypre_SchwarzCreate ( void );
HYPRE_Int hypre_SchwarzDestroy ( void *data );
HYPRE_Int hypre_SchwarzSetup ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_SchwarzSolve ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_SchwarzCFSolve ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt );
HYPRE_Int hypre_SchwarzSetVariant ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_SchwarzSetDomainType ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_SchwarzSetOverlap ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_SchwarzSetNumFunctions ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_SchwarzSetNonSymm ( void *data, HYPRE_Int value );
HYPRE_Int hypre_SchwarzSetRelaxWeight ( void *data, HYPRE_Real relax_weight );
HYPRE_Int hypre_SchwarzSetDomainStructure ( void *data, hypre_CSRMatrix *domain_structure );
HYPRE_Int hypre_SchwarzSetScale ( void *data, HYPRE_Real *scale );
HYPRE_Int hypre_SchwarzReScale ( void *data, HYPRE_Int size, HYPRE_Real value );
HYPRE_Int hypre_SchwarzSetDofFunc ( void *data, HYPRE_Int *dof_func );

/* par_stats.c */
HYPRE_Int hypre_BoomerAMGSetupStats ( void *amg_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_BoomerAMGWriteSolverParams ( void *data );
const char* hypre_BoomerAMGGetProlongationName( hypre_ParAMGData *amg_data );
const char* hypre_BoomerAMGGetAggProlongationName( hypre_ParAMGData *amg_data );
const char* hypre_BoomerAMGGetCoarseningName( hypre_ParAMGData *amg_data );
const char* hypre_BoomerAMGGetCycleName( hypre_ParAMGData *amg_data );
HYPRE_Int hypre_BoomerAMGPrintGeneralInfo( hypre_ParAMGData *amg_data, HYPRE_Int shift );

/* par_strength.c */
HYPRE_Int hypre_BoomerAMGCreateS ( hypre_ParCSRMatrix *A, HYPRE_Real strength_threshold,
                                   HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabs ( hypre_ParCSRMatrix *A, HYPRE_Real strength_threshold,
                                      HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSCommPkg ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                          HYPRE_Int **col_offd_S_to_A_ptr );
HYPRE_Int hypre_BoomerAMGCreate2ndS ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                      HYPRE_Int num_paths, HYPRE_BigInt *coarse_row_starts, hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker ( hypre_IntArray *CF_marker,
                                           hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarkerHost ( hypre_IntArray *CF_marker,
                                               hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarkerDevice ( hypre_IntArray *CF_marker,
                                                 hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2 ( hypre_IntArray *CF_marker,
                                            hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2Host ( hypre_IntArray *CF_marker,
                                                hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2Device ( hypre_IntArray *CF_marker,
                                                  hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCreateSHost(hypre_ParCSRMatrix *A, HYPRE_Real strength_threshold,
                                     HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr);
HYPRE_Int hypre_BoomerAMGCreateSDevice(hypre_ParCSRMatrix *A, HYPRE_Int abs_soc,
                                       HYPRE_Real strength_threshold, HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                       hypre_ParCSRMatrix **S_ptr);
HYPRE_Int hypre_BoomerAMGCreateSabsHost ( hypre_ParCSRMatrix *A, HYPRE_Real strength_threshold,
                                          HYPRE_Real max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreate2ndSDevice( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                           HYPRE_Int num_paths, HYPRE_BigInt *coarse_row_starts, hypre_ParCSRMatrix **C_ptr);
HYPRE_Int hypre_BoomerAMGMakeSocFromSDevice( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S);


/* par_sv_interp.c */
HYPRE_Int hypre_BoomerAMGSmoothInterpVectors ( hypre_ParCSRMatrix *A, HYPRE_Int num_smooth_vecs,
                                               hypre_ParVector **smooth_vecs, HYPRE_Int smooth_steps );
HYPRE_Int hypre_BoomerAMGCoarsenInterpVectors ( hypre_ParCSRMatrix *P, HYPRE_Int num_smooth_vecs,
                                                hypre_ParVector **smooth_vecs, HYPRE_Int *CF_marker, hypre_ParVector ***new_smooth_vecs,
                                                HYPRE_Int expand_level, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMG_GMExpandInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_Int num_smooth_vecs, hypre_ParVector **smooth_vecs, HYPRE_Int *nf, HYPRE_Int *dof_func,
                                           hypre_IntArray **coarse_dof_func, HYPRE_Int variant, HYPRE_Int level, HYPRE_Real abs_trunc,
                                           HYPRE_Real *weights, HYPRE_Int q_max, HYPRE_Int *CF_marker, HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMGRefineInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                        HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                        HYPRE_Int level );

/* par_sv_interp_ln.c */
HYPRE_Int hypre_BoomerAMG_LNExpandInterp ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, hypre_IntArray **coarse_dof_func,
                                           HYPRE_Int *CF_marker, HYPRE_Int level, HYPRE_Real *weights, HYPRE_Int num_smooth_vecs,
                                           hypre_ParVector **smooth_vecs, HYPRE_Real abs_trunc, HYPRE_Int q_max,
                                           HYPRE_Int interp_vec_first_level );

/* par_sv_interp_lsfit.c */
HYPRE_Int hypre_BoomerAMGFitInterpVectors ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                            HYPRE_Int num_smooth_vecs, hypre_ParVector **smooth_vecs, hypre_ParVector **coarse_smooth_vecs,
                                            HYPRE_Real delta, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                            HYPRE_Int max_elmts, HYPRE_Real trunc_factor, HYPRE_Int variant, HYPRE_Int level );

/* partial.c */
HYPRE_Int hypre_BoomerAMGBuildPartialExtPIInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                   hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                   HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                   HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialStdInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                 HYPRE_Int max_elmts, HYPRE_Int sep_weight, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialExtInterp ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, HYPRE_Real trunc_factor,
                                                 HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );

/* par_vardifconv.c */
HYPRE_ParCSRMatrix GenerateVarDifConv ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                        HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                        HYPRE_Real eps, HYPRE_ParVector *rhs_ptr );
HYPRE_Real afun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real bfun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real cfun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real dfun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real efun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real ffun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real gfun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real rfun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real bndfun ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );

/* par_vardifconv_rs.c */
HYPRE_ParCSRMatrix GenerateRSVarDifConv ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          HYPRE_Real eps, HYPRE_ParVector *rhs_ptr, HYPRE_Int type );
HYPRE_Real afun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real bfun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real cfun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real dfun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real efun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real ffun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real gfun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real rfun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );
HYPRE_Real bndfun_rs ( HYPRE_Real xx, HYPRE_Real yy, HYPRE_Real zz );


/* pcg_par.c */
void *hypre_ParKrylovCAlloc ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
HYPRE_Int hypre_ParKrylovFree ( void *ptr );
void *hypre_ParKrylovCreateVector ( void *vvector );
void *hypre_ParKrylovCreateVectorArray ( HYPRE_Int n, void *vvector );
HYPRE_Int hypre_ParKrylovDestroyVector ( void *vvector );
void *hypre_ParKrylovMatvecCreate ( void *A, void *x );
HYPRE_Int hypre_ParKrylovMatvec ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x,
                                  HYPRE_Complex beta, void *y );
HYPRE_Int hypre_ParKrylovMatvecT ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x,
                                   HYPRE_Complex beta, void *y );
HYPRE_Int hypre_ParKrylovMatvecDestroy ( void *matvec_data );
HYPRE_Real hypre_ParKrylovInnerProd ( void *x, void *y );
HYPRE_Int hypre_ParKrylovMassInnerProd ( void *x, void **y, HYPRE_Int k, HYPRE_Int unroll,
                                         void *result );
HYPRE_Int hypre_ParKrylovMassDotpTwo ( void *x, void *y, void **z, HYPRE_Int k, HYPRE_Int unroll,
                                       void *result_x, void *result_y );
HYPRE_Int hypre_ParKrylovMassAxpy( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k,
                                   HYPRE_Int unroll);
HYPRE_Int hypre_ParKrylovCopyVector ( void *x, void *y );
HYPRE_Int hypre_ParKrylovClearVector ( void *x );
HYPRE_Int hypre_ParKrylovScaleVector ( HYPRE_Complex alpha, void *x );
HYPRE_Int hypre_ParKrylovAxpy ( HYPRE_Complex alpha, void *x, void *y );
HYPRE_Int hypre_ParKrylovCommInfo ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
HYPRE_Int hypre_ParKrylovIdentitySetup ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_ParKrylovIdentity ( void *vdata, void *A, void *b, void *x );

/* schwarz.c */
HYPRE_Int hypre_AMGNodalSchwarzSmoother ( hypre_CSRMatrix *A, HYPRE_Int num_functions,
                                          HYPRE_Int option, hypre_CSRMatrix **domain_structure_pointer );
HYPRE_Int hypre_ParMPSchwarzSolve ( hypre_ParCSRMatrix *par_A, hypre_CSRMatrix *A_boundary,
                                    hypre_ParVector *rhs_vector, hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x,
                                    HYPRE_Real relax_wt, HYPRE_Real *scale, hypre_ParVector *Vtemp, HYPRE_Int *pivots,
                                    HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                 hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, HYPRE_Real relax_wt,
                                 hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, HYPRE_Real relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzFWSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, HYPRE_Real relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFFWSolve ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                     hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, HYPRE_Real relax_wt,
                                     hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                     HYPRE_Int use_nonsymm );
HYPRE_Int transpose_matrix_create ( HYPRE_Int **i_face_element_pointer,
                                    HYPRE_Int **j_face_element_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                    HYPRE_Int num_elements, HYPRE_Int num_faces );
HYPRE_Int matrix_matrix_product ( HYPRE_Int **i_element_edge_pointer,
                                  HYPRE_Int **j_element_edge_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_edge, HYPRE_Int *j_face_edge, HYPRE_Int num_elements, HYPRE_Int num_faces,
                                  HYPRE_Int num_edges );
HYPRE_Int hypre_AMGCreateDomainDof ( hypre_CSRMatrix *A, HYPRE_Int domain_type, HYPRE_Int overlap,
                                     HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_CSRMatrix **domain_structure_pointer,
                                     HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AMGeAgglomerate ( HYPRE_Int *i_AE_element, HYPRE_Int *j_AE_element,
                                  HYPRE_Int *i_face_face, HYPRE_Int *j_face_face, HYPRE_Int *w_face_face, HYPRE_Int *i_face_element,
                                  HYPRE_Int *j_face_element, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_to_prefer_weight, HYPRE_Int *i_face_weight, HYPRE_Int num_faces,
                                  HYPRE_Int num_elements, HYPRE_Int *num_AEs_pointer );
HYPRE_Int hypre_update_entry ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_remove_entry ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_move_entry ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                             HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_matinv ( HYPRE_Real *x, HYPRE_Real *a, HYPRE_Int k );
HYPRE_Int hypre_parCorrRes ( hypre_ParCSRMatrix *A, hypre_ParVector *x, hypre_Vector *rhs,
                             hypre_Vector **tmp_ptr );
HYPRE_Int hypre_AdSchwarzSolve ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                 hypre_CSRMatrix *domain_structure, HYPRE_Real *scale, hypre_ParVector *par_x,
                                 hypre_ParVector *par_aux, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AdSchwarzCFSolve ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                   hypre_CSRMatrix *domain_structure, HYPRE_Real *scale, hypre_ParVector *par_x,
                                   hypre_ParVector *par_aux, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_GenerateScale ( hypre_CSRMatrix *domain_structure, HYPRE_Int num_variables,
                                HYPRE_Real relaxation_weight, HYPRE_Real **scale_pointer );
HYPRE_Int hypre_ParAdSchwarzSolve ( hypre_ParCSRMatrix *A, hypre_ParVector *F,
                                    hypre_CSRMatrix *domain_structure, HYPRE_Real *scale, hypre_ParVector *X, hypre_ParVector *Vtemp,
                                    HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAMGCreateDomainDof ( hypre_ParCSRMatrix *A, HYPRE_Int domain_type,
                                        HYPRE_Int overlap, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        hypre_CSRMatrix **domain_structure_pointer, HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParGenerateScale ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                   HYPRE_Real relaxation_weight, HYPRE_Real **scale_pointer );
HYPRE_Int hypre_ParGenerateHybridScale ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                         hypre_CSRMatrix **A_boundary_pointer, HYPRE_Real **scale_pointer );

/* par_restr.c,  par_lr_restr.c */
HYPRE_Int hypre_BoomerAMGBuildRestrAIR( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        HYPRE_Real filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr,
                                        HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_Int hypre_BoomerAMGBuildRestrDist2AIR( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Real filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr, HYPRE_Int AIR1_5,
                                             HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIR( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                               HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int NeumannDeg,
                                               HYPRE_Real strong_thresholdR, HYPRE_Real filter_thresholdR, HYPRE_Int debug_flag,
                                               hypre_ParCSRMatrix **R_ptr);
HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIRDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                     HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int NeumannDeg,
                                                     HYPRE_Real strong_thresholdR, HYPRE_Real filter_thresholdR, HYPRE_Int debug_flag,
                                                     hypre_ParCSRMatrix **R_ptr);
HYPRE_Int hypre_BoomerAMGCFMarkerTo1minus1Device( HYPRE_Int *CF_marker, HYPRE_Int size );

#ifdef HYPRE_USING_DSUPERLU
/* superlu.c */
HYPRE_Int hypre_SLUDistSetup( HYPRE_Solver *solver, hypre_ParCSRMatrix *A, HYPRE_Int print_level);
HYPRE_Int hypre_SLUDistSolve( void* solver, hypre_ParVector *b, hypre_ParVector *x);
HYPRE_Int hypre_SLUDistDestroy( void* solver);
#endif

/* par_mgr.c */
void *hypre_MGRCreate ( void );
HYPRE_Int hypre_MGRDestroy ( void *mgr_vdata );
HYPRE_Int hypre_MGRCycle( void *mgr_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );
void *hypre_MGRCreateFrelaxVcycleData( void );
HYPRE_Int hypre_MGRDestroyFrelaxVcycleData( void *mgr_vdata );
void *hypre_MGRCreateGSElimData( void );
HYPRE_Int hypre_MGRDestroyGSElimData( void *mgr_vdata );
HYPRE_Int hypre_MGRSetupFrelaxVcycleData( void *mgr_vdata, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          HYPRE_Int level );
HYPRE_Int hypre_MGRFrelaxVcycle ( void *mgr_vdata, hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRSetCpointsByBlock( void *mgr_vdata, HYPRE_Int  block_size,
                                      HYPRE_Int  max_num_levels,
                                      HYPRE_Int *block_num_coarse_points,
                                      HYPRE_Int  **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByContiguousBlock( void *mgr_vdata, HYPRE_Int block_size,
                                                HYPRE_Int  max_num_levels,
                                                HYPRE_BigInt *begin_idx_array,
                                                HYPRE_Int *block_num_coarse_points,
                                                HYPRE_Int **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByPointMarkerArray( void *mgr_vdata, HYPRE_Int block_size,
                                                 HYPRE_Int  max_num_levels,
                                                 HYPRE_Int *block_num_coarse_points,
                                                 HYPRE_Int **block_coarse_indexes,
                                                 HYPRE_Int *point_marker_array );
HYPRE_Int hypre_MGRCoarsen( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                            HYPRE_Int final_coarse_size, HYPRE_Int *final_coarse_indexes,
                            HYPRE_Int debug_flag, hypre_IntArray **CF_marker,
                            HYPRE_Int last_level );
HYPRE_Int hypre_MGRSetReservedCoarseNodes( void *mgr_vdata, HYPRE_Int reserved_coarse_size,
                                           HYPRE_BigInt *reserved_coarse_nodes );
HYPRE_Int hypre_MGRSetReservedCpointsLevelToKeep( void *mgr_vdata, HYPRE_Int level );
HYPRE_Int hypre_MGRSetMaxGlobalSmoothIters( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetGlobalSmoothType( void *mgr_vdata, HYPRE_Int iter_type );
HYPRE_Int hypre_MGRSetNonCpointsToFpoints( void *mgr_vdata, HYPRE_Int nonCptToFptFlag );
//HYPRE_Int hypre_MGRInitCFMarker(HYPRE_Int num_variables, HYPRE_Int *CF_marker, HYPRE_Int initial_coarse_size,HYPRE_Int *initial_coarse_indexes);
//HYPRE_Int hypre_MGRUpdateCoarseIndexes(HYPRE_Int num_variables, HYPRE_Int *CF_marker, HYPRE_Int initial_coarse_size,HYPRE_Int *initial_coarse_indexes);
HYPRE_Int hypre_ParCSRMatrixExtractBlockDiagHost( hypre_ParCSRMatrix *par_A, HYPRE_Int blk_size,
                                                  HYPRE_Int num_points, HYPRE_Int point_type,
                                                  HYPRE_Int *CF_marker, HYPRE_Int diag_size,
                                                  HYPRE_Int diag_type, HYPRE_Real *diag_data );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrix( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                             HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                             HYPRE_Int diag_type, hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrixHost( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                                 HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                                 HYPRE_Int diag_type,
                                                 hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_MGRSetCoarseSolver( void *mgr_vdata,
                                    HYPRE_Int (*cgrid_solver_solve)(void*, void*, void*, void*),
                                    HYPRE_Int (*cgrid_solver_setup)(void*, void*, void*, void*),
                                    void *coarse_grid_solver );
HYPRE_Int hypre_MGRSetFSolver( void *mgr_vdata,
                               HYPRE_Int (*fine_grid_solver_solve)(void*, void*, void*, void*),
                               HYPRE_Int (*fine_grid_solver_setup)(void*, void*, void*, void*),
                               void *fsolver );
HYPRE_Int hypre_MGRSetFSolverAtLevel( HYPRE_Int level, void *mgr_vdata, void *fsolver );
HYPRE_Int hypre_MGRSetup( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRSolve( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector  *u );
HYPRE_Int hypre_block_jacobi_scaling( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **B_ptr,
                                      void *mgr_vdata, HYPRE_Int debug_flag );
HYPRE_Int hypre_MGRBlockRelaxSolveDevice( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          hypre_ParVector *Vtemp, HYPRE_Real relax_weight );
HYPRE_Int hypre_MGRBlockRelaxSolve( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int n_block, HYPRE_Int left_size,
                                    HYPRE_Int method, HYPRE_Real *diaginv,
                                    hypre_ParVector *Vtemp );
HYPRE_Int hypre_BlockDiagInvLapack( HYPRE_Real *diag, HYPRE_Int N, HYPRE_Int blk_size );
HYPRE_Int hypre_MGRBlockRelaxSetup( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                    HYPRE_Real **diaginvptr );
//HYPRE_Int hypre_blockRelax(hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
//                           HYPRE_Int blk_size, HYPRE_Int reserved_coarse_size, HYPRE_Int method, hypre_ParVector *Vtemp,
//                           hypre_ParVector *Ztemp);
HYPRE_Int hypre_block_jacobi_solve( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int method, HYPRE_Real *diaginv,
                                    hypre_ParVector *Vtemp );
//HYPRE_Int hypre_MGRBuildAffRAP( MPI_Comm comm, HYPRE_Int local_num_variables, HYPRE_Int num_functions,
//HYPRE_Int *dof_func, HYPRE_Int *CF_marker, HYPRE_Int **coarse_dof_func_ptr, HYPRE_BigInt **coarse_pnts_global_ptr,
//hypre_ParCSRMatrix *A, HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_f_ptr, hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRGetSubBlock( hypre_ParCSRMatrix *A, HYPRE_Int *row_cf_marker,
                                HYPRE_Int *col_cf_marker, HYPRE_Int debug_flag,
                                hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRBuildAff( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                             hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRApproximateInverse( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **A_inv );
HYPRE_Int hypre_MGRAddVectorP( hypre_IntArray *CF_marker, HYPRE_Int point_type, HYPRE_Real a,
                               hypre_ParVector *fromVector, HYPRE_Real b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRAddVectorR( hypre_IntArray *CF_marker, HYPRE_Int point_type, HYPRE_Real a,
                               hypre_ParVector *fromVector, HYPRE_Real b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRTruncateAcfCPRDevice( hypre_ParCSRMatrix  *A_CF,
                                         hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRComputeNonGalerkinCoarseGrid(hypre_ParCSRMatrix *A_FF,
                                                hypre_ParCSRMatrix *A_FC,
                                                hypre_ParCSRMatrix *A_CF,
                                                hypre_ParCSRMatrix *A_CC,
                                                hypre_ParCSRMatrix *Wp, hypre_ParCSRMatrix *Wr,
                                                HYPRE_Int bsize, HYPRE_Int ordering,
                                                HYPRE_Int method, HYPRE_Int max_elmts,
                                                hypre_ParCSRMatrix **A_H_ptr);
HYPRE_Int hypre_MGRSetAffSolverType( void *systg_vdata, HYPRE_Int *aff_solver_type );
HYPRE_Int hypre_MGRSetCoarseSolverType( void *systg_vdata, HYPRE_Int coarse_solver_type );
HYPRE_Int hypre_MGRSetCoarseSolverIter( void *systg_vdata, HYPRE_Int coarse_solver_iter );
HYPRE_Int hypre_MGRSetFineSolverIter( void *systg_vdata, HYPRE_Int fine_solver_iter );
HYPRE_Int hypre_MGRSetFineSolverMaxLevels( void *systg_vdata, HYPRE_Int fine_solver_max_levels );
HYPRE_Int hypre_MGRSetMaxCoarseLevels( void *mgr_vdata, HYPRE_Int maxlev );
HYPRE_Int hypre_MGRSetBlockSize( void *mgr_vdata, HYPRE_Int bsize );
HYPRE_Int hypre_MGRSetRelaxType( void *mgr_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_MGRSetFRelaxMethod( void *mgr_vdata, HYPRE_Int relax_method );
HYPRE_Int hypre_MGRSetLevelFRelaxMethod( void *mgr_vdata, HYPRE_Int *relax_method );
HYPRE_Int hypre_MGRSetLevelFRelaxType( void *mgr_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_MGRSetLevelFRelaxNumFunctions( void *mgr_vdata, HYPRE_Int *num_functions );
HYPRE_Int hypre_MGRSetCoarseGridMethod( void *mgr_vdata, HYPRE_Int *cg_method );
HYPRE_Int hypre_MGRSetRestrictType( void *mgr_vdata, HYPRE_Int restrictType );
HYPRE_Int hypre_MGRSetLevelRestrictType( void *mgr_vdata, HYPRE_Int *restrictType );
HYPRE_Int hypre_MGRSetInterpType( void *mgr_vdata, HYPRE_Int interpType );
HYPRE_Int hypre_MGRSetLevelInterpType( void *mgr_vdata, HYPRE_Int *interpType );
HYPRE_Int hypre_MGRSetNumRelaxSweeps( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetLevelNumRelaxSweeps( void *mgr_vdata, HYPRE_Int *nsweeps );
HYPRE_Int hypre_MGRSetNumInterpSweeps( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRestrictSweeps( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetLevelSmoothType( void *mgr_vdata, HYPRE_Int *level_smooth_type );
HYPRE_Int hypre_MGRSetLevelSmoothIters( void *mgr_vdata, HYPRE_Int *level_smooth_iters );
HYPRE_Int hypre_MGRSetGlobalSmoothCycle( void *mgr_vdata, HYPRE_Int global_smooth_cycle );
HYPRE_Int hypre_MGRSetPrintLevel( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetFrelaxPrintLevel( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetCoarseGridPrintLevel( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetTruncateCoarseGridThreshold( void *mgr_vdata, HYPRE_Real threshold );
HYPRE_Int hypre_MGRSetBlockJacobiBlockSize( void *mgr_vdata, HYPRE_Int blk_size );
HYPRE_Int hypre_MGRSetLogging( void *mgr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_MGRSetMaxIter( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetPMaxElmts( void *mgr_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_MGRSetLevelPMaxElmts( void *mgr_vdata, HYPRE_Int *P_max_elmts );
HYPRE_Int hypre_MGRSetTol( void *mgr_vdata, HYPRE_Real tol );
HYPRE_Int hypre_MGRDataPrint(void *mgr_vdata);
#ifdef HYPRE_USING_DSUPERLU
void *hypre_MGRDirectSolverCreate( void );
HYPRE_Int hypre_MGRDirectSolverSetup( void *solver, hypre_ParCSRMatrix *A,
                                      hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRDirectSolverSolve( void *solver, hypre_ParCSRMatrix *A,
                                      hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRDirectSolverDestroy( void *solver );
#endif
// Accessor functions
HYPRE_Int hypre_MGRGetNumIterations( void *mgr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_MGRGetFinalRelativeResidualNorm( void *mgr_vdata, HYPRE_Real *res_norm );
HYPRE_Int hypre_MGRGetCoarseGridConvergenceFactor( void *mgr_data, HYPRE_Real *conv_factor );

/* par_mgr_interp.c */
HYPRE_Int hypre_MGRBuildInterp( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                                HYPRE_Int block_jacobi_bsize,
                                hypre_ParCSRMatrix **P_tr, HYPRE_Int method,
                                HYPRE_Int num_sweeps_post );
HYPRE_Int hypre_MGRBuildRestrict( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                  hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *A_CF,
                                  hypre_IntArray *CF_marker, HYPRE_BigInt *num_cpts_global,
                                  HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                                  HYPRE_Real strong_threshold, HYPRE_Real max_row_sum,
                                  HYPRE_Int blk_size, HYPRE_Int method,
                                  hypre_ParCSRMatrix **W_ptr, hypre_ParCSRMatrix **R_ptr,
                                  hypre_ParCSRMatrix **RT_ptr );
HYPRE_Int hypre_MGRBuildPFromWp( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                 HYPRE_Int *CF_marker, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPFromWpHost( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                     HYPRE_Int *CF_marker, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildBlockJacobiWp( hypre_ParCSRMatrix *A_FF, hypre_ParCSRMatrix *A_FC,
                                       HYPRE_Int blk_size, hypre_ParCSRMatrix **Wp_ptr );
HYPRE_Int hypre_MGRBuildPBlockJacobi( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                      hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *Wp,
                                      HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildP( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                           HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                           HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPHost( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                               HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                               hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildInterpApproximateInverse( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                  HYPRE_BigInt *num_cpts_global,
                                                  hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRTruncateAcfCPR( hypre_ParCSRMatrix *A_CF, hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRBuildRFromW( HYPRE_Int *C_map, HYPRE_Int *F_map,
                                HYPRE_BigInt global_num_rows_R, HYPRE_BigInt global_num_cols_R,
                                HYPRE_BigInt *row_starts_R, HYPRE_BigInt *col_starts_R,
                                hypre_ParCSRMatrix *W, hypre_ParCSRMatrix **R_ptr );
HYPRE_Int hypre_MGRBlockColLumpedRestrict( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                           hypre_ParCSRMatrix *A_CF, hypre_IntArray *CF_marker,
                                           HYPRE_Int block_dim, hypre_ParCSRMatrix **W_ptr,
                                           hypre_ParCSRMatrix **R_ptr);
HYPRE_Int hypre_MGRColLumpedRestrict(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                     hypre_ParCSRMatrix *A_CF, hypre_IntArray *CF_marker,
                                     hypre_ParCSRMatrix **W_ptr, hypre_ParCSRMatrix **R_ptr);

/* par_mgr_coarsen.c */
HYPRE_Int hypre_MGRCoarseParms( MPI_Comm comm, HYPRE_Int num_rows, hypre_IntArray *CF_marker,
                                HYPRE_BigInt *row_starts_cpts, HYPRE_BigInt *row_starts_fpts );

/* par_mgr_device.c */
HYPRE_Int hypre_MGRRelaxL1JacobiDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        HYPRE_Int *CF_marker_host, HYPRE_Int relax_points,
                                        HYPRE_Real relax_weight, HYPRE_Real *l1_norms,
                                        hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_MGRBuildPFromWpDevice( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                       HYPRE_Int *CF_marker, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_MGRBuildPDevice( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                 HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_ParCSRMatrixExtractBlockDiagDevice( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                                    HYPRE_Int num_points, HYPRE_Int point_type,
                                                    HYPRE_Int *CF_marker, HYPRE_Int diag_size,
                                                    HYPRE_Int diag_type, HYPRE_Int *B_diag_i,
                                                    HYPRE_Int *B_diag_j,
                                                    HYPRE_Complex *B_diag_data );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrixDevice( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                                   HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                                   HYPRE_Int diag_type,
                                                   hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_MGRComputeNonGalerkinCGDevice( hypre_ParCSRMatrix *A_FF, hypre_ParCSRMatrix *A_FC,
                                               hypre_ParCSRMatrix *A_CF, hypre_ParCSRMatrix *A_CC,
                                               hypre_ParCSRMatrix *Wp, hypre_ParCSRMatrix *Wr,
                                               HYPRE_Int blk_size, HYPRE_Int method,
                                               HYPRE_Complex threshold,
                                               hypre_ParCSRMatrix **A_H_ptr );

/* par_mgr_stats.c */
HYPRE_Int hypre_MGRSetupStats( void *mgr_vdata );

/* par_ilu.c */
void *hypre_ILUCreate ( void );
HYPRE_Int hypre_ILUDestroy ( void *ilu_vdata );
HYPRE_Int hypre_ILUSetLevelOfFill( void *ilu_vdata, HYPRE_Int lfil );
HYPRE_Int hypre_ILUSetMaxNnzPerRow( void *ilu_vdata, HYPRE_Int nzmax );
HYPRE_Int hypre_ILUSetDropThreshold( void *ilu_vdata, HYPRE_Real threshold );
HYPRE_Int hypre_ILUSetDropThresholdArray( void *ilu_vdata, HYPRE_Real *threshold );
HYPRE_Int hypre_ILUSetType( void *ilu_vdata, HYPRE_Int ilu_type );
HYPRE_Int hypre_ILUSetMaxIter( void *ilu_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_ILUSetTol( void *ilu_vdata, HYPRE_Real tol );
HYPRE_Int hypre_ILUSetIterativeSetupType( void *ilu_vdata, HYPRE_Int iter_setup_type );
HYPRE_Int hypre_ILUSetIterativeSetupOption( void *ilu_vdata, HYPRE_Int iter_setup_option );
HYPRE_Int hypre_ILUSetIterativeSetupMaxIter( void *ilu_vdata, HYPRE_Int iter_setup_max_iter );
HYPRE_Int hypre_ILUSetIterativeSetupTolerance( void *ilu_vdata, HYPRE_Real iter_setup_tolerance );
HYPRE_Int hypre_ILUGetIterativeSetupHistory( void *ilu_vdata,
                                             HYPRE_Complex **iter_setup_history );
HYPRE_Int hypre_ILUSetTriSolve( void *ilu_vdata, HYPRE_Int tri_solve );
HYPRE_Int hypre_ILUSetLowerJacobiIters( void *ilu_vdata, HYPRE_Int lower_jacobi_iters );
HYPRE_Int hypre_ILUSetUpperJacobiIters( void *ilu_vdata, HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSetPrintLevel( void *ilu_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_ILUSetLogging( void *ilu_vdata, HYPRE_Int logging );
HYPRE_Int hypre_ILUSetLocalReordering( void *ilu_vdata, HYPRE_Int ordering_type );
HYPRE_Int hypre_ILUSetSchurSolverMaxIter( void *ilu_vdata, HYPRE_Int ss_max_iter );
HYPRE_Int hypre_ILUSetSchurSolverTol( void *ilu_vdata, HYPRE_Real ss_tol );
HYPRE_Int hypre_ILUSetSchurSolverAbsoluteTol( void *ilu_vdata, HYPRE_Real ss_absolute_tol );
HYPRE_Int hypre_ILUSetSchurSolverLogging( void *ilu_vdata, HYPRE_Int ss_logging );
HYPRE_Int hypre_ILUSetSchurSolverPrintLevel( void *ilu_vdata, HYPRE_Int ss_print_level );
HYPRE_Int hypre_ILUSetSchurSolverRelChange( void *ilu_vdata, HYPRE_Int ss_rel_change );
HYPRE_Int hypre_ILUSetSchurPrecondILUType( void *ilu_vdata, HYPRE_Int sp_ilu_type );
HYPRE_Int hypre_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, HYPRE_Int sp_ilu_lfil );
HYPRE_Int hypre_ILUSetSchurPrecondILUMaxNnzPerRow( void *ilu_vdata,
                                                   HYPRE_Int sp_ilu_max_row_nnz );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, HYPRE_Real sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThresholdArray( void *ilu_vdata,
                                                         HYPRE_Real *sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondPrintLevel( void *ilu_vdata, HYPRE_Int sp_print_level );
HYPRE_Int hypre_ILUSetSchurPrecondMaxIter( void *ilu_vdata, HYPRE_Int sp_max_iter );
HYPRE_Int hypre_ILUSetSchurPrecondTriSolve( void *ilu_vdata, HYPRE_Int sp_tri_solve );
HYPRE_Int hypre_ILUSetSchurPrecondLowerJacobiIters( void *ilu_vdata,
                                                    HYPRE_Int sp_lower_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurPrecondUpperJacobiIters( void *ilu_vdata,
                                                    HYPRE_Int sp_upper_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurPrecondTol( void *ilu_vdata, HYPRE_Int sp_tol );
HYPRE_Int hypre_ILUSetSchurNSHDropThreshold( void *ilu_vdata, HYPRE_Real threshold );
HYPRE_Int hypre_ILUSetSchurNSHDropThresholdArray( void *ilu_vdata, HYPRE_Real *threshold );
HYPRE_Int hypre_ILUGetNumIterations( void *ilu_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ILUGetFinalRelativeResidualNorm( void *ilu_vdata, HYPRE_Real *res_norm );
HYPRE_Int hypre_ILUWriteSolverParams( void *ilu_vdata );

HYPRE_Int hypre_ILUMinHeapAddI( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIIIi( HYPRE_Int *heap, HYPRE_Int *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIRIi( HYPRE_Int *heap, HYPRE_Real *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxrHeapAddRabsI( HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveI( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIIIi( HYPRE_Int *heap, HYPRE_Int *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIRIi( HYPRE_Int *heap, HYPRE_Real *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxrHeapRemoveRabsI( HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxQSplitRabsI( HYPRE_Real *arrayR, HYPRE_Int *arrayI, HYPRE_Int left,
                                   HYPRE_Int bound, HYPRE_Int right );
HYPRE_Int hypre_ILUMaxRabs( HYPRE_Real *array_data, HYPRE_Int *array_j, HYPRE_Int start,
                            HYPRE_Int end, HYPRE_Int nLU, HYPRE_Int *rperm, HYPRE_Real *value,
                            HYPRE_Int *index, HYPRE_Real *l1_norm, HYPRE_Int *nnz );

HYPRE_Int hypre_ILUGetPermddPQPre( HYPRE_Int n, HYPRE_Int nLU, HYPRE_Int *A_diag_i,
                                   HYPRE_Int *A_diag_j, HYPRE_Real *A_diag_data,
                                   HYPRE_Real tol, HYPRE_Int *perm, HYPRE_Int *rperm,
                                   HYPRE_Int *pperm_pre, HYPRE_Int *qperm_pre, HYPRE_Int *nB );
HYPRE_Int hypre_ILUGetPermddPQ( hypre_ParCSRMatrix *A, HYPRE_Int **io_pperm, HYPRE_Int **io_qperm,
                                HYPRE_Real tol, HYPRE_Int *nB, HYPRE_Int *nI,
                                HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetInteriorExteriorPerm( hypre_ParCSRMatrix *A,
                                            HYPRE_MemoryLocation memory_location,
                                            HYPRE_Int **perm, HYPRE_Int *nLU,
                                            HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetLocalPerm( hypre_ParCSRMatrix *A, HYPRE_Int **perm_ptr,
                                 HYPRE_Int *nLU, HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUBuildRASExternalMatrix( hypre_ParCSRMatrix *A, HYPRE_Int *rperm,
                                           HYPRE_Int **E_i, HYPRE_Int **E_j, HYPRE_Real **E_data );
HYPRE_Int hypre_ILUSortOffdColmap( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ILULocalRCMBuildFinalPerm( HYPRE_Int start, HYPRE_Int end,
                                           HYPRE_Int * G_perm, HYPRE_Int *perm, HYPRE_Int *qperm,
                                           HYPRE_Int **permp, HYPRE_Int **qpermp );
HYPRE_Int hypre_ILULocalRCM( hypre_CSRMatrix *A, HYPRE_Int start, HYPRE_Int end,
                             HYPRE_Int **permp, HYPRE_Int **qpermp, HYPRE_Int sym );
HYPRE_Int hypre_ILULocalRCMMindegree( HYPRE_Int n, HYPRE_Int *degree,
                                      HYPRE_Int *marker, HYPRE_Int *rootp );
HYPRE_Int hypre_ILULocalRCMOrder( hypre_CSRMatrix *A, HYPRE_Int *perm );
HYPRE_Int hypre_ILULocalRCMFindPPNode( hypre_CSRMatrix *A, HYPRE_Int *rootp, HYPRE_Int *marker );
HYPRE_Int hypre_ILULocalRCMBuildLevel( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                       HYPRE_Int *level_i, HYPRE_Int *level_j, HYPRE_Int *nlevp );
HYPRE_Int hypre_ILULocalRCMNumbering( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                      HYPRE_Int *perm, HYPRE_Int *current_nump );
HYPRE_Int hypre_ILULocalRCMQsort( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end,
                                  HYPRE_Int *degree );
HYPRE_Int hypre_ILULocalRCMReverse( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end );

#if defined(HYPRE_USING_GPU)
HYPRE_Int hypre_ParILUSchurGMRESDummySolveDevice( void *ilu_vdata, void *ilu_vdata2,
                                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParILUSchurGMRESCommInfoDevice( void *ilu_vdata, HYPRE_Int *my_id,
                                                HYPRE_Int *num_procs );
HYPRE_Int hypre_ParILURAPSchurGMRESSolveDevice( void *ilu_vdata, void *ilu_vdata2,
                                                hypre_ParVector *par_f, hypre_ParVector *par_u );
HYPRE_Int hypre_ParILURAPSchurGMRESMatvecDevice( void *matvec_data, HYPRE_Complex alpha,
                                                 void *ilu_vdata, void *x,
                                                 HYPRE_Complex beta, void *y );
#endif
HYPRE_Int hypre_ParILURAPSchurGMRESCommInfoHost( void *ilu_vdata, HYPRE_Int *my_id,
                                                 HYPRE_Int *num_procs );
HYPRE_Int hypre_ParILURAPSchurGMRESSolveHost( void *ilu_vdata, void *ilu_vdata2,
                                              hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParILURAPSchurGMRESMatvecHost( void *matvec_data, HYPRE_Complex alpha,
                                               void *ilu_vdata, void *x,
                                               HYPRE_Complex beta, void *y );

void * hypre_NSHCreate( void );
HYPRE_Int hypre_NSHDestroy( void *data );
HYPRE_Int hypre_NSHWriteSolverParams( void *nsh_vdata );
HYPRE_Int hypre_NSHSetPrintLevel( void *nsh_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_NSHSetLogging( void *nsh_vdata, HYPRE_Int logging );
HYPRE_Int hypre_NSHSetMaxIter( void *nsh_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_NSHSetTol( void *nsh_vdata, HYPRE_Real tol );
HYPRE_Int hypre_NSHSetGlobalSolver( void *nsh_vdata, HYPRE_Int global_solver );
HYPRE_Int hypre_NSHSetDropThreshold( void *nsh_vdata, HYPRE_Real droptol );
HYPRE_Int hypre_NSHSetDropThresholdArray( void *nsh_vdata, HYPRE_Real *droptol );
HYPRE_Int hypre_NSHSetMRMaxIter( void *nsh_vdata, HYPRE_Int mr_max_iter );
HYPRE_Int hypre_NSHSetMRTol( void *nsh_vdata, HYPRE_Real mr_tol );
HYPRE_Int hypre_NSHSetMRMaxRowNnz( void *nsh_vdata, HYPRE_Int mr_max_row_nnz );
HYPRE_Int hypre_NSHSetColVersion( void *nsh_vdata, HYPRE_Int mr_col_version );
HYPRE_Int hypre_NSHSetNSHMaxIter( void *nsh_vdata, HYPRE_Int nsh_max_iter );
HYPRE_Int hypre_NSHSetNSHTol( void *nsh_vdata, HYPRE_Real nsh_tol );
HYPRE_Int hypre_NSHSetNSHMaxRowNnz( void *nsh_vdata, HYPRE_Int nsh_max_row_nnz );
HYPRE_Int hypre_CSRMatrixNormFro( hypre_CSRMatrix *A, HYPRE_Real *norm_io);
HYPRE_Int hypre_CSRMatrixResNormFro( hypre_CSRMatrix *A, HYPRE_Real *norm_io);
HYPRE_Int hypre_ParCSRMatrixNormFro( hypre_ParCSRMatrix *A, HYPRE_Real *norm_io);
HYPRE_Int hypre_ParCSRMatrixResNormFro( hypre_ParCSRMatrix *A, HYPRE_Real *norm_io);
HYPRE_Int hypre_CSRMatrixTrace( hypre_CSRMatrix *A, HYPRE_Real *trace_io);
HYPRE_Int hypre_CSRMatrixDropInplace( hypre_CSRMatrix *A, HYPRE_Real droptol,
                                      HYPRE_Int max_row_nnz );
HYPRE_Int hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal( hypre_CSRMatrix *matA,
                                                        hypre_CSRMatrix **M,
                                                        HYPRE_Real droptol, HYPRE_Real tol,
                                                        HYPRE_Real eps_tol, HYPRE_Int max_row_nnz,
                                                        HYPRE_Int max_iter,
                                                        HYPRE_Int print_level );
HYPRE_Int hypre_ILUParCSRInverseNSH( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **M,
                                     HYPRE_Real *droptol, HYPRE_Real mr_tol,
                                     HYPRE_Real nsh_tol, HYPRE_Real eps_tol,
                                     HYPRE_Int mr_max_row_nnz, HYPRE_Int nsh_max_row_nnz,
                                     HYPRE_Int mr_max_iter, HYPRE_Int nsh_max_iter,
                                     HYPRE_Int mr_col_version, HYPRE_Int print_level );

/* par_ilu_setup.c */
HYPRE_Int hypre_ILUSetup( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParILUExtractEBFC( hypre_CSRMatrix *A_diag, HYPRE_Int nLU,
                                   hypre_CSRMatrix **Bp, hypre_CSRMatrix **Cp,
                                   hypre_CSRMatrix **Ep, hypre_CSRMatrix **Fp );
HYPRE_Int hypre_ParILURAPReorder( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                  HYPRE_Int *rqperm, hypre_ParCSRMatrix **A_pq );
HYPRE_Int hypre_ILUSetupLDUtoCusparse( hypre_ParCSRMatrix *L, HYPRE_Real *D,
                                       hypre_ParCSRMatrix  *U, hypre_ParCSRMatrix **LDUp );
HYPRE_Int hypre_ILUSetupRAPMILU0( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **ALUp,
                                  HYPRE_Int modified );
HYPRE_Int hypre_ILUSetupRAPILU0Device( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                       HYPRE_Int nLU, hypre_ParCSRMatrix **Apermptr,
                                       hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **ALUptr,
                                       hypre_CSRMatrix **BLUptr, hypre_CSRMatrix **CLUptr,
                                       hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr,
                                       HYPRE_Int test_opt );
HYPRE_Int hypre_ILUSetupRAPILU0( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr, HYPRE_Real **Dptr,
                                 hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **mLptr,
                                 HYPRE_Real **mDptr, hypre_ParCSRMatrix **mUptr,
                                 HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILU0( hypre_ParCSRMatrix  *A, HYPRE_Int *perm, HYPRE_Int *qperm,
                              HYPRE_Int nLU, HYPRE_Int nI, hypre_ParCSRMatrix **Lptr,
                              HYPRE_Real **Dptr, hypre_ParCSRMatrix **Uptr,
                              hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupMILU0( hypre_ParCSRMatrix *A, HYPRE_Int *permp,
                               HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                               hypre_ParCSRMatrix **Lptr, HYPRE_Real **Dptr,
                               hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                               HYPRE_Int **u_end, HYPRE_Int modified );
HYPRE_Int hypre_ILUSetupILUKSymbolic( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                      HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                      HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                      HYPRE_Int *U_diag_i, HYPRE_Int *S_diag_i,
                                      HYPRE_Int **L_diag_j, HYPRE_Int **U_diag_j,
                                      HYPRE_Int **S_diag_j, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUK( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *permp,
                              HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                              hypre_ParCSRMatrix **Lptr, HYPRE_Real **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUT( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Real *tol,
                              HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU,
                              HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, HYPRE_Real **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_NSHSetup( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSetupILU0RAS( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 HYPRE_Real **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUKRASSymbolic( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                         HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                                         HYPRE_Int *E_i, HYPRE_Int *E_j, HYPRE_Int ext,
                                         HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                         HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                         HYPRE_Int *U_diag_i, HYPRE_Int **L_diag_j,
                                         HYPRE_Int **U_diag_j );
HYPRE_Int hypre_ILUSetupILUKRAS( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 HYPRE_Real **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUTRAS( hypre_ParCSRMatrix *A, HYPRE_Int lfil,
                                 HYPRE_Real *tol, HYPRE_Int *perm, HYPRE_Int nLU,
                                 hypre_ParCSRMatrix **Lptr, HYPRE_Real **Dptr,
                                 hypre_ParCSRMatrix **Uptr );

/* par_ilu_setup_device.c */
#if defined(HYPRE_USING_GPU)
HYPRE_Int hypre_ILUSetupDevice(hypre_ParILUData *ilu_data, hypre_ParCSRMatrix *A,
                               HYPRE_Int *perm_data, HYPRE_Int *qperm_data,
                               HYPRE_Int n, HYPRE_Int nLU, hypre_CSRMatrix **BLUptr,
                               hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **Eptr,
                               hypre_CSRMatrix **Fptr);
HYPRE_Int hypre_ILUSetupIterativeILU0Device( hypre_CSRMatrix *A, HYPRE_Int type,
                                             HYPRE_Int option, HYPRE_Int max_iter,
                                             HYPRE_Real tolerance, HYPRE_Int *num_iter_ptr,
                                             HYPRE_Real **history_ptr );
#endif

/* par_ilu_solve.c */
HYPRE_Int hypre_ILUSolve( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSolveSchurGMRES( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int *perm,
                                    HYPRE_Int *qperm, HYPRE_Int nLU,
                                    hypre_ParCSRMatrix *L, HYPRE_Real *D,
                                    hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                    hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                    HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                                    hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveSchurNSH( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                  hypre_ParCSRMatrix *L, HYPRE_Real *D,
                                  hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                  hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                  HYPRE_Solver schur_solver, hypre_ParVector *rhs,
                                  hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveLU( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                            hypre_ParCSRMatrix *L, HYPRE_Real *D, hypre_ParCSRMatrix *U,
                            hypre_ParVector *ftemp, hypre_ParVector *utemp );
HYPRE_Int hypre_ILUSolveLUIter( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                hypre_ParCSRMatrix *L, HYPRE_Real *D, hypre_ParCSRMatrix *U,
                                hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                HYPRE_Int lower_jacobi_iters, HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSolveLURAS( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                               HYPRE_Int *perm, hypre_ParCSRMatrix *L, HYPRE_Real *D,
                               hypre_ParCSRMatrix *U, hypre_ParVector *ftemp,
                               hypre_ParVector *utemp, HYPRE_Real *fext, HYPRE_Real *uext );
HYPRE_Int hypre_ILUSolveRAPGMRESHost( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                      hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                      hypre_ParCSRMatrix *L, HYPRE_Real *D, hypre_ParCSRMatrix *U,
                                      hypre_ParCSRMatrix *mL, HYPRE_Real *mD,
                                      hypre_ParCSRMatrix *mU, hypre_ParVector *ftemp,
                                      hypre_ParVector *utemp, hypre_ParVector *xtemp,
                                      hypre_ParVector *ytemp, HYPRE_Solver schur_solver,
                                      HYPRE_Solver schur_precond, hypre_ParVector *rhs,
                                      hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_NSHSolve( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_NSHSolveInverse( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, hypre_ParCSRMatrix *M,
                                 hypre_ParVector *ftemp, hypre_ParVector *utemp );

/* par_ilu_solve_device.c */
#if defined(HYPRE_USING_GPU)
HYPRE_Int hypre_ILUSolveLUDevice( hypre_ParCSRMatrix *A, hypre_CSRMatrix *matLU_d,
                                  hypre_ParVector *f, hypre_ParVector *u,
                                  HYPRE_Int *perm, hypre_ParVector *ftemp,
                                  hypre_ParVector *utemp );
HYPRE_Int hypre_ILUApplyLowerJacIterDevice( hypre_CSRMatrix *A, hypre_Vector *input,
                                            hypre_Vector *work, hypre_Vector *output,
                                            HYPRE_Int lower_jacobi_iters );
HYPRE_Int hypre_ILUApplyUpperJacIterDevice( hypre_CSRMatrix *A, hypre_Vector *input,
                                            hypre_Vector *work, hypre_Vector *output,
                                            hypre_Vector *diag, HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUApplyLowerUpperJacIterDevice( hypre_CSRMatrix *A, hypre_Vector *work1,
                                                 hypre_Vector *work2, hypre_Vector *inout,
                                                 hypre_Vector *diag, HYPRE_Int lower_jacobi_iters,
                                                 HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSolveLUIterDevice( hypre_ParCSRMatrix *A, hypre_CSRMatrix *matLU,
                                      hypre_ParVector *f, hypre_ParVector *u,
                                      HYPRE_Int *perm, hypre_ParVector *ftemp,
                                      hypre_ParVector *utemp, hypre_ParVector *xtemp,
                                      hypre_Vector **diag_ptr, HYPRE_Int lower_jacobi_iters,
                                      HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ParILUSchurGMRESMatvecDevice( void *matvec_data, HYPRE_Complex alpha,
                                              void *ilu_vdata, void *x,
                                              HYPRE_Complex beta, void *y );
HYPRE_Int hypre_ILUSolveSchurGMRESDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                          hypre_ParVector *u, HYPRE_Int *perm,
                                          HYPRE_Int nLU, hypre_ParCSRMatrix *S,
                                          hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                          HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                                          hypre_ParVector *rhs, hypre_ParVector *x,
                                          HYPRE_Int *u_end, hypre_CSRMatrix *matBLU_d,
                                          hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d );
HYPRE_Int hypre_ILUSolveSchurGMRESJacIterDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                 hypre_ParVector *u, HYPRE_Int *perm,
                                                 HYPRE_Int nLU, hypre_ParCSRMatrix *S,
                                                 hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                                 HYPRE_Solver schur_solver,
                                                 HYPRE_Solver schur_precond,
                                                 hypre_ParVector *rhs, hypre_ParVector *x,
                                                 HYPRE_Int *u_end, hypre_CSRMatrix *matBLU_d,
                                                 hypre_CSRMatrix *matE_d, hypre_CSRMatrix *matF_d,
                                                 hypre_ParVector *ztemp,
                                                 hypre_Vector **Adiag_diag,
                                                 hypre_Vector **Sdiag_diag,
                                                 HYPRE_Int lower_jacobi_iters,
                                                 HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ParILUSchurGMRESMatvecJacIterDevice( void *matvec_data, HYPRE_Complex alpha,
                                                     void *ilu_vdata, void *x, HYPRE_Complex beta,
                                                     void *y );
HYPRE_Int hypre_ILUSolveRAPGMRESDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                        hypre_ParCSRMatrix *S, hypre_ParVector *ftemp,
                                        hypre_ParVector *utemp, hypre_ParVector *xtemp,
                                        hypre_ParVector *ytemp, HYPRE_Solver schur_solver,
                                        HYPRE_Solver schur_precond, hypre_ParVector *rhs,
                                        hypre_ParVector *x, HYPRE_Int *u_end,
                                        hypre_ParCSRMatrix *Aperm, hypre_CSRMatrix *matALU_d,
                                        hypre_CSRMatrix *matBLU_d, hypre_CSRMatrix *matE_d,
                                        hypre_CSRMatrix *matF_d, HYPRE_Int test_opt );
#endif

/* par_amgdd.c */
void *hypre_BoomerAMGDDCreate ( void );
HYPRE_Int hypre_BoomerAMGDDDestroy ( void *data );
HYPRE_Int hypre_BoomerAMGDDSetStartLevel ( void *data, HYPRE_Int start_level );
HYPRE_Int hypre_BoomerAMGDDGetStartLevel ( void *data, HYPRE_Int *start_level );
HYPRE_Int hypre_BoomerAMGDDSetFACNumCycles ( void *data, HYPRE_Int fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDGetFACNumCycles ( void *data, HYPRE_Int *fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDSetFACCycleType ( void *data, HYPRE_Int fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDGetFACCycleType ( void *data, HYPRE_Int *fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDSetFACNumRelax ( void *data, HYPRE_Int fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDGetFACNumRelax ( void *data, HYPRE_Int *fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxType ( void *data, HYPRE_Int fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxType ( void *data, HYPRE_Int *fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxWeight ( void *data, HYPRE_Real fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxWeight ( void *data, HYPRE_Real *fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDSetPadding ( void *data, HYPRE_Int padding );
HYPRE_Int hypre_BoomerAMGDDGetPadding ( void *data, HYPRE_Int *padding );
HYPRE_Int hypre_BoomerAMGDDSetNumGhostLayers ( void *data, HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDGetNumGhostLayers ( void *data, HYPRE_Int *num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDSetUserFACRelaxation( void *data,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int hypre_BoomerAMGDDGetAMG ( void *data, void **amg_solver );

/* par_amgdd_solve.c */
HYPRE_Int hypre_BoomerAMGDDSolve ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGDD_Cycle ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_BoomerAMGDD_ResidualCommunication ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Complex* hypre_BoomerAMGDD_PackResidualBuffer ( hypre_AMGDDCompGrid **compGrid,
                                                      hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );
HYPRE_Int hypre_BoomerAMGDD_UnpackResidualBuffer ( HYPRE_Complex *buffer,
                                                   hypre_AMGDDCompGrid **compGrid, hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level,
                                                   HYPRE_Int proc );

/* par_amgdd_setup.c */
HYPRE_Int hypre_BoomerAMGDDSetup ( void *amgdd_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );

/* par_amgdd_fac_cycle.c */
HYPRE_Int hypre_BoomerAMGDD_FAC ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_Cycle ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_type,
                                        HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_FCycle ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_Interpolate ( hypre_AMGDDCompGrid *compGrid_f,
                                              hypre_AMGDDCompGrid *compGrid_c );
HYPRE_Int hypre_BoomerAMGDD_FAC_Restrict ( hypre_AMGDDCompGrid *compGrid_f,
                                           hypre_AMGDDCompGrid *compGrid_c, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_Relax ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1JacobiHost ( void *amgdd_vdata, HYPRE_Int level,
                                                 HYPRE_Int relax_set );
HYPRE_Int hypre_BoomerAMGDD_FAC_Jacobi ( void *amgdd_vdata, HYPRE_Int level,
                                         HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_JacobiHost ( void *amgdd_vdata, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGDD_FAC_GaussSeidel ( void *amgdd_vdata, HYPRE_Int level,
                                              HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1Jacobi ( void *amgdd_vdata, HYPRE_Int level,
                                             HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_OrderedGaussSeidel ( void *amgdd_vdata, HYPRE_Int level,
                                                     HYPRE_Int cycle_param );

/* par_amgdd_fac_cycles_device.c */
HYPRE_Int hypre_BoomerAMGDD_FAC_JacobiDevice ( void *amgdd_vdata, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1JacobiDevice ( void *amgdd_vdata, HYPRE_Int level,
                                                   HYPRE_Int relax_set );

/* par_amgdd_comp_grid.c */
hypre_AMGDDCompGridMatrix* hypre_AMGDDCompGridMatrixCreate( void );
HYPRE_Int hypre_AMGDDCompGridMatrixDestroy ( hypre_AMGDDCompGridMatrix *matrix );
HYPRE_Int hypre_AMGDDCompGridMatvec ( HYPRE_Complex alpha, hypre_AMGDDCompGridMatrix *A,
                                      hypre_AMGDDCompGridVector *x, HYPRE_Complex beta, hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridRealMatvec ( HYPRE_Complex alpha, hypre_AMGDDCompGridMatrix *A,
                                          hypre_AMGDDCompGridVector *x, HYPRE_Complex beta, hypre_AMGDDCompGridVector *y );
hypre_AMGDDCompGridVector* hypre_AMGDDCompGridVectorCreate( void );
HYPRE_Int hypre_AMGDDCompGridVectorInitialize ( hypre_AMGDDCompGridVector *vector,
                                                HYPRE_Int num_owned, HYPRE_Int num_nonowned, HYPRE_Int num_real );
HYPRE_Int hypre_AMGDDCompGridVectorDestroy ( hypre_AMGDDCompGridVector *vector );
HYPRE_Real hypre_AMGDDCompGridVectorInnerProd ( hypre_AMGDDCompGridVector *x,
                                                hypre_AMGDDCompGridVector *y );
HYPRE_Real hypre_AMGDDCompGridVectorRealInnerProd ( hypre_AMGDDCompGridVector *x,
                                                    hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorScale ( HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorRealScale ( HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorAxpy ( HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealAxpy ( HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorSetConstantValues ( hypre_AMGDDCompGridVector *vector,
                                                       HYPRE_Complex value );
HYPRE_Int hypre_AMGDDCompGridVectorRealSetConstantValues ( hypre_AMGDDCompGridVector *vector,
                                                           HYPRE_Complex value );
HYPRE_Int hypre_AMGDDCompGridVectorCopy ( hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealCopy ( hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
hypre_AMGDDCompGrid *hypre_AMGDDCompGridCreate( void );
HYPRE_Int hypre_AMGDDCompGridDestroy ( hypre_AMGDDCompGrid *compGrid );
HYPRE_Int hypre_AMGDDCompGridInitialize ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int padding,
                                          HYPRE_Int level );
HYPRE_Int hypre_AMGDDCompGridSetupRelax ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridFinalize ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridSetupRealDofMarker ( hypre_AMGDDCompGrid **compGrid,
                                                  HYPRE_Int num_levels, HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_AMGDDCompGridResize ( hypre_AMGDDCompGrid *compGrid, HYPRE_Int new_size,
                                      HYPRE_Int need_coarse_info );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndices ( hypre_AMGDDCompGrid **compGrid,
                                                 HYPRE_Int *num_added_nodes, HYPRE_Int ****recv_map, HYPRE_Int num_recv_procs,
                                                 HYPRE_Int **A_tmp_info, HYPRE_Int start_level, HYPRE_Int num_levels );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndicesP ( hypre_ParAMGDDData *amgdd_data );
hypre_AMGDDCommPkg *hypre_AMGDDCommPkgCreate ( HYPRE_Int num_levels );
HYPRE_Int hypre_AMGDDCommPkgSendLevelDestroy ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
HYPRE_Int hypre_AMGDDCommPkgRecvLevelDestroy ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
HYPRE_Int hypre_AMGDDCommPkgDestroy ( hypre_AMGDDCommPkg *compGridCommPkg );
HYPRE_Int hypre_AMGDDCommPkgFinalize ( hypre_ParAMGData* amg_data,
                                       hypre_AMGDDCommPkg *compGridCommPkg, hypre_AMGDDCompGrid **compGrid );

/* par_amgdd_helpers.c */
HYPRE_Int hypre_BoomerAMGDD_SetupNearestProcessorNeighbors ( hypre_ParCSRMatrix *A,
                                                             hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding,
                                                             HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDD_RecursivelyBuildPsiComposite ( HYPRE_Int node, HYPRE_Int m,
                                                           hypre_AMGDDCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int use_sort );
HYPRE_Int hypre_BoomerAMGDD_MarkCoarse ( HYPRE_Int *list, HYPRE_Int *marker,
                                         HYPRE_Int *owned_coarse_indices, HYPRE_Int *nonowned_coarse_indices, HYPRE_Int *sort_map,
                                         HYPRE_Int num_owned, HYPRE_Int total_num_nodes, HYPRE_Int num_owned_coarse, HYPRE_Int list_size,
                                         HYPRE_Int dist, HYPRE_Int use_sort, HYPRE_Int *nodes_to_add );
HYPRE_Int hypre_BoomerAMGDD_UnpackRecvBuffer ( hypre_ParAMGDDData *amgdd_data,
                                               HYPRE_Int *recv_buffer, HYPRE_Int **A_tmp_info, HYPRE_Int *recv_map_send_buffer_size,
                                               HYPRE_Int *nodes_added_on_level, HYPRE_Int current_level, HYPRE_Int buffer_number );
HYPRE_Int* hypre_BoomerAMGDD_PackSendBuffer ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int proc,
                                              HYPRE_Int current_level, HYPRE_Int *padding, HYPRE_Int *send_flag_buffer_size );
HYPRE_Int hypre_BoomerAMGDD_PackRecvMapSendBuffer ( HYPRE_Int *recv_map_send_buffer,
                                                    HYPRE_Int **recv_red_marker, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size,
                                                    HYPRE_Int current_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_UnpackSendFlagBuffer ( hypre_AMGDDCompGrid **compGrid,
                                                   HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes,
                                                   HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo ( hypre_ParAMGDDData* amgdd_data );
HYPRE_Int hypre_BoomerAMGDD_FixUpRecvMaps ( hypre_AMGDDCompGrid **compGrid,
                                            hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int start_level, HYPRE_Int num_levels );

/* par_fsai.c */
void* hypre_FSAICreate( void );
HYPRE_Int hypre_FSAIDestroy ( void *data );
HYPRE_Int hypre_FSAISetAlgoType ( void *data, HYPRE_Int algo_type );
HYPRE_Int hypre_FSAISetLocalSolveType ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_FSAISetMaxSteps ( void *data, HYPRE_Int max_steps );
HYPRE_Int hypre_FSAISetMaxStepSize ( void *data, HYPRE_Int max_step_size );
HYPRE_Int hypre_FSAISetMaxNnzRow ( void *data, HYPRE_Int max_nnz_row );
HYPRE_Int hypre_FSAISetNumLevels ( void *data, HYPRE_Int num_levels );
HYPRE_Int hypre_FSAISetThreshold ( void *data, HYPRE_Real threshold );
HYPRE_Int hypre_FSAISetKapTolerance ( void *data, HYPRE_Real kap_tolerance );
HYPRE_Int hypre_FSAISetMaxIterations ( void *data, HYPRE_Int max_iterations );
HYPRE_Int hypre_FSAISetEigMaxIters ( void *data, HYPRE_Int eig_max_iters );
HYPRE_Int hypre_FSAISetZeroGuess ( void *data, HYPRE_Int zero_guess );
HYPRE_Int hypre_FSAISetTolerance ( void *data, HYPRE_Real tolerance );
HYPRE_Int hypre_FSAISetOmega ( void *data, HYPRE_Real omega );
HYPRE_Int hypre_FSAISetLogging ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_FSAISetNumIterations ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_FSAISetPrintLevel ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_FSAIGetAlgoType ( void *data, HYPRE_Int *algo_type );
HYPRE_Int hypre_FSAIGetLocalSolveType ( void *data, HYPRE_Int *local_solve_type );
HYPRE_Int hypre_FSAIGetMaxSteps ( void *data, HYPRE_Int *max_steps );
HYPRE_Int hypre_FSAIGetMaxStepSize ( void *data, HYPRE_Int *max_step_size );
HYPRE_Int hypre_FSAIGetMaxNnzRow ( void *data, HYPRE_Int *max_nnz_row );
HYPRE_Int hypre_FSAIGetNumLevels ( void *data, HYPRE_Int *num_levels );
HYPRE_Int hypre_FSAIGetThreshold ( void *data, HYPRE_Real *threshold );
HYPRE_Int hypre_FSAIGetKapTolerance ( void *data, HYPRE_Real *kap_tolerance );
HYPRE_Int hypre_FSAIGetMaxIterations ( void *data, HYPRE_Int *max_iterations );
HYPRE_Int hypre_FSAIGetEigMaxIters ( void *data, HYPRE_Int *eig_max_iters );
HYPRE_Int hypre_FSAIGetZeroGuess ( void *data, HYPRE_Int *zero_guess );
HYPRE_Int hypre_FSAIGetTolerance ( void *data, HYPRE_Real *tolerance );
HYPRE_Int hypre_FSAIGetOmega ( void *data, HYPRE_Real *omega );
HYPRE_Int hypre_FSAIGetLogging ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_FSAIGetNumIterations ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FSAIGetPrintLevel ( void *data, HYPRE_Int *print_level );

/* par_fsai_setup.c */
HYPRE_Int hypre_CSRMatrixExtractDenseMat ( hypre_CSRMatrix *A, hypre_Vector *A_sub,
                                           HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                                           HYPRE_Int *marker );
HYPRE_Int hypre_CSRMatrixExtractDenseRow ( hypre_CSRMatrix *A, hypre_Vector *A_subrow,
                                           HYPRE_Int *marker, HYPRE_Int row_num );
HYPRE_Int hypre_FindKapGrad ( hypre_CSRMatrix *A_diag, hypre_Vector *kaporin_gradient,
                              HYPRE_Int *kap_grad_nonzeros, hypre_Vector *G_temp,
                              HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                              HYPRE_Int max_row_size, HYPRE_Int row_num, HYPRE_Int *kg_marker );
HYPRE_Int hypre_AddToPattern ( hypre_Vector *kaporin_gradient, HYPRE_Int *kap_grad_nonzeros,
                               HYPRE_Int *S_Pattern, HYPRE_Int *S_nnz, HYPRE_Int *kg_marker,
                               HYPRE_Int max_step_size );
HYPRE_Int hypre_FSAISetup ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupNative ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupOMPDyn ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_FSAIPrintStats ( void *fsai_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIComputeOmega ( void *fsai_vdata, hypre_ParCSRMatrix *A );
void hypre_swap2_ci ( HYPRE_Complex *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_qsort2_ci ( HYPRE_Complex *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
HYPRE_Int hypre_FSAIDumpLocalLSDense ( void *fsai_vdata, const char *filename,
                                       hypre_ParCSRMatrix *A );

/* par_fsai_solve.c */
HYPRE_Int hypre_FSAISolve ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                            hypre_ParVector *x );
HYPRE_Int hypre_FSAIApply ( void *fsai_vdata, HYPRE_Complex alpha, hypre_ParVector *b,
                            hypre_ParVector *x );

/* par_fsai_device.c */
HYPRE_Int hypre_FSAISetupDevice( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                 hypre_ParVector *f, hypre_ParVector *u );
