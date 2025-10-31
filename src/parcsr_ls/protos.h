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
HYPRE_Int hypre_AMGHybridSetPrecond ( void *pcg_vdata,
                                      HYPRE_Int (*solve)(void*, void*, void*, void*),
                                      HYPRE_Int (*setup)(void*, void*, void*, void*),
                                      void *pcg_precond );
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

HYPRE_Int hypre_AMGHybridSetCycleStruct ( void *AMGhybrid_vdata, HYPRE_Int *cycle_struct,
                                             HYPRE_Int cycle_num_nodes );
HYPRE_Int hypre_AMGHybridSetRelaxNodeTypes ( void *AMGhybrid_vdata, HYPRE_Int *relax_node_types );
HYPRE_Int hypre_AMGHybridSetRelaxNodeOrder ( void *AMGhybrid_vdata, HYPRE_Int *relax_node_order );
HYPRE_Int hypre_AMGHybridSetRelaxNodeOuterWeights ( void *AMGhybrid_vdata,
                                                     HYPRE_Real *relax_node_outerweights );
HYPRE_Int hypre_AMGHybridSetRelaxNodeWeights ( void *AMGhybrid_vdata,
                                                 HYPRE_Real *relax_node_weights );
HYPRE_Int hypre_AMGHybridSetRelaxEdgeWeights ( void *AMGhybrid_vdata,
                                                 HYPRE_Real *relax_edge_weights );
 
HYPRE_Int hypre_AMGHybridSetNodeNumSweeps ( void *AMGhybrid_vdata,
                                                 HYPRE_Int *node_num_sweeps );
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

/* HYPRE_parcsr_int.c */
HYPRE_Int hypre_ParSetRandomValues ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_ParPrintVector ( void *v, const char *file );
void *hypre_ParReadVector ( MPI_Comm comm, const char *file );
HYPRE_Int hypre_ParVectorSize ( void *x );
HYPRE_Int aux_maskCount ( HYPRE_Int n, HYPRE_Int *mask );
void aux_indexFromMask ( HYPRE_Int n, HYPRE_Int *mask, HYPRE_Int *index );

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
HYPRE_Int hypre_BoomerAMGSetKappaCycleVal( void *data, HYPRE_Int kappa );
HYPRE_Int hypre_BoomerAMGGetKappaCycleVal( void *data, HYPRE_Int *kappa );
HYPRE_Int hypre_BoomerAMGSetCycleStruct ( void *data, HYPRE_Int *cycle_struct, HYPRE_Int cycle_num_nodes);
HYPRE_Int hypre_BoomerAMGGetCycleStruct ( void *data, HYPRE_Int **cycle_struct, HYPRE_Int *cycle_num_nodes);
HYPRE_Int hypre_BoomerAMGSetConvergeType ( void *data, HYPRE_Int type );
HYPRE_Int hypre_BoomerAMGGetConvergeType ( void *data, HYPRE_Int *type );
HYPRE_Int hypre_BoomerAMGSetTol ( void *data, HYPRE_Real tol );
HYPRE_Int hypre_BoomerAMGGetTol ( void *data, HYPRE_Real *tol );
HYPRE_Int hypre_BoomerAMGSetNodeNumSweeps ( void *data, HYPRE_Int *node_num_sweeps );
HYPRE_Int hypre_BoomerAMGGetNodeNumSweeps (void *data, HYPRE_Int **node_num_sweeps );
HYPRE_Int hypre_BoomerAMGSetNumSweeps ( void *data, HYPRE_Int num_sweeps );
HYPRE_Int hypre_BoomerAMGSetCycleNumSweeps ( void *data, HYPRE_Int num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleNumSweeps ( void *data, HYPRE_Int *num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetNumGridSweeps ( void *data, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGGetNumGridSweeps ( void *data, HYPRE_Int **num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGSetRelaxType ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BoomerAMGSetRelaxNodeTypes ( void *data, HYPRE_Int *relax_node_types );
HYPRE_Int hypre_BoomerAMGSetRelaxNodeOrder ( void *data, HYPRE_Int *relax_node_order );
HYPRE_Int hypre_BoomerAMGGetRelaxNodeTypes( void *data, HYPRE_Int **relax_node_types );
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
HYPRE_Int hypre_BoomerAMGSetRelaxNodeOuterWeights ( void *data, HYPRE_Real *relax_node_outerweights );
HYPRE_Int hypre_BoomerAMGGetRelaxNodeOuterWeights ( void *data, HYPRE_Real **relax_node_outerweights );
HYPRE_Int hypre_BoomerAMGSetRelaxNodeWeights ( void *data, HYPRE_Real *relax_node_weights );
HYPRE_Int hypre_BoomerAMGGetRelaxNodeWeights ( void *data, HYPRE_Real **relax_node_weights );
HYPRE_Int hypre_BoomerAMGSetRelaxEdgeWeights ( void *data, HYPRE_Real *relax_edge_weights );
HYPRE_Int hypre_BoomerAMGGetRelaxEdgeWeights ( void *data, HYPRE_Real **relax_edge_weights );
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
HYPRE_Int hypre_BoomerAMGSetFilterFunctions ( void *data, HYPRE_Int filter_functions );
HYPRE_Int hypre_BoomerAMGGetFilterFunctions ( void *data, HYPRE_Int *filter_functions );
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
hypre_ParChebyData * hypre_ParChebyCreate( void );
HYPRE_Int hypre_ParChebyDestroy( hypre_ParChebyData *cheby_data );
HYPRE_Int hypre_ParChebySetMaxIterations( hypre_ParChebyData *cheby_data,
                                          HYPRE_Int max_iterations );
HYPRE_Int hypre_ParChebyGetMaxIterations( hypre_ParChebyData *cheby_data,
                                          HYPRE_Int *max_iterations );
HYPRE_Int hypre_ParChebySetZeroGuess( hypre_ParChebyData *cheby_data, HYPRE_Int zero_guess );
HYPRE_Int hypre_ParChebyGetZeroGuess( hypre_ParChebyData *cheby_data, HYPRE_Int *zero_guess );
HYPRE_Int hypre_ParChebySetTolerance( hypre_ParChebyData *cheby_data, HYPRE_Real tol );
HYPRE_Int hypre_ParChebyGetTolerance( hypre_ParChebyData *cheby_data, HYPRE_Real *tol );
HYPRE_Int hypre_ParChebySetPrintLevel( hypre_ParChebyData *cheby_data, HYPRE_Int print_level );
HYPRE_Int hypre_ParChebyGetPrintLevel( hypre_ParChebyData *cheby_data, HYPRE_Int *print_level );
HYPRE_Int hypre_ParChebySetLogging( hypre_ParChebyData *cheby_data, HYPRE_Int logging );
HYPRE_Int hypre_ParChebyGetLogging( hypre_ParChebyData *cheby_data, HYPRE_Int *logging );
HYPRE_Int hypre_ParChebySetOrder( hypre_ParChebyData *cheby_data, HYPRE_Int order );
HYPRE_Int hypre_ParChebyGetOrder( hypre_ParChebyData *cheby_data, HYPRE_Int *order );
HYPRE_Int hypre_ParChebySetVariant( hypre_ParChebyData *cheby_data, HYPRE_Int variant );
HYPRE_Int hypre_ParChebyGetVariant( hypre_ParChebyData *cheby_data, HYPRE_Int *variant );
HYPRE_Int hypre_ParChebySetScale( hypre_ParChebyData *cheby_data, HYPRE_Int scale );
HYPRE_Int hypre_ParChebyGetScale( hypre_ParChebyData *cheby_data, HYPRE_Int *scale );
HYPRE_Int hypre_ParChebySetEigRatio( hypre_ParChebyData *cheby_data, HYPRE_Real eig_ratio );
HYPRE_Int hypre_ParChebyGetEigRatio( hypre_ParChebyData *cheby_data, HYPRE_Real *eig_ratio );
HYPRE_Int hypre_ParChebySetEigEst( hypre_ParChebyData *cheby_data, HYPRE_Int eig_est );
HYPRE_Int hypre_ParChebyGetEigEst( hypre_ParChebyData *cheby_data, HYPRE_Int *eig_est );
HYPRE_Int hypre_ParChebySetMinMaxEigEst( hypre_ParChebyData *cheby_data,
                                         HYPRE_Real eig_min_est, HYPRE_Real eig_max_est );
HYPRE_Int hypre_ParChebyGetMinMaxEigEst( hypre_ParChebyData *cheby_data,
                                         HYPRE_Real *eig_min_est, HYPRE_Real *eig_max_est );
HYPRE_Int hypre_ParChebySetTempVectors( hypre_ParChebyData *cheby_data,
                                        hypre_ParVector *Ptemp, hypre_ParVector *Rtemp,
                                        hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

/* par_cheby_setup.c */
HYPRE_Int hypre_ParChebySetup( hypre_ParChebyData *cheby_data, hypre_ParCSRMatrix *A,
                               hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParCSRRelax_Cheby_Setup( hypre_ParCSRMatrix *A, HYPRE_Real max_eig,
                                         HYPRE_Real min_eig, HYPRE_Real fraction,
                                         HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                         HYPRE_Real **coefs_ptr, HYPRE_Real **ds_ptr );

/* par_cheby_solve.c */
HYPRE_Int hypre_ParChebySolve( hypre_ParChebyData *cheby_data, hypre_ParCSRMatrix *A,
                               hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParCSRRelax_Cheby_Solve( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                         HYPRE_Real *ds_data, HYPRE_Real *coefs,
                                         HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                         hypre_ParVector *u, hypre_ParVector *v,
                                         hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                         hypre_ParVector *tmp_vec );

/* par_cheby_device.c */
HYPRE_Int hypre_ParCSRRelax_Cheby_SolveDevice( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                               HYPRE_Real *ds_data, HYPRE_Real *coefs,
                                               HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                               hypre_ParVector *u, hypre_ParVector *v,
                                               hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                               hypre_ParVector *tmp_vec );

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
HYPRE_Int hypre_map3 ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_BigInt iz, HYPRE_Int p, HYPRE_Int q,
                       HYPRE_Int r, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part,
                       HYPRE_BigInt *nz_part );

/* par_laplace_9pt.c */
HYPRE_BigInt hypre_map2 ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_Int p, HYPRE_Int q,
                          HYPRE_BigInt nx, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part );

/* par_laplace.c */
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

/* par_relax_more_device.c */
HYPRE_Int hypre_ParCSRMaxEigEstimateDevice ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                             HYPRE_Real *max_eig, HYPRE_Real *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCGDevice ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                               HYPRE_Int max_iter, HYPRE_Real *max_eig, HYPRE_Real *min_eig );

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
HYPRE_Int hypre_ParKrylovInnerProdTagged( void *x, void *y, HYPRE_Int *num_tags_ptr,
                                          HYPRE_Complex **iprod_ptr );
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
HYPRE_Int hypre_MGRSetFSolverAtLevel( void *mgr_vdata, void *fsolver, HYPRE_Int level );
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
HYPRE_Int hypre_block_jacobi_solve( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int method, HYPRE_Real *diaginv,
                                    hypre_ParVector *Vtemp );
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
HYPRE_Int hypre_MGRSetNonGalerkinMaxElmts( void *mgr_vdata, HYPRE_Int max_elmts );
HYPRE_Int hypre_MGRSetLevelNonGalerkinMaxElmts( void *mgr_vdata, HYPRE_Int *max_elmts );
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
HYPRE_Int hypre_MGRSetGlobalSmootherAtLevel( void *mgr_vdata, HYPRE_Solver smoother,
                                             HYPRE_Int level );
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
                                hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *S,
                                hypre_IntArray *CF_marker, HYPRE_BigInt *num_cpts_global,
                                HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
                                HYPRE_Int block_jacobi_bsize, HYPRE_Int method,
                                HYPRE_Int num_sweeps_post, hypre_ParCSRMatrix **Wp_ptr,
                                hypre_ParCSRMatrix **P_ptr );
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
HYPRE_Int hypre_MGRBuildBlockJacobiP( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                      hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *Wp,
                                      HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildP( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                           HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                           HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPHost( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                               hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                               HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                               hypre_ParCSRMatrix **Wp_ptr, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildInterpApproximateInverse( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                  HYPRE_BigInt *num_cpts_global,
                                                  hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRTruncateAcfCPR( hypre_ParCSRMatrix *A_CF, hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRBuildRFromWr( hypre_IntArray *C_map, hypre_IntArray *F_map,
                                 HYPRE_BigInt global_num_rows_R, HYPRE_BigInt global_num_cols_R,
                                 HYPRE_BigInt *row_starts_R, HYPRE_BigInt *col_starts_R,
                                 hypre_ParCSRMatrix *Wr, hypre_ParCSRMatrix **R_ptr );
HYPRE_Int hypre_MGRBlockColLumpedRestrict( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                           hypre_ParCSRMatrix *A_CF, hypre_IntArray *CF_marker,
                                           HYPRE_Int blk_dim, hypre_ParCSRMatrix **Wr_ptr,
                                           hypre_ParCSRMatrix **R_ptr);
HYPRE_Int hypre_MGRColLumpedRestrict(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                     hypre_ParCSRMatrix *A_CF, hypre_IntArray *CF_marker,
                                     hypre_ParCSRMatrix **Wr_ptr, hypre_ParCSRMatrix **R_ptr);

/* par_mgr_rap.c */
HYPRE_Int hypre_MGRBuildCoarseOperator(void *mgr_data, hypre_ParCSRMatrix *A_FF,
                                       hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *A_CF,
                                       hypre_ParCSRMatrix **A_CC_ptr, hypre_ParCSRMatrix *Wp,
                                       hypre_ParCSRMatrix *Wr, HYPRE_Int level);

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
HYPRE_Int hypre_MGRBuildRFromWrDevice(hypre_IntArray *C_map, hypre_IntArray *F_map,
                                      hypre_ParCSRMatrix *Wr, hypre_ParCSRMatrix *R);

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
