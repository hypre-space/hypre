/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include <HYPRE_config.h>

#include "HYPRE_parcsr_ls.h"

#ifndef hypre_PARCSR_LS_HEADER
#define hypre_PARCSR_LS_HEADER

#include "_hypre_utilities.h"
#include "krylov.h"
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_lobpcg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int prev; int next; } Link;


/* ame.c */
void *hypre_AMECreate ( void );
int hypre_AMEDestroy ( void *esolver );
int hypre_AMESetAMSSolver ( void *esolver , void *ams_solver );
int hypre_AMESetMassMatrix ( void *esolver , hypre_ParCSRMatrix *M );
int hypre_AMESetBlockSize ( void *esolver , int block_size );
int hypre_AMESetMaxIter ( void *esolver , int maxit );
int hypre_AMESetTol ( void *esolver , double tol );
int hypre_AMESetPrintLevel ( void *esolver , int print_level );
int hypre_AMESetup ( void *esolver );
int hypre_AMEDiscrDivFreeComponent ( void *esolver , hypre_ParVector *b );
void hypre_AMEOperatorA ( void *data , void *x , void *y );
void hypre_AMEMultiOperatorA ( void *data , void *x , void *y );
void hypre_AMEOperatorM ( void *data , void *x , void *y );
void hypre_AMEMultiOperatorM ( void *data , void *x , void *y );
void hypre_AMEOperatorB ( void *data , void *x , void *y );
void hypre_AMEMultiOperatorB ( void *data , void *x , void *y );
int hypre_AMESolve ( void *esolver );
int hypre_AMEGetEigenvectors ( void *esolver , HYPRE_ParVector **eigenvectors_ptr );
int hypre_AMEGetEigenvalues ( void *esolver , double **eigenvalues_ptr );

/* amg_hybrid.c */
void *hypre_AMGHybridCreate ( void );
int hypre_AMGHybridDestroy ( void *AMGhybrid_vdata );
int hypre_AMGHybridSetTol ( void *AMGhybrid_vdata , double tol );
int hypre_AMGHybridSetAbsoluteTol ( void *AMGhybrid_vdata , double a_tol );
int hypre_AMGHybridSetConvergenceTol ( void *AMGhybrid_vdata , double cf_tol );
int hypre_AMGHybridSetDSCGMaxIter ( void *AMGhybrid_vdata , int dscg_max_its );
int hypre_AMGHybridSetPCGMaxIter ( void *AMGhybrid_vdata , int pcg_max_its );
int hypre_AMGHybridSetSetupType ( void *AMGhybrid_vdata , int setup_type );
int hypre_AMGHybridSetSolverType ( void *AMGhybrid_vdata , int solver_type );
int hypre_AMGHybridSetKDim ( void *AMGhybrid_vdata , int k_dim );
int hypre_AMGHybridSetStopCrit ( void *AMGhybrid_vdata , int stop_crit );
int hypre_AMGHybridSetTwoNorm ( void *AMGhybrid_vdata , int two_norm );
int hypre_AMGHybridSetRelChange ( void *AMGhybrid_vdata , int rel_change );
int hypre_AMGHybridSetPrecond ( void *pcg_vdata , int (*pcg_precond_solve )(), int (*pcg_precond_setup )(), void *pcg_precond );
int hypre_AMGHybridSetLogging ( void *AMGhybrid_vdata , int logging );
int hypre_AMGHybridSetPrintLevel ( void *AMGhybrid_vdata , int print_level );
int hypre_AMGHybridSetStrongThreshold ( void *AMGhybrid_vdata , double strong_threshold );
int hypre_AMGHybridSetMaxRowSum ( void *AMGhybrid_vdata , double max_row_sum );
int hypre_AMGHybridSetTruncFactor ( void *AMGhybrid_vdata , double trunc_factor );
int hypre_AMGHybridSetPMaxElmts ( void *AMGhybrid_vdata , int P_max_elmts );
int hypre_AMGHybridSetMaxLevels ( void *AMGhybrid_vdata , int max_levels );
int hypre_AMGHybridSetMeasureType ( void *AMGhybrid_vdata , int measure_type );
int hypre_AMGHybridSetCoarsenType ( void *AMGhybrid_vdata , int coarsen_type );
int hypre_AMGHybridSetInterpType ( void *AMGhybrid_vdata , int interp_type );
int hypre_AMGHybridSetCycleType ( void *AMGhybrid_vdata , int cycle_type );
int hypre_AMGHybridSetNumSweeps ( void *AMGhybrid_vdata , int num_sweeps );
int hypre_AMGHybridSetCycleNumSweeps ( void *AMGhybrid_vdata , int num_sweeps , int k );
int hypre_AMGHybridSetRelaxType ( void *AMGhybrid_vdata , int relax_type );
int hypre_AMGHybridSetCycleRelaxType ( void *AMGhybrid_vdata , int relax_type , int k );
int hypre_AMGHybridSetRelaxOrder ( void *AMGhybrid_vdata , int relax_order );
int hypre_AMGHybridSetNumGridSweeps ( void *AMGhybrid_vdata , int *num_grid_sweeps );
int hypre_AMGHybridSetGridRelaxType ( void *AMGhybrid_vdata , int *grid_relax_type );
int hypre_AMGHybridSetGridRelaxPoints ( void *AMGhybrid_vdata , int **grid_relax_points );
int hypre_AMGHybridSetRelaxWeight ( void *AMGhybrid_vdata , double *relax_weight );
int hypre_AMGHybridSetOmega ( void *AMGhybrid_vdata , double *omega );
int hypre_AMGHybridSetRelaxWt ( void *AMGhybrid_vdata , double relax_wt );
int hypre_AMGHybridSetLevelRelaxWt ( void *AMGhybrid_vdata , double relax_wt , int level );
int hypre_AMGHybridSetOuterWt ( void *AMGhybrid_vdata , double outer_wt );
int hypre_AMGHybridSetLevelOuterWt ( void *AMGhybrid_vdata , double outer_wt , int level );
int hypre_AMGHybridSetNumPaths ( void *AMGhybrid_vdata , int num_paths );
int hypre_AMGHybridSetDofFunc ( void *AMGhybrid_vdata , int *dof_func );
int hypre_AMGHybridSetAggNumLevels ( void *AMGhybrid_vdata , int agg_num_levels );
int hypre_AMGHybridSetNumFunctions ( void *AMGhybrid_vdata , int num_functions );
int hypre_AMGHybridSetNodal ( void *AMGhybrid_vdata , int nodal );
int hypre_AMGHybridGetNumIterations ( void *AMGhybrid_vdata , int *num_its );
int hypre_AMGHybridGetDSCGNumIterations ( void *AMGhybrid_vdata , int *dscg_num_its );
int hypre_AMGHybridGetPCGNumIterations ( void *AMGhybrid_vdata , int *pcg_num_its );
int hypre_AMGHybridGetFinalRelativeResidualNorm ( void *AMGhybrid_vdata , double *final_rel_res_norm );
int hypre_AMGHybridSetup ( void *AMGhybrid_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
int hypre_AMGHybridSolve ( void *AMGhybrid_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );

/* ams.c */
int hypre_ParCSRRelax ( hypre_ParCSRMatrix *A , hypre_ParVector *f , int relax_type , int relax_times , double *l1_norms , double relax_weight , double omega , hypre_ParVector *u , hypre_ParVector *v );
hypre_ParVector *hypre_ParVectorInRangeOf ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInDomainOf ( hypre_ParCSRMatrix *A );
int hypre_ParVectorBlockSplit ( hypre_ParVector *x , hypre_ParVector *x_ [3 ], int dim );
int hypre_ParVectorBlockGather ( hypre_ParVector *x , hypre_ParVector *x_ [3 ], int dim );
int hypre_BoomerAMGBlockSolve ( void *B , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
int hypre_ParCSRMatrixFixZeroRows ( hypre_ParCSRMatrix *A );
int hypre_ParCSRComputeL1Norms ( hypre_ParCSRMatrix *A , int option , double **l1_norm_ptr );
int hypre_ParCSRMatrixSetDiagRows ( hypre_ParCSRMatrix *A , double d );
void *hypre_AMSCreate ( void );
int hypre_AMSDestroy ( void *solver );
int hypre_AMSSetDimension ( void *solver , int dim );
int hypre_AMSSetDiscreteGradient ( void *solver , hypre_ParCSRMatrix *G );
int hypre_AMSSetCoordinateVectors ( void *solver , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z );
int hypre_AMSSetEdgeConstantVectors ( void *solver , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz );
int hypre_AMSSetAlphaPoissonMatrix ( void *solver , hypre_ParCSRMatrix *A_Pi );
int hypre_AMSSetBetaPoissonMatrix ( void *solver , hypre_ParCSRMatrix *A_G );
int hypre_AMSSetMaxIter ( void *solver , int maxit );
int hypre_AMSSetTol ( void *solver , double tol );
int hypre_AMSSetCycleType ( void *solver , int cycle_type );
int hypre_AMSSetPrintLevel ( void *solver , int print_level );
int hypre_AMSSetSmoothingOptions ( void *solver , int A_relax_type , int A_relax_times , double A_relax_weight , double A_omega );
int hypre_AMSSetAlphaAMGOptions ( void *solver , int B_Pi_coarsen_type , int B_Pi_agg_levels , int B_Pi_relax_type , double B_Pi_theta , int B_Pi_interp_type , int B_Pi_Pmax );
int hypre_AMSSetBetaAMGOptions ( void *solver , int B_G_coarsen_type , int B_G_agg_levels , int B_G_relax_type , double B_G_theta , int B_G_interp_type , int B_G_Pmax );
int hypre_AMSComputePi ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *G , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz , int dim , hypre_ParCSRMatrix **Pi_ptr );
int hypre_AMSComputePixyz ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *G , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz , int dim , hypre_ParCSRMatrix **Pix_ptr , hypre_ParCSRMatrix **Piy_ptr , hypre_ParCSRMatrix **Piz_ptr );
int hypre_AMSComputeGPi ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *G , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *z , hypre_ParVector *Gx , hypre_ParVector *Gy , hypre_ParVector *Gz , int dim , hypre_ParCSRMatrix **GPi_ptr );
int hypre_AMSSetup ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
int hypre_AMSSolve ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
int hypre_ParCSRSubspacePrec ( hypre_ParCSRMatrix *A0 , int A0_relax_type , int A0_relax_times , double *A0_l1_norms , double A0_relax_weight , double A0_omega , hypre_ParCSRMatrix **A , HYPRE_Solver *B , hypre_ParCSRMatrix **P , hypre_ParVector **r , hypre_ParVector **g , hypre_ParVector *x , hypre_ParVector *y , hypre_ParVector *r0 , hypre_ParVector *g0 , char *cycle );
int hypre_AMSGetNumIterations ( void *solver , int *num_iterations );
int hypre_AMSGetFinalRelativeResidualNorm ( void *solver , double *rel_resid_norm );
int hypre_AMSConstructDiscreteGradient ( hypre_ParCSRMatrix *A , hypre_ParVector *x_coord , int *edge_vertex , hypre_ParCSRMatrix **G_ptr );
int hypre_AMSFEISetup ( void *solver , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x , int num_vert , int num_local_vert , int *vert_number , double *vert_coord , int num_edges , int *edge_vertex );
int hypre_AMSFEIDestroy ( void *solver );

/* aux_interp.c */
void insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg , int *IN_marker , int *node_add , int num_cols_A_offd , int full_off_procNodes , int num_procs , int *OUT_marker );
int alt_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg , hypre_ParCSRCommPkg *extend_comm_pkg , int *IN_marker , int full_off_procNodes , int *OUT_marker );
int hypre_ParCSRFindExtendCommPkg ( hypre_ParCSRMatrix *A , int newoff , int *found , hypre_ParCSRCommPkg **extend_comm_pkg );
void hypre_ParCSRCommExtendA ( hypre_ParCSRMatrix *A , int newoff , int *found , int *p_num_recvs , int **p_recv_procs , int **p_recv_vec_starts , int *p_num_sends , int **p_send_procs , int **p_send_map_starts , int **p_send_map_elmts , int **p_node_add );
int hypre_ssort ( int *data , int n );
int index_of_minimum ( int *data , int n );
void swap_int ( int *data , int a , int b );
void initialize_vecs ( int diag_n , int offd_n , int *diag_ftc , int *offd_ftc , int *diag_pm , int *offd_pm , int *tmp_CF );
int new_offd_nodes ( int **found , int num_cols_A_offd , int *A_ext_i , int *A_ext_j , int num_cols_S_offd , int *col_map_offd , int col_1 , int col_n , int *Sop_i , int *Sop_j , int *CF_marker , hypre_ParCSRCommPkg *comm_pkg );

/* block_tridiag.c */
void *hypre_BlockTridiagCreate ( void );
int hypre_BlockTridiagDestroy ( void *data );
int hypre_BlockTridiagSetup ( void *data , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
int hypre_BlockTridiagSolve ( void *data , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
int hypre_BlockTridiagSetIndexSet ( void *data , int n , int *inds );
int hypre_BlockTridiagSetAMGStrengthThreshold ( void *data , double thresh );
int hypre_BlockTridiagSetAMGNumSweeps ( void *data , int nsweeps );
int hypre_BlockTridiagSetAMGRelaxType ( void *data , int relax_type );
int hypre_BlockTridiagSetPrintLevel ( void *data , int print_level );

/* driver.c */
int BuildParFromFile ( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian ( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParDifConv ( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParFromOneFile ( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildRhsParFromOneFile ( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix A , HYPRE_ParVector *b_ptr );
int BuildParLaplacian9pt ( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian27pt ( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );

/* HYPRE_ame.c */
int HYPRE_AMECreate ( HYPRE_Solver *esolver );
int HYPRE_AMEDestroy ( HYPRE_Solver esolver );
int HYPRE_AMESetup ( HYPRE_Solver esolver );
int HYPRE_AMESolve ( HYPRE_Solver esolver );
int HYPRE_AMESetAMSSolver ( HYPRE_Solver esolver , HYPRE_Solver ams_solver );
int HYPRE_AMESetMassMatrix ( HYPRE_Solver esolver , HYPRE_ParCSRMatrix M );
int HYPRE_AMESetBlockSize ( HYPRE_Solver esolver , int block_size );
int HYPRE_AMESetMaxIter ( HYPRE_Solver esolver , int maxit );
int HYPRE_AMESetTol ( HYPRE_Solver esolver , double tol );
int HYPRE_AMESetPrintLevel ( HYPRE_Solver esolver , int print_level );
int HYPRE_AMEGetEigenvalues ( HYPRE_Solver esolver , double **eigenvalues );
int HYPRE_AMEGetEigenvectors ( HYPRE_Solver esolver , HYPRE_ParVector **eigenvectors );

/* HYPRE_ams.c */
int HYPRE_AMSCreate ( HYPRE_Solver *solver );
int HYPRE_AMSDestroy ( HYPRE_Solver solver );
int HYPRE_AMSSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_AMSSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_AMSSetDimension ( HYPRE_Solver solver , int dim );
int HYPRE_AMSSetDiscreteGradient ( HYPRE_Solver solver , HYPRE_ParCSRMatrix G );
int HYPRE_AMSSetCoordinateVectors ( HYPRE_Solver solver , HYPRE_ParVector x , HYPRE_ParVector y , HYPRE_ParVector z );
int HYPRE_AMSSetEdgeConstantVectors ( HYPRE_Solver solver , HYPRE_ParVector Gx , HYPRE_ParVector Gy , HYPRE_ParVector Gz );
int HYPRE_AMSSetAlphaPoissonMatrix ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A_alpha );
int HYPRE_AMSSetBetaPoissonMatrix ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A_beta );
int HYPRE_AMSSetMaxIter ( HYPRE_Solver solver , int maxit );
int HYPRE_AMSSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_AMSSetCycleType ( HYPRE_Solver solver , int cycle_type );
int HYPRE_AMSSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_AMSSetSmoothingOptions ( HYPRE_Solver solver , int relax_type , int relax_times , double relax_weight , double omega );
int HYPRE_AMSSetAlphaAMGOptions ( HYPRE_Solver solver , int alpha_coarsen_type , int alpha_agg_levels , int alpha_relax_type , double alpha_strength_threshold , int alpha_interp_type , int alpha_Pmax );
int HYPRE_AMSSetBetaAMGOptions ( HYPRE_Solver solver , int beta_coarsen_type , int beta_agg_levels , int beta_relax_type , double beta_strength_threshold , int beta_interp_type , int beta_Pmax );
int HYPRE_AMSGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_AMSGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *rel_resid_norm );
int HYPRE_AMSConstructDiscreteGradient ( HYPRE_ParCSRMatrix A , HYPRE_ParVector x_coord , int *edge_vertex , HYPRE_ParCSRMatrix *G );
int HYPRE_AMSFEISetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x , int *EdgeNodeList_ , int *NodeNumbers_ , int numEdges_ , int numLocalNodes_ , int numNodes_ , double *NodalCoord_ );
int HYPRE_AMSFEIDestroy ( HYPRE_Solver solver );

/* HYPRE_parcsr_amg.c */
int HYPRE_BoomerAMGCreate ( HYPRE_Solver *solver );
int HYPRE_BoomerAMGDestroy ( HYPRE_Solver solver );
int HYPRE_BoomerAMGSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSolveT ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSetRestriction ( HYPRE_Solver solver , int restr_par );
int HYPRE_BoomerAMGSetMaxLevels ( HYPRE_Solver solver , int max_levels );
int HYPRE_BoomerAMGGetMaxLevels ( HYPRE_Solver solver , int *max_levels );
int HYPRE_BoomerAMGSetStrongThreshold ( HYPRE_Solver solver , double strong_threshold );
int HYPRE_BoomerAMGGetStrongThreshold ( HYPRE_Solver solver , double *strong_threshold );
int HYPRE_BoomerAMGSetMaxRowSum ( HYPRE_Solver solver , double max_row_sum );
int HYPRE_BoomerAMGGetMaxRowSum ( HYPRE_Solver solver , double *max_row_sum );
int HYPRE_BoomerAMGSetTruncFactor ( HYPRE_Solver solver , double trunc_factor );
int HYPRE_BoomerAMGGetTruncFactor ( HYPRE_Solver solver , double *trunc_factor );
int HYPRE_BoomerAMGSetPMaxElmts ( HYPRE_Solver solver , int P_max_elmts );
int HYPRE_BoomerAMGGetPMaxElmts ( HYPRE_Solver solver , int *P_max_elmts );
int HYPRE_BoomerAMGSetJacobiTruncThreshold ( HYPRE_Solver solver , double jacobi_trunc_threshold );
int HYPRE_BoomerAMGGetJacobiTruncThreshold ( HYPRE_Solver solver , double *jacobi_trunc_threshold );
int HYPRE_BoomerAMGSetPostInterpType ( HYPRE_Solver solver , int post_interp_type );
int HYPRE_BoomerAMGGetPostInterpType ( HYPRE_Solver solver , int *post_interp_type );
int HYPRE_BoomerAMGSetSCommPkgSwitch ( HYPRE_Solver solver , double S_commpkg_switch );
int HYPRE_BoomerAMGSetInterpType ( HYPRE_Solver solver , int interp_type );
int HYPRE_BoomerAMGSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_BoomerAMGSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_BoomerAMGGetMaxIter ( HYPRE_Solver solver , int *max_iter );
int HYPRE_BoomerAMGSetCoarsenType ( HYPRE_Solver solver , int coarsen_type );
int HYPRE_BoomerAMGGetCoarsenType ( HYPRE_Solver solver , int *coarsen_type );
int HYPRE_BoomerAMGSetMeasureType ( HYPRE_Solver solver , int measure_type );
int HYPRE_BoomerAMGGetMeasureType ( HYPRE_Solver solver , int *measure_type );
int HYPRE_BoomerAMGSetSetupType ( HYPRE_Solver solver , int setup_type );
int HYPRE_BoomerAMGSetCycleType ( HYPRE_Solver solver , int cycle_type );
int HYPRE_BoomerAMGGetCycleType ( HYPRE_Solver solver , int *cycle_type );
int HYPRE_BoomerAMGSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_BoomerAMGGetTol ( HYPRE_Solver solver , double *tol );
int HYPRE_BoomerAMGSetNumGridSweeps ( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_BoomerAMGSetNumSweeps ( HYPRE_Solver solver , int num_sweeps );
int HYPRE_BoomerAMGSetCycleNumSweeps ( HYPRE_Solver solver , int num_sweeps , int k );
int HYPRE_BoomerAMGGetCycleNumSweeps ( HYPRE_Solver solver , int *num_sweeps , int k );
int HYPRE_BoomerAMGInitGridRelaxation ( int **num_grid_sweeps_ptr , int **grid_relax_type_ptr , int ***grid_relax_points_ptr , int coarsen_type , double **relax_weights_ptr , int max_levels );
int HYPRE_BoomerAMGSetGridRelaxType ( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_BoomerAMGSetRelaxType ( HYPRE_Solver solver , int relax_type );
int HYPRE_BoomerAMGSetCycleRelaxType ( HYPRE_Solver solver , int relax_type , int k );
int HYPRE_BoomerAMGGetCycleRelaxType ( HYPRE_Solver solver , int *relax_type , int k );
int HYPRE_BoomerAMGSetRelaxOrder ( HYPRE_Solver solver , int relax_order );
int HYPRE_BoomerAMGSetGridRelaxPoints ( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_BoomerAMGSetRelaxWeight ( HYPRE_Solver solver , double *relax_weight );
int HYPRE_BoomerAMGSetRelaxWt ( HYPRE_Solver solver , double relax_wt );
int HYPRE_BoomerAMGSetLevelRelaxWt ( HYPRE_Solver solver , double relax_wt , int level );
int HYPRE_BoomerAMGSetOmega ( HYPRE_Solver solver , double *omega );
int HYPRE_BoomerAMGSetOuterWt ( HYPRE_Solver solver , double outer_wt );
int HYPRE_BoomerAMGSetLevelOuterWt ( HYPRE_Solver solver , double outer_wt , int level );
int HYPRE_BoomerAMGSetSmoothType ( HYPRE_Solver solver , int smooth_type );
int HYPRE_BoomerAMGGetSmoothType ( HYPRE_Solver solver , int *smooth_type );
int HYPRE_BoomerAMGSetSmoothNumLevels ( HYPRE_Solver solver , int smooth_num_levels );
int HYPRE_BoomerAMGGetSmoothNumLevels ( HYPRE_Solver solver , int *smooth_num_levels );
int HYPRE_BoomerAMGSetSmoothNumSweeps ( HYPRE_Solver solver , int smooth_num_sweeps );
int HYPRE_BoomerAMGGetSmoothNumSweeps ( HYPRE_Solver solver , int *smooth_num_sweeps );
int HYPRE_BoomerAMGSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_BoomerAMGGetLogging ( HYPRE_Solver solver , int *logging );
int HYPRE_BoomerAMGSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_BoomerAMGGetPrintLevel ( HYPRE_Solver solver , int *print_level );
int HYPRE_BoomerAMGSetPrintFileName ( HYPRE_Solver solver , const char *print_file_name );
int HYPRE_BoomerAMGSetDebugFlag ( HYPRE_Solver solver , int debug_flag );
int HYPRE_BoomerAMGGetDebugFlag ( HYPRE_Solver solver , int *debug_flag );
int HYPRE_BoomerAMGGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_BoomerAMGGetCumNumIterations ( HYPRE_Solver solver , int *cum_num_iterations );
int HYPRE_BoomerAMGGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );
int HYPRE_BoomerAMGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *rel_resid_norm );
int HYPRE_BoomerAMGSetVariant ( HYPRE_Solver solver , int variant );
int HYPRE_BoomerAMGGetVariant ( HYPRE_Solver solver , int *variant );
int HYPRE_BoomerAMGSetOverlap ( HYPRE_Solver solver , int overlap );
int HYPRE_BoomerAMGGetOverlap ( HYPRE_Solver solver , int *overlap );
int HYPRE_BoomerAMGSetDomainType ( HYPRE_Solver solver , int domain_type );
int HYPRE_BoomerAMGGetDomainType ( HYPRE_Solver solver , int *domain_type );
int HYPRE_BoomerAMGSetSchwarzRlxWeight ( HYPRE_Solver solver , double schwarz_rlx_weight );
int HYPRE_BoomerAMGGetSchwarzRlxWeight ( HYPRE_Solver solver , double *schwarz_rlx_weight );
int HYPRE_BoomerAMGSetSchwarzUseNonSymm ( HYPRE_Solver solver , int use_nonsymm );
int HYPRE_BoomerAMGSetSym ( HYPRE_Solver solver , int sym );
int HYPRE_BoomerAMGSetLevel ( HYPRE_Solver solver , int level );
int HYPRE_BoomerAMGSetThreshold ( HYPRE_Solver solver , double threshold );
int HYPRE_BoomerAMGSetFilter ( HYPRE_Solver solver , double filter );
int HYPRE_BoomerAMGSetDropTol ( HYPRE_Solver solver , double drop_tol );
int HYPRE_BoomerAMGSetMaxNzPerRow ( HYPRE_Solver solver , int max_nz_per_row );
int HYPRE_BoomerAMGSetEuclidFile ( HYPRE_Solver solver , char *euclidfile );
int HYPRE_BoomerAMGSetEuLevel ( HYPRE_Solver solver , int eu_level );
int HYPRE_BoomerAMGSetEuSparseA ( HYPRE_Solver solver , double eu_sparse_A );
int HYPRE_BoomerAMGSetEuBJ ( HYPRE_Solver solver , int eu_bj );
int HYPRE_BoomerAMGSetNumFunctions ( HYPRE_Solver solver , int num_functions );
int HYPRE_BoomerAMGGetNumFunctions ( HYPRE_Solver solver , int *num_functions );
int HYPRE_BoomerAMGSetNodal ( HYPRE_Solver solver , int nodal );
int HYPRE_BoomerAMGSetNodalDiag ( HYPRE_Solver solver , int nodal );
int HYPRE_BoomerAMGSetDofFunc ( HYPRE_Solver solver , int *dof_func );
int HYPRE_BoomerAMGSetNumPaths ( HYPRE_Solver solver , int num_paths );
int HYPRE_BoomerAMGSetAggNumLevels ( HYPRE_Solver solver , int agg_num_levels );
int HYPRE_BoomerAMGSetNumCRRelaxSteps ( HYPRE_Solver solver , int num_CR_relax_steps );
int HYPRE_BoomerAMGSetCRRate ( HYPRE_Solver solver , double CR_rate );
int HYPRE_BoomerAMGSetCRStrongTh ( HYPRE_Solver solver , double CR_strong_th );
int HYPRE_BoomerAMGSetISType ( HYPRE_Solver solver , int IS_type );
int HYPRE_BoomerAMGSetCRUseCG ( HYPRE_Solver solver , int CR_use_CG );
int HYPRE_BoomerAMGSetGSMG ( HYPRE_Solver solver , int gsmg );
int HYPRE_BoomerAMGSetNumSamples ( HYPRE_Solver solver , int gsmg );
int HYPRE_BoomerAMGSetCGCIts ( HYPRE_Solver solver , int its );
int HYPRE_BoomerAMGSetPlotGrids ( HYPRE_Solver solver , int plotgrids );
int HYPRE_BoomerAMGSetPlotFileName ( HYPRE_Solver solver , const char *plotfilename );
int HYPRE_BoomerAMGSetCoordDim ( HYPRE_Solver solver , int coorddim );
int HYPRE_BoomerAMGSetCoordinates ( HYPRE_Solver solver , float *coordinates );

/* HYPRE_parcsr_bicgstab.c */
int HYPRE_ParCSRBiCGSTABCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRBiCGSTABDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRBiCGSTABSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSTABSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSTABSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRBiCGSTABSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_ParCSRBiCGSTABSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRBiCGSTABSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRBiCGSTABSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRBiCGSTABSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRBiCGSTABGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRBiCGSTABSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRBiCGSTABSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_ParCSRBiCGSTABGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_block.c */
int HYPRE_BlockTridiagCreate ( HYPRE_Solver *solver );
int HYPRE_BlockTridiagDestroy ( HYPRE_Solver solver );
int HYPRE_BlockTridiagSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BlockTridiagSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BlockTridiagSetIndexSet ( HYPRE_Solver solver , int n , int *inds );
int HYPRE_BlockTridiagSetAMGStrengthThreshold ( HYPRE_Solver solver , double thresh );
int HYPRE_BlockTridiagSetAMGNumSweeps ( HYPRE_Solver solver , int num_sweeps );
int HYPRE_BlockTridiagSetAMGRelaxType ( HYPRE_Solver solver , int relax_type );
int HYPRE_BlockTridiagSetPrintLevel ( HYPRE_Solver solver , int print_level );

/* HYPRE_parcsr_cgnr.c */
int HYPRE_ParCSRCGNRCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRCGNRDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRCGNRSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRCGNRSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRCGNRSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRCGNRSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRCGNRSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRCGNRSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRCGNRSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precondT , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRCGNRGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRCGNRSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRCGNRGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_Euclid.c */
int HYPRE_EuclidCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_EuclidDestroy ( HYPRE_Solver solver );
int HYPRE_EuclidSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_EuclidSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector bb , HYPRE_ParVector xx );
int HYPRE_EuclidSetParams ( HYPRE_Solver solver , int argc , char *argv []);
int HYPRE_EuclidSetParamsFromFile ( HYPRE_Solver solver , char *filename );
int HYPRE_EuclidSetLevel ( HYPRE_Solver solver , int level );
int HYPRE_EuclidSetBJ ( HYPRE_Solver solver , int bj );
int HYPRE_EuclidSetStats ( HYPRE_Solver solver , int eu_stats );
int HYPRE_EuclidSetMem ( HYPRE_Solver solver , int eu_mem );
int HYPRE_EuclidSetILUT ( HYPRE_Solver solver , double ilut );
int HYPRE_EuclidSetSparseA ( HYPRE_Solver solver , double sparse_A );
int HYPRE_EuclidSetRowScale ( HYPRE_Solver solver , int row_scale );

/* HYPRE_parcsr_flexgmres.c */
int HYPRE_ParCSRFlexGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRFlexGMRESDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRFlexGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRFlexGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRFlexGMRESSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_ParCSRFlexGMRESSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRFlexGMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_ParCSRFlexGMRESSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRFlexGMRESSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRFlexGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRFlexGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRFlexGMRESSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRFlexGMRESSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_ParCSRFlexGMRESGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_ParCSRFlexGMRESSetModifyPC ( HYPRE_Solver solver , HYPRE_PtrToModifyPCFcn modify_pc );

/* HYPRE_parcsr_gmres.c */
int HYPRE_ParCSRGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRGMRESDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRGMRESSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_ParCSRGMRESSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRGMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_ParCSRGMRESSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRGMRESSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRGMRESSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRGMRESSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRGMRESSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_ParCSRGMRESGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_hybrid.c */
int HYPRE_ParCSRHybridCreate ( HYPRE_Solver *solver );
int HYPRE_ParCSRHybridDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRHybridSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRHybridSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRHybridSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRHybridSetAbsoluteTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRHybridSetConvergenceTol ( HYPRE_Solver solver , double cf_tol );
int HYPRE_ParCSRHybridSetDSCGMaxIter ( HYPRE_Solver solver , int dscg_max_its );
int HYPRE_ParCSRHybridSetPCGMaxIter ( HYPRE_Solver solver , int pcg_max_its );
int HYPRE_ParCSRHybridSetSetupType ( HYPRE_Solver solver , int setup_type );
int HYPRE_ParCSRHybridSetSolverType ( HYPRE_Solver solver , int solver_type );
int HYPRE_ParCSRHybridSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_ParCSRHybridSetTwoNorm ( HYPRE_Solver solver , int two_norm );
int HYPRE_ParCSRHybridSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRHybridSetRelChange ( HYPRE_Solver solver , int rel_change );
int HYPRE_ParCSRHybridSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRHybridSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRHybridSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_ParCSRHybridSetStrongThreshold ( HYPRE_Solver solver , double strong_threshold );
int HYPRE_ParCSRHybridSetMaxRowSum ( HYPRE_Solver solver , double max_row_sum );
int HYPRE_ParCSRHybridSetTruncFactor ( HYPRE_Solver solver , double trunc_factor );
int HYPRE_ParCSRHybridSetPMaxElmts ( HYPRE_Solver solver , int p_max );
int HYPRE_ParCSRHybridSetMaxLevels ( HYPRE_Solver solver , int max_levels );
int HYPRE_ParCSRHybridSetMeasureType ( HYPRE_Solver solver , int measure_type );
int HYPRE_ParCSRHybridSetCoarsenType ( HYPRE_Solver solver , int coarsen_type );
int HYPRE_ParCSRHybridSetInterpType ( HYPRE_Solver solver , int interp_type );
int HYPRE_ParCSRHybridSetCycleType ( HYPRE_Solver solver , int cycle_type );
int HYPRE_ParCSRHybridSetNumGridSweeps ( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_ParCSRHybridSetGridRelaxType ( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_ParCSRHybridSetGridRelaxPoints ( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_ParCSRHybridSetNumSweeps ( HYPRE_Solver solver , int num_sweeps );
int HYPRE_ParCSRHybridSetCycleNumSweeps ( HYPRE_Solver solver , int num_sweeps , int k );
int HYPRE_ParCSRHybridSetRelaxType ( HYPRE_Solver solver , int relax_type );
int HYPRE_ParCSRHybridSetCycleRelaxType ( HYPRE_Solver solver , int relax_type , int k );
int HYPRE_ParCSRHybridSetRelaxOrder ( HYPRE_Solver solver , int relax_order );
int HYPRE_ParCSRHybridSetRelaxWt ( HYPRE_Solver solver , double relax_wt );
int HYPRE_ParCSRHybridSetLevelRelaxWt ( HYPRE_Solver solver , double relax_wt , int level );
int HYPRE_ParCSRHybridSetOuterWt ( HYPRE_Solver solver , double outer_wt );
int HYPRE_ParCSRHybridSetLevelOuterWt ( HYPRE_Solver solver , double outer_wt , int level );
int HYPRE_ParCSRHybridSetRelaxWeight ( HYPRE_Solver solver , double *relax_weight );
int HYPRE_ParCSRHybridSetOmega ( HYPRE_Solver solver , double *omega );
int HYPRE_ParCSRHybridSetAggNumLevels ( HYPRE_Solver solver , int agg_num_levels );
int HYPRE_ParCSRHybridSetNumPaths ( HYPRE_Solver solver , int num_paths );
int HYPRE_ParCSRHybridSetNumFunctions ( HYPRE_Solver solver , int num_functions );
int HYPRE_ParCSRHybridSetNodal ( HYPRE_Solver solver , int nodal );
int HYPRE_ParCSRHybridSetDofFunc ( HYPRE_Solver solver , int *dof_func );
int HYPRE_ParCSRHybridGetNumIterations ( HYPRE_Solver solver , int *num_its );
int HYPRE_ParCSRHybridGetDSCGNumIterations ( HYPRE_Solver solver , int *dscg_num_its );
int HYPRE_ParCSRHybridGetPCGNumIterations ( HYPRE_Solver solver , int *pcg_num_its );
int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_int.c */
int hypre_ParSetRandomValues ( void *v , int seed );
int hypre_ParPrintVector ( void *v , const char *file );
void *hypre_ParReadVector ( MPI_Comm comm , const char *file );
int hypre_ParVectorSize ( void *x );
int hypre_ParCSRMultiVectorPrint ( void *x_ , const char *fileName );
void *hypre_ParCSRMultiVectorRead ( MPI_Comm comm , void *ii_ , const char *fileName );
int aux_maskCount ( int n , int *mask );
void aux_indexFromMask ( int n , int *mask , int *index );
int HYPRE_TempParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
int HYPRE_ParCSRSetupInterpreter ( mv_InterfaceInterpreter *i );
int HYPRE_ParCSRSetupMatvec ( HYPRE_MatvecFunctions *mv );

/* HYPRE_parcsr_lgmres.c */
int HYPRE_ParCSRLGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRLGMRESDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRLGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRLGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRLGMRESSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_ParCSRLGMRESSetAugDim ( HYPRE_Solver solver , int aug_dim );
int HYPRE_ParCSRLGMRESSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRLGMRESSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_ParCSRLGMRESSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRLGMRESSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRLGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRLGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRLGMRESSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRLGMRESSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_ParCSRLGMRESGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_ParaSails.c */
int HYPRE_ParCSRParaSailsCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRParaSailsDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRParaSailsSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRParaSailsSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRParaSailsSetParams ( HYPRE_Solver solver , double thresh , int nlevels );
int HYPRE_ParCSRParaSailsSetFilter ( HYPRE_Solver solver , double filter );
int HYPRE_ParCSRParaSailsGetFilter ( HYPRE_Solver solver , double *filter );
int HYPRE_ParCSRParaSailsSetSym ( HYPRE_Solver solver , int sym );
int HYPRE_ParCSRParaSailsSetLoadbal ( HYPRE_Solver solver , double loadbal );
int HYPRE_ParCSRParaSailsGetLoadbal ( HYPRE_Solver solver , double *loadbal );
int HYPRE_ParCSRParaSailsSetReuse ( HYPRE_Solver solver , int reuse );
int HYPRE_ParCSRParaSailsSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParaSailsCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParaSailsDestroy ( HYPRE_Solver solver );
int HYPRE_ParaSailsSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParaSailsSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParaSailsSetParams ( HYPRE_Solver solver , double thresh , int nlevels );
int HYPRE_ParaSailsSetThresh ( HYPRE_Solver solver , double thresh );
int HYPRE_ParaSailsGetThresh ( HYPRE_Solver solver , double *thresh );
int HYPRE_ParaSailsSetNlevels ( HYPRE_Solver solver , int nlevels );
int HYPRE_ParaSailsGetNlevels ( HYPRE_Solver solver , int *nlevels );
int HYPRE_ParaSailsSetFilter ( HYPRE_Solver solver , double filter );
int HYPRE_ParaSailsGetFilter ( HYPRE_Solver solver , double *filter );
int HYPRE_ParaSailsSetSym ( HYPRE_Solver solver , int sym );
int HYPRE_ParaSailsGetSym ( HYPRE_Solver solver , int *sym );
int HYPRE_ParaSailsSetLoadbal ( HYPRE_Solver solver , double loadbal );
int HYPRE_ParaSailsGetLoadbal ( HYPRE_Solver solver , double *loadbal );
int HYPRE_ParaSailsSetReuse ( HYPRE_Solver solver , int reuse );
int HYPRE_ParaSailsGetReuse ( HYPRE_Solver solver , int *reuse );
int HYPRE_ParaSailsSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParaSailsGetLogging ( HYPRE_Solver solver , int *logging );
int HYPRE_ParaSailsBuildIJMatrix ( HYPRE_Solver solver , HYPRE_IJMatrix *pij_A );

/* HYPRE_parcsr_pcg.c */
int HYPRE_ParCSRPCGCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRPCGDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRPCGSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRPCGSetAbsoluteTol ( HYPRE_Solver solver , double a_tol );
int HYPRE_ParCSRPCGSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRPCGSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRPCGSetTwoNorm ( HYPRE_Solver solver , int two_norm );
int HYPRE_ParCSRPCGSetRelChange ( HYPRE_Solver solver , int rel_change );
int HYPRE_ParCSRPCGSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRPCGGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRPCGSetPrintLevel ( HYPRE_Solver solver , int level );
int HYPRE_ParCSRPCGSetLogging ( HYPRE_Solver solver , int level );
int HYPRE_ParCSRPCGGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_ParCSRDiagScaleSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector y , HYPRE_ParVector x );
int HYPRE_ParCSRDiagScale ( HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx );

/* HYPRE_parcsr_pilut.c */
int HYPRE_ParCSRPilutCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRPilutDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRPilutSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPilutSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPilutSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRPilutSetDropTolerance ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRPilutSetFactorRowSize ( HYPRE_Solver solver , int size );

/* HYPRE_parcsr_schwarz.c */
int HYPRE_SchwarzCreate ( HYPRE_Solver *solver );
int HYPRE_SchwarzDestroy ( HYPRE_Solver solver );
int HYPRE_SchwarzSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_SchwarzSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_SchwarzSetVariant ( HYPRE_Solver solver , int variant );
int HYPRE_SchwarzSetOverlap ( HYPRE_Solver solver , int overlap );
int HYPRE_SchwarzSetDomainType ( HYPRE_Solver solver , int domain_type );
int HYPRE_SchwarzSetDomainStructure ( HYPRE_Solver solver , HYPRE_CSRMatrix domain_structure );
int HYPRE_SchwarzSetNumFunctions ( HYPRE_Solver solver , int num_functions );
int HYPRE_SchwarzSetNonSymm ( HYPRE_Solver solver , int use_nonsymm );
int HYPRE_SchwarzSetRelaxWeight ( HYPRE_Solver solver , double relax_weight );
int HYPRE_SchwarzSetDofFunc ( HYPRE_Solver solver , int *dof_func );

/* par_amg.c */
void *hypre_BoomerAMGCreate ( void );
int hypre_BoomerAMGDestroy ( void *data );
int hypre_BoomerAMGSetRestriction ( void *data , int restr_par );
int hypre_BoomerAMGSetMaxLevels ( void *data , int max_levels );
int hypre_BoomerAMGGetMaxLevels ( void *data , int *max_levels );
int hypre_BoomerAMGSetStrongThreshold ( void *data , double strong_threshold );
int hypre_BoomerAMGGetStrongThreshold ( void *data , double *strong_threshold );
int hypre_BoomerAMGSetMaxRowSum ( void *data , double max_row_sum );
int hypre_BoomerAMGGetMaxRowSum ( void *data , double *max_row_sum );
int hypre_BoomerAMGSetTruncFactor ( void *data , double trunc_factor );
int hypre_BoomerAMGGetTruncFactor ( void *data , double *trunc_factor );
int hypre_BoomerAMGSetPMaxElmts ( void *data , int P_max_elmts );
int hypre_BoomerAMGGetPMaxElmts ( void *data , int *P_max_elmts );
int hypre_BoomerAMGSetJacobiTruncThreshold ( void *data , double jacobi_trunc_threshold );
int hypre_BoomerAMGGetJacobiTruncThreshold ( void *data , double *jacobi_trunc_threshold );
int hypre_BoomerAMGSetPostInterpType ( void *data , int post_interp_type );
int hypre_BoomerAMGGetPostInterpType ( void *data , int *post_interp_type );
int hypre_BoomerAMGSetSCommPkgSwitch ( void *data , double S_commpkg_switch );
int hypre_BoomerAMGGetSCommPkgSwitch ( void *data , double *S_commpkg_switch );
int hypre_BoomerAMGSetInterpType ( void *data , int interp_type );
int hypre_BoomerAMGGetInterpType ( void *data , int *interp_type );
int hypre_BoomerAMGSetMinIter ( void *data , int min_iter );
int hypre_BoomerAMGGetMinIter ( void *data , int *min_iter );
int hypre_BoomerAMGSetMaxIter ( void *data , int max_iter );
int hypre_BoomerAMGGetMaxIter ( void *data , int *max_iter );
int hypre_BoomerAMGSetCoarsenType ( void *data , int coarsen_type );
int hypre_BoomerAMGGetCoarsenType ( void *data , int *coarsen_type );
int hypre_BoomerAMGSetMeasureType ( void *data , int measure_type );
int hypre_BoomerAMGGetMeasureType ( void *data , int *measure_type );
int hypre_BoomerAMGSetSetupType ( void *data , int setup_type );
int hypre_BoomerAMGGetSetupType ( void *data , int *setup_type );
int hypre_BoomerAMGSetCycleType ( void *data , int cycle_type );
int hypre_BoomerAMGGetCycleType ( void *data , int *cycle_type );
int hypre_BoomerAMGSetTol ( void *data , double tol );
int hypre_BoomerAMGGetTol ( void *data , double *tol );
int hypre_BoomerAMGSetNumSweeps ( void *data , int num_sweeps );
int hypre_BoomerAMGSetCycleNumSweeps ( void *data , int num_sweeps , int k );
int hypre_BoomerAMGGetCycleNumSweeps ( void *data , int *num_sweeps , int k );
int hypre_BoomerAMGSetNumGridSweeps ( void *data , int *num_grid_sweeps );
int hypre_BoomerAMGGetNumGridSweeps ( void *data , int **num_grid_sweeps );
int hypre_BoomerAMGSetRelaxType ( void *data , int relax_type );
int hypre_BoomerAMGSetCycleRelaxType ( void *data , int relax_type , int k );
int hypre_BoomerAMGGetCycleRelaxType ( void *data , int *relax_type , int k );
int hypre_BoomerAMGSetRelaxOrder ( void *data , int relax_order );
int hypre_BoomerAMGGetRelaxOrder ( void *data , int *relax_order );
int hypre_BoomerAMGSetGridRelaxType ( void *data , int *grid_relax_type );
int hypre_BoomerAMGGetGridRelaxType ( void *data , int **grid_relax_type );
int hypre_BoomerAMGSetGridRelaxPoints ( void *data , int **grid_relax_points );
int hypre_BoomerAMGGetGridRelaxPoints ( void *data , int ***grid_relax_points );
int hypre_BoomerAMGSetRelaxWeight ( void *data , double *relax_weight );
int hypre_BoomerAMGGetRelaxWeight ( void *data , double **relax_weight );
int hypre_BoomerAMGSetRelaxWt ( void *data , double relax_weight );
int hypre_BoomerAMGSetLevelRelaxWt ( void *data , double relax_weight , int level );
int hypre_BoomerAMGGetLevelRelaxWt ( void *data , double *relax_weight , int level );
int hypre_BoomerAMGSetOmega ( void *data , double *omega );
int hypre_BoomerAMGGetOmega ( void *data , double **omega );
int hypre_BoomerAMGSetOuterWt ( void *data , double omega );
int hypre_BoomerAMGSetLevelOuterWt ( void *data , double omega , int level );
int hypre_BoomerAMGGetLevelOuterWt ( void *data , double *omega , int level );
int hypre_BoomerAMGSetSmoothType ( void *data , int smooth_type );
int hypre_BoomerAMGGetSmoothType ( void *data , int *smooth_type );
int hypre_BoomerAMGSetSmoothNumLevels ( void *data , int smooth_num_levels );
int hypre_BoomerAMGGetSmoothNumLevels ( void *data , int *smooth_num_levels );
int hypre_BoomerAMGSetSmoothNumSweeps ( void *data , int smooth_num_sweeps );
int hypre_BoomerAMGGetSmoothNumSweeps ( void *data , int *smooth_num_sweeps );
int hypre_BoomerAMGSetLogging ( void *data , int logging );
int hypre_BoomerAMGGetLogging ( void *data , int *logging );
int hypre_BoomerAMGSetPrintLevel ( void *data , int print_level );
int hypre_BoomerAMGGetPrintLevel ( void *data , int *print_level );
int hypre_BoomerAMGSetPrintFileName ( void *data , const char *print_file_name );
int hypre_BoomerAMGGetPrintFileName ( void *data , char **print_file_name );
int hypre_BoomerAMGSetNumIterations ( void *data , int num_iterations );
int hypre_BoomerAMGSetDebugFlag ( void *data , int debug_flag );
int hypre_BoomerAMGGetDebugFlag ( void *data , int *debug_flag );
int hypre_BoomerAMGSetGSMG ( void *data , int par );
int hypre_BoomerAMGSetNumSamples ( void *data , int par );
int hypre_BoomerAMGSetCGCIts ( void *data , int its );
int hypre_BoomerAMGSetPlotGrids ( void *data , int plotgrids );
int hypre_BoomerAMGSetPlotFileName ( void *data , const char *plot_file_name );
int hypre_BoomerAMGSetCoordDim ( void *data , int coorddim );
int hypre_BoomerAMGSetCoordinates ( void *data , float *coordinates );
int hypre_BoomerAMGSetNumFunctions ( void *data , int num_functions );
int hypre_BoomerAMGGetNumFunctions ( void *data , int *num_functions );
int hypre_BoomerAMGSetNodal ( void *data , int nodal );
int hypre_BoomerAMGSetNodalDiag ( void *data , int nodal );
int hypre_BoomerAMGSetNumPaths ( void *data , int num_paths );
int hypre_BoomerAMGSetAggNumLevels ( void *data , int agg_num_levels );
int hypre_BoomerAMGSetNumCRRelaxSteps ( void *data , int num_CR_relax_steps );
int hypre_BoomerAMGSetCRRate ( void *data , double CR_rate );
int hypre_BoomerAMGSetCRStrongTh ( void *data , double CR_strong_th );
int hypre_BoomerAMGSetISType ( void *data , int IS_type );
int hypre_BoomerAMGSetCRUseCG ( void *data , int CR_use_CG );
int hypre_BoomerAMGSetNumPoints ( void *data , int num_points );
int hypre_BoomerAMGSetDofFunc ( void *data , int *dof_func );
int hypre_BoomerAMGSetPointDofMap ( void *data , int *point_dof_map );
int hypre_BoomerAMGSetDofPoint ( void *data , int *dof_point );
int hypre_BoomerAMGGetNumIterations ( void *data , int *num_iterations );
int hypre_BoomerAMGGetCumNumIterations ( void *data , int *cum_num_iterations );
int hypre_BoomerAMGGetResidual ( void *data , hypre_ParVector **resid );
int hypre_BoomerAMGGetRelResidualNorm ( void *data , double *rel_resid_norm );
int hypre_BoomerAMGSetVariant ( void *data , int variant );
int hypre_BoomerAMGGetVariant ( void *data , int *variant );
int hypre_BoomerAMGSetOverlap ( void *data , int overlap );
int hypre_BoomerAMGGetOverlap ( void *data , int *overlap );
int hypre_BoomerAMGSetDomainType ( void *data , int domain_type );
int hypre_BoomerAMGGetDomainType ( void *data , int *domain_type );
int hypre_BoomerAMGSetSchwarzRlxWeight ( void *data , double schwarz_rlx_weight );
int hypre_BoomerAMGGetSchwarzRlxWeight ( void *data , double *schwarz_rlx_weight );
int hypre_BoomerAMGSetSchwarzUseNonSymm ( void *data , int use_nonsymm );
int hypre_BoomerAMGSetSym ( void *data , int sym );
int hypre_BoomerAMGSetLevel ( void *data , int level );
int hypre_BoomerAMGSetThreshold ( void *data , double thresh );
int hypre_BoomerAMGSetFilter ( void *data , double filter );
int hypre_BoomerAMGSetDropTol ( void *data , double drop_tol );
int hypre_BoomerAMGSetMaxNzPerRow ( void *data , int max_nz_per_row );
int hypre_BoomerAMGSetEuclidFile ( void *data , char *euclidfile );
int hypre_BoomerAMGSetEuLevel ( void *data , int eu_level );
int hypre_BoomerAMGSetEuSparseA ( void *data , double eu_sparse_A );
int hypre_BoomerAMGSetEuBJ ( void *data , int eu_bj );

/* par_amg_setup.c */
int hypre_BoomerAMGSetup ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solve.c */
int hypre_BoomerAMGSolve ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solveT.c */
int hypre_BoomerAMGSolveT ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
int hypre_BoomerAMGCycleT ( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );
int hypre_BoomerAMGRelaxT ( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp );

/* par_cgc_coarsen.c */
int hypre_BoomerAMGCoarsenCGCb ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int coarsen_type , int cgc_its , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenCGC ( hypre_ParCSRMatrix *S , int numberofgrids , int coarsen_type , int *CF_marker );
int AmgCGCPrepare ( hypre_ParCSRMatrix *S , int nlocal , int *CF_marker , int **CF_marker_offd , int coarsen_type , int **vrange );
int AmgCGCGraphAssemble ( hypre_ParCSRMatrix *S , int *vertexrange , int *CF_marker , int *CF_marker_offd , int coarsen_type , HYPRE_IJMatrix *ijG );
int AmgCGCChoose ( hypre_CSRMatrix *G , int *vertexrange , int mpisize , int **coarse );
int AmgCGCBoundaryFix ( hypre_ParCSRMatrix *S , int *CF_marker , int *CF_marker_offd );

/* par_cg_relax_wt.c */
int hypre_BoomerAMGCGRelaxWt ( void *amg_vdata , int level , int num_cg_sweeps , double *rlx_wt_ptr );
int hypre_Bisection ( int n , double *diag , double *offd , double y , double z , double tol , int k , double *ev_ptr );

/* par_coarsen.c */
int hypre_BoomerAMGCoarsen ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int CF_init , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenRuge ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int coarsen_type , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenFalgout ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenHMIS ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenPMIS ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int CF_init , int debug_flag , int **CF_marker_ptr );

/* par_coarse_parms.c */
int hypre_BoomerAMGCoarseParms ( MPI_Comm comm , int local_num_variables , int num_functions , int *dof_func , int *CF_marker , int **coarse_dof_func_ptr , int **coarse_pnts_global_ptr );

/* par_coordinates.c */
float *GenerateCoordinates ( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , int coorddim );

/* par_cr.c */
int hypre_BoomerAMGCoarsenCR1 ( hypre_ParCSRMatrix *A , int **CF_marker_ptr , int *coarse_size_ptr , int num_CR_relax_steps , int IS_type , int CRaddCpoints );
int cr ( int *A_i , int *A_j , double *A_data , int n , int *cf , int rlx , double omega , double tg , int mu );
int GraphAdd ( Link *list , int *head , int *tail , int index , int istack );
int GraphRemove ( Link *list , int *head , int *tail , int index );
int IndepSetGreedy ( int *A_i , int *A_j , int n , int *cf );
int IndepSetGreedyS ( int *A_i , int *A_j , int n , int *cf );
int fptjaccr ( int *cf , int *A_i , int *A_j , double *A_data , int n , double *e0 , double omega , double *e1 );
int fptgscr ( int *cf , int *A_i , int *A_j , double *A_data , int n , double *e0 , double *e1 );
int formu ( int *cf , int n , double *e1 , int *A_i , double rho );
int hypre_BoomerAMGIndepRS ( hypre_ParCSRMatrix *S , int measure_type , int debug_flag , int *CF_marker );
int hypre_BoomerAMGIndepRSa ( hypre_ParCSRMatrix *S , int measure_type , int debug_flag , int *CF_marker );
int hypre_BoomerAMGIndepHMIS ( hypre_ParCSRMatrix *S , int measure_type , int debug_flag , int *CF_marker );
int hypre_BoomerAMGIndepHMISa ( hypre_ParCSRMatrix *S , int measure_type , int debug_flag , int *CF_marker );
int hypre_BoomerAMGIndepPMIS ( hypre_ParCSRMatrix *S , int CF_init , int debug_flag , int *CF_marker );
int hypre_BoomerAMGIndepPMISa ( hypre_ParCSRMatrix *S , int CF_init , int debug_flag , int *CF_marker );
int hypre_BoomerAMGCoarsenCR ( hypre_ParCSRMatrix *A , int **CF_marker_ptr , int *coarse_size_ptr , int num_CR_relax_steps , int IS_type , int num_functions , int rlx_type , double relax_weight , double omega , double theta , HYPRE_Solver smoother , hypre_ParCSRMatrix *AN , int useCG , hypre_ParCSRMatrix *S );

/* par_cycle.c */
int hypre_BoomerAMGCycle ( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );

/* par_difconv.c */
HYPRE_ParCSRMatrix GenerateDifConv ( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );

/* par_gsmg.c */
int hypre_ParCSRMatrixClone ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **Sp , int copy_data );
int hypre_ParCSRMatrixFillSmooth ( int nsamples , double *samples , hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int num_functions , int *dof_func );
double hypre_ParCSRMatrixChooseThresh ( hypre_ParCSRMatrix *S );
int hypre_ParCSRMatrixThreshold ( hypre_ParCSRMatrix *A , double thresh );
int hypre_BoomerAMGCreateSmoothVecs ( void *data , hypre_ParCSRMatrix *A , int num_sweeps , int level , double **SmoothVecs_p );
int hypre_BoomerAMGCreateSmoothDirs ( void *data , hypre_ParCSRMatrix *A , double *SmoothVecs , double thresh , int num_functions , int *dof_func , hypre_ParCSRMatrix **S_ptr );
int hypre_BoomerAMGNormalizeVecs ( int n , int num , double *V );
int hypre_BoomerAMGFitVectors ( int ip , int n , int num , const double *V , int nc , const int *ind , double *val );
int hypre_BoomerAMGBuildInterpLS ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int num_smooth , double *SmoothVecs , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildInterpGSMG ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , hypre_ParCSRMatrix **P_ptr );

/* par_indepset.c */
int hypre_BoomerAMGIndepSetInit ( hypre_ParCSRMatrix *S , double *measure_array , int seq_rand );
int hypre_BoomerAMGIndepSet ( hypre_ParCSRMatrix *S , double *measure_array , int *graph_array , int graph_array_size , int *graph_array_offd , int graph_array_offd_size , int *IS_marker , int *IS_marker_offd );

/* par_interp.c */
int hypre_BoomerAMGBuildInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildInterpHE ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildDirInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGInterpTruncation ( hypre_ParCSRMatrix *P , double trunc_factor , int max_elmts );
void hypre_qsort2abs ( int *v , double *w , int left , int right );
int hypre_BoomerAMGBuildInterpModUnk ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_jacobi_interp.c */
void hypre_BoomerAMGJacobiInterp ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , hypre_ParCSRMatrix *S , int num_functions , int *dof_func , int *CF_marker , int level , double truncation_threshold , double truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_1 ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , hypre_ParCSRMatrix *S , int *CF_marker , int level , double truncation_threshold , double truncation_threshold_minus , int *dof_func , int *dof_func_offd , double weight_AF );
void hypre_BoomerAMGTruncateInterp ( hypre_ParCSRMatrix *P , double eps , double dlt , int *CF_marker );
int hypre_ParCSRMatrix_dof_func_offd ( hypre_ParCSRMatrix *A , int num_functions , int *dof_func , int **dof_func_offd );

/* par_laplace_27pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian27pt ( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );
int hypre_map3 ( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* par_laplace_9pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian9pt ( MPI_Comm comm , int nx , int ny , int P , int Q , int p , int q , double *value );
int hypre_map2 ( int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part );

/* par_laplace.c */
HYPRE_ParCSRMatrix GenerateLaplacian ( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );
int hypre_map ( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );
HYPRE_ParCSRMatrix GenerateSysLaplacian ( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , int num_fun , double *mtrx , double *value );
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef ( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , int num_fun , double *mtrx , double *value );

/* par_lr_interp.c */
int hypre_BoomerAMGBuildStdInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int sep_weight , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildExtPIInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildExtPICCInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildFFInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildFF1Interp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_multi_interp.c */
int hypre_BoomerAMGBuildMultipass ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int P_max_elmts , int weight_option , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_nodal_systems.c */
int hypre_BoomerAMGCreateNodalA ( hypre_ParCSRMatrix *A , int num_functions , int *dof_func , int option , int diag_option , hypre_ParCSRMatrix **AN_ptr );
int hypre_BoomerAMGCreateScalarCFS ( hypre_ParCSRMatrix *SN , int *CFN_marker , int *col_offd_SN_to_AN , int num_functions , int nodal , int data , int **dof_func_ptr , int **CF_marker_ptr , int **col_offd_S_to_A_ptr , hypre_ParCSRMatrix **S_ptr );
int hypre_BoomerAMGCreateScalarCF ( int *CFN_marker , int num_functions , int num_nodes , int **dof_func_ptr , int **CF_marker_ptr );

/* par_rap.c */
hypre_CSRMatrix *hypre_ExchangeRAPData ( hypre_CSRMatrix *RAP_int , hypre_ParCSRCommPkg *comm_pkg_RT );
int hypre_BoomerAMGBuildCoarseOperator ( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr );

/* par_rap_communication.c */
int hypre_GetCommPkgRTFromCommPkgA ( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , int *fine_to_coarse_offd );
int hypre_GenerateSendMapAndCommPkg ( MPI_Comm comm , int num_sends , int num_recvs , int *recv_procs , int *send_procs , int *recv_vec_starts , hypre_ParCSRMatrix *A );

/* par_relax.c */
int hypre_BoomerAMGRelax ( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , double omega , hypre_ParVector *u , hypre_ParVector *Vtemp );
int gselim ( double *A , double *x , int n );

/* par_relax_interface.c */
int hypre_BoomerAMGRelaxIF ( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_order , int cycle_type , double relax_weight , double omega , hypre_ParVector *u , hypre_ParVector *Vtemp );

/* par_rotate_7pt.c */
HYPRE_ParCSRMatrix GenerateRotate7pt ( MPI_Comm comm , int nx , int ny , int P , int Q , int p , int q , double alpha , double eps );

/* par_scaled_matnorm.c */
int hypre_ParCSRMatrixScaledNorm ( hypre_ParCSRMatrix *A , double *scnorm );

/* par_schwarz.c */
void *hypre_SchwarzCreate ( void );
int hypre_SchwarzDestroy ( void *data );
int hypre_SchwarzSetup ( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
int hypre_SchwarzSolve ( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
int hypre_SchwarzCFSolve ( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u , int *CF_marker , int rlx_pt );
int hypre_SchwarzSetVariant ( void *data , int variant );
int hypre_SchwarzSetDomainType ( void *data , int domain_type );
int hypre_SchwarzSetOverlap ( void *data , int overlap );
int hypre_SchwarzSetNumFunctions ( void *data , int num_functions );
int hypre_SchwarzSetNonSymm ( void *data , int value );
int hypre_SchwarzSetRelaxWeight ( void *data , double relax_weight );
int hypre_SchwarzSetDomainStructure ( void *data , hypre_CSRMatrix *domain_structure );
int hypre_SchwarzSetScale ( void *data , double *scale );
int hypre_SchwarzReScale ( void *data , int size , double value );
int hypre_SchwarzSetDofFunc ( void *data , int *dof_func );

/* par_stats.c */
int hypre_BoomerAMGSetupStats ( void *amg_vdata , hypre_ParCSRMatrix *A );
int hypre_BoomerAMGWriteSolverParams ( void *data );

/* par_strength.c */
int hypre_BoomerAMGCreateS ( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int num_functions , int *dof_func , hypre_ParCSRMatrix **S_ptr );
int hypre_BoomerAMGCreateSabs ( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int num_functions , int *dof_func , hypre_ParCSRMatrix **S_ptr );
int hypre_BoomerAMGCreateSCommPkg ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *S , int **col_offd_S_to_A_ptr );
int hypre_BoomerAMGCreate2ndS ( hypre_ParCSRMatrix *S , int *CF_marker , int num_paths , int *coarse_row_starts , hypre_ParCSRMatrix **C_ptr );
int hypre_BoomerAMGCorrectCFMarker ( int *CF_marker , int num_var , int *new_CF_marker );

/* par_vardifconv.c */
HYPRE_ParCSRMatrix GenerateVarDifConv ( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double eps , HYPRE_ParVector *rhs_ptr );
double afun ( double xx , double yy , double zz );
double bfun ( double xx , double yy , double zz );
double cfun ( double xx , double yy , double zz );
double dfun ( double xx , double yy , double zz );
double efun ( double xx , double yy , double zz );
double ffun ( double xx , double yy , double zz );
double gfun ( double xx , double yy , double zz );
double rfun ( double xx , double yy , double zz );
double bndfun ( double xx , double yy , double zz );

/* pcg_par.c */
char *hypre_ParKrylovCAlloc ( int count , int elt_size );
int hypre_ParKrylovFree ( char *ptr );
void *hypre_ParKrylovCreateVector ( void *vvector );
void *hypre_ParKrylovCreateVectorArray ( int n , void *vvector );
int hypre_ParKrylovDestroyVector ( void *vvector );
void *hypre_ParKrylovMatvecCreate ( void *A , void *x );
int hypre_ParKrylovMatvec ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovMatvecT ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovMatvecDestroy ( void *matvec_data );
double hypre_ParKrylovInnerProd ( void *x , void *y );
int hypre_ParKrylovCopyVector ( void *x , void *y );
int hypre_ParKrylovClearVector ( void *x );
int hypre_ParKrylovScaleVector ( double alpha , void *x );
int hypre_ParKrylovAxpy ( double alpha , void *x , void *y );
int hypre_ParKrylovCommInfo ( void *A , int *my_id , int *num_procs );
int hypre_ParKrylovIdentitySetup ( void *vdata , void *A , void *b , void *x );
int hypre_ParKrylovIdentity ( void *vdata , void *A , void *b , void *x );

/* schwarz.c */
int hypre_AMGNodalSchwarzSmoother ( hypre_CSRMatrix *A , int num_functions , int option , hypre_CSRMatrix **domain_structure_pointer );
int hypre_ParMPSchwarzSolve ( hypre_ParCSRMatrix *par_A , hypre_CSRMatrix *A_boundary , hypre_ParVector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , double relax_wt , double *scale , hypre_ParVector *Vtemp , int *pivots , int use_nonsymm );
int hypre_MPSchwarzSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , double relax_wt , hypre_Vector *aux_vector , int *pivots , int use_nonsymm );
int hypre_MPSchwarzCFSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , double relax_wt , hypre_Vector *aux_vector , int *CF_marker , int rlx_pt , int *pivots , int use_nonsymm );
int hypre_MPSchwarzFWSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , double relax_wt , hypre_Vector *aux_vector , int *pivots , int use_nonsymm );
int hypre_MPSchwarzCFFWSolve ( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , double relax_wt , hypre_Vector *aux_vector , int *CF_marker , int rlx_pt , int *pivots , int use_nonsymm );
int transpose_matrix_create ( int **i_face_element_pointer , int **j_face_element_pointer , int *i_element_face , int *j_element_face , int num_elements , int num_faces );
int matrix_matrix_product ( int **i_element_edge_pointer , int **j_element_edge_pointer , int *i_element_face , int *j_element_face , int *i_face_edge , int *j_face_edge , int num_elements , int num_faces , int num_edges );
int hypre_AMGCreateDomainDof ( hypre_CSRMatrix *A , int domain_type , int overlap , int num_functions , int *dof_func , hypre_CSRMatrix **domain_structure_pointer , int **piv_pointer , int use_nonsymm );
int hypre_AMGeAgglomerate ( int *i_AE_element , int *j_AE_element , int *i_face_face , int *j_face_face , int *w_face_face , int *i_face_element , int *j_face_element , int *i_element_face , int *j_element_face , int *i_face_to_prefer_weight , int *i_face_weight , int num_faces , int num_elements , int *num_AEs_pointer );
int update_entry ( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int remove_entry ( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int move_entry ( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int matinv ( double *x , double *a , int k );
int hypre_parCorrRes ( hypre_ParCSRMatrix *A , hypre_ParVector *x , hypre_Vector *rhs , double **tmp_ptr );
int hypre_AdSchwarzSolve ( hypre_ParCSRMatrix *par_A , hypre_ParVector *par_rhs , hypre_CSRMatrix *domain_structure , double *scale , hypre_ParVector *par_x , hypre_ParVector *par_aux , int *pivots , int use_nonsymm );
int hypre_AdSchwarzCFSolve ( hypre_ParCSRMatrix *par_A , hypre_ParVector *par_rhs , hypre_CSRMatrix *domain_structure , double *scale , hypre_ParVector *par_x , hypre_ParVector *par_aux , int *CF_marker , int rlx_pt , int *pivots , int use_nonsymm );
int hypre_GenerateScale ( hypre_CSRMatrix *domain_structure , int num_variables , double relaxation_weight , double **scale_pointer );
int hypre_ParAdSchwarzSolve ( hypre_ParCSRMatrix *A , hypre_ParVector *F , hypre_CSRMatrix *domain_structure , double *scale , hypre_ParVector *X , hypre_ParVector *Vtemp , int *pivots , int use_nonsymm );
int hypre_ParAMGCreateDomainDof ( hypre_ParCSRMatrix *A , int domain_type , int overlap , int num_functions , int *dof_func , hypre_CSRMatrix **domain_structure_pointer , int **piv_pointer , int use_nonsymm );
int hypre_ParGenerateScale ( hypre_ParCSRMatrix *A , hypre_CSRMatrix *domain_structure , double relaxation_weight , double **scale_pointer );
int hypre_ParGenerateHybridScale ( hypre_ParCSRMatrix *A , hypre_CSRMatrix *domain_structure , hypre_CSRMatrix **A_boundary_pointer , double **scale_pointer );

#ifdef __cplusplus
}
#endif

#endif

