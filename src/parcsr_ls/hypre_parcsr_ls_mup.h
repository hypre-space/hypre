
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

#ifndef HYPRE_PARCSR_LS_MUP_HEADER
#define HYPRE_PARCSR_LS_MUP_HEADER

#include "_hypre_parcsr_ls.h"

#if defined (HYPRE_MIXED_PRECISION)

HYPRE_Int hypre_ADSComputePi_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *G,
                               hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z, hypre_ParCSRMatrix *PiNDx,
                               hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_ADSComputePi_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *G,
                               hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z, hypre_ParCSRMatrix *PiNDx,
                               hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_ADSComputePi_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C, hypre_ParCSRMatrix *G,
                               hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z, hypre_ParCSRMatrix *PiNDx,
                               hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_ADSComputePixyz_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C,
                                  hypre_ParCSRMatrix *G, hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z,
                                  hypre_ParCSRMatrix *PiNDx, hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz,
                                  hypre_ParCSRMatrix **Pix_ptr, hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_ADSComputePixyz_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C,
                                  hypre_ParCSRMatrix *G, hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z,
                                  hypre_ParCSRMatrix *PiNDx, hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz,
                                  hypre_ParCSRMatrix **Pix_ptr, hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_ADSComputePixyz_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *C,
                                  hypre_ParCSRMatrix *G, hypre_ParVector *x, hypre_ParVector *y, hypre_ParVector *z,
                                  hypre_ParCSRMatrix *PiNDx, hypre_ParCSRMatrix *PiNDy, hypre_ParCSRMatrix *PiNDz,
                                  hypre_ParCSRMatrix **Pix_ptr, hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
void *hypre_ADSCreate_flt  ( void );
void *hypre_ADSCreate_dbl  ( void );
void *hypre_ADSCreate_long_dbl  ( void );
HYPRE_Int hypre_ADSDestroy_flt  ( void *solver );
HYPRE_Int hypre_ADSDestroy_dbl  ( void *solver );
HYPRE_Int hypre_ADSDestroy_long_dbl  ( void *solver );
HYPRE_Int hypre_ADSGetFinalRelativeResidualNorm_flt  ( void *solver, hypre_float *rel_resid_norm );
HYPRE_Int hypre_ADSGetFinalRelativeResidualNorm_dbl  ( void *solver, hypre_double *rel_resid_norm );
HYPRE_Int hypre_ADSGetFinalRelativeResidualNorm_long_dbl  ( void *solver, hypre_long_double *rel_resid_norm );
HYPRE_Int hypre_ADSGetNumIterations_flt  ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ADSGetNumIterations_dbl  ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ADSGetNumIterations_long_dbl  ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ADSSetAMGOptions_flt  ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                   HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, hypre_float B_Pi_theta,
                                   HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_ADSSetAMGOptions_dbl  ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                   HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, hypre_double B_Pi_theta,
                                   HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_ADSSetAMGOptions_long_dbl  ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                   HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, hypre_long_double B_Pi_theta,
                                   HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_ADSSetAMSOptions_flt  ( void *solver, HYPRE_Int B_C_cycle_type,
                                   HYPRE_Int B_C_coarsen_type, HYPRE_Int B_C_agg_levels, HYPRE_Int B_C_relax_type,
                                   hypre_float B_C_theta, HYPRE_Int B_C_interp_type, HYPRE_Int B_C_Pmax );
HYPRE_Int hypre_ADSSetAMSOptions_dbl  ( void *solver, HYPRE_Int B_C_cycle_type,
                                   HYPRE_Int B_C_coarsen_type, HYPRE_Int B_C_agg_levels, HYPRE_Int B_C_relax_type,
                                   hypre_double B_C_theta, HYPRE_Int B_C_interp_type, HYPRE_Int B_C_Pmax );
HYPRE_Int hypre_ADSSetAMSOptions_long_dbl  ( void *solver, HYPRE_Int B_C_cycle_type,
                                   HYPRE_Int B_C_coarsen_type, HYPRE_Int B_C_agg_levels, HYPRE_Int B_C_relax_type,
                                   hypre_long_double B_C_theta, HYPRE_Int B_C_interp_type, HYPRE_Int B_C_Pmax );
HYPRE_Int hypre_ADSSetChebySmoothingOptions_flt  ( void *solver, HYPRE_Int A_cheby_order,
                                              hypre_float A_cheby_fraction );
HYPRE_Int hypre_ADSSetChebySmoothingOptions_dbl  ( void *solver, HYPRE_Int A_cheby_order,
                                              hypre_double A_cheby_fraction );
HYPRE_Int hypre_ADSSetChebySmoothingOptions_long_dbl  ( void *solver, HYPRE_Int A_cheby_order,
                                              hypre_long_double A_cheby_fraction );
HYPRE_Int hypre_ADSSetCoordinateVectors_flt  ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_ADSSetCoordinateVectors_dbl  ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_ADSSetCoordinateVectors_long_dbl  ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_ADSSetCycleType_flt  ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_ADSSetCycleType_dbl  ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_ADSSetCycleType_long_dbl  ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_ADSSetDiscreteCurl_flt  ( void *solver, hypre_ParCSRMatrix *C );
HYPRE_Int hypre_ADSSetDiscreteCurl_dbl  ( void *solver, hypre_ParCSRMatrix *C );
HYPRE_Int hypre_ADSSetDiscreteCurl_long_dbl  ( void *solver, hypre_ParCSRMatrix *C );
HYPRE_Int hypre_ADSSetDiscreteGradient_flt  ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_ADSSetDiscreteGradient_dbl  ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_ADSSetDiscreteGradient_long_dbl  ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_ADSSetInterpolations_flt  ( void *solver, hypre_ParCSRMatrix *RT_Pi,
                                       hypre_ParCSRMatrix *RT_Pix, hypre_ParCSRMatrix *RT_Piy,
                                       hypre_ParCSRMatrix *RT_Piz, hypre_ParCSRMatrix *ND_Pi,
                                       hypre_ParCSRMatrix *ND_Pix, hypre_ParCSRMatrix *ND_Piy,
                                       hypre_ParCSRMatrix *ND_Piz );
HYPRE_Int hypre_ADSSetInterpolations_dbl  ( void *solver, hypre_ParCSRMatrix *RT_Pi,
                                       hypre_ParCSRMatrix *RT_Pix, hypre_ParCSRMatrix *RT_Piy,
                                       hypre_ParCSRMatrix *RT_Piz, hypre_ParCSRMatrix *ND_Pi,
                                       hypre_ParCSRMatrix *ND_Pix, hypre_ParCSRMatrix *ND_Piy,
                                       hypre_ParCSRMatrix *ND_Piz );
HYPRE_Int hypre_ADSSetInterpolations_long_dbl  ( void *solver, hypre_ParCSRMatrix *RT_Pi,
                                       hypre_ParCSRMatrix *RT_Pix, hypre_ParCSRMatrix *RT_Piy,
                                       hypre_ParCSRMatrix *RT_Piz, hypre_ParCSRMatrix *ND_Pi,
                                       hypre_ParCSRMatrix *ND_Pix, hypre_ParCSRMatrix *ND_Piy,
                                       hypre_ParCSRMatrix *ND_Piz );
HYPRE_Int hypre_ADSSetMaxIter_flt  ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_ADSSetMaxIter_dbl  ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_ADSSetMaxIter_long_dbl  ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_ADSSetPrintLevel_flt  ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_ADSSetPrintLevel_dbl  ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_ADSSetPrintLevel_long_dbl  ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_ADSSetSmoothingOptions_flt  ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, hypre_float A_relax_weight, hypre_float A_omega );
HYPRE_Int hypre_ADSSetSmoothingOptions_dbl  ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, hypre_double A_relax_weight, hypre_double A_omega );
HYPRE_Int hypre_ADSSetSmoothingOptions_long_dbl  ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, hypre_long_double A_relax_weight, hypre_long_double A_omega );
HYPRE_Int hypre_ADSSetTol_flt  ( void *solver, hypre_float tol );
HYPRE_Int hypre_ADSSetTol_dbl  ( void *solver, hypre_double tol );
HYPRE_Int hypre_ADSSetTol_long_dbl  ( void *solver, hypre_long_double tol );
HYPRE_Int hypre_ADSSetup_flt  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ADSSetup_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ADSSetup_long_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ADSSolve_flt  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ADSSolve_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_ADSSolve_long_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
void *hypre_AMECreate_flt  ( void );
void *hypre_AMECreate_dbl  ( void );
void *hypre_AMECreate_long_dbl  ( void );
HYPRE_Int hypre_AMEDestroy_flt  ( void *esolver );
HYPRE_Int hypre_AMEDestroy_dbl  ( void *esolver );
HYPRE_Int hypre_AMEDestroy_long_dbl  ( void *esolver );
HYPRE_Int hypre_AMEDiscrDivFreeComponent_flt  ( void *esolver, hypre_ParVector *b );
HYPRE_Int hypre_AMEDiscrDivFreeComponent_dbl  ( void *esolver, hypre_ParVector *b );
HYPRE_Int hypre_AMEDiscrDivFreeComponent_long_dbl  ( void *esolver, hypre_ParVector *b );
HYPRE_Int hypre_AMEGetEigenvalues_flt  ( void *esolver, hypre_float **eigenvalues_ptr );
HYPRE_Int hypre_AMEGetEigenvalues_dbl  ( void *esolver, hypre_double **eigenvalues_ptr );
HYPRE_Int hypre_AMEGetEigenvalues_long_dbl  ( void *esolver, hypre_long_double **eigenvalues_ptr );
HYPRE_Int hypre_AMEGetEigenvectors_flt  ( void *esolver, HYPRE_ParVector **eigenvectors_ptr );
HYPRE_Int hypre_AMEGetEigenvectors_dbl  ( void *esolver, HYPRE_ParVector **eigenvectors_ptr );
HYPRE_Int hypre_AMEGetEigenvectors_long_dbl  ( void *esolver, HYPRE_ParVector **eigenvectors_ptr );
void hypre_AMEMultiOperatorA_flt  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorA_dbl  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorA_long_dbl  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorB_flt  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorB_dbl  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorB_long_dbl  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorM_flt  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorM_dbl  ( void *data, void *x, void *y );
void hypre_AMEMultiOperatorM_long_dbl  ( void *data, void *x, void *y );
void hypre_AMEOperatorA_flt  ( void *data, void *x, void *y );
void hypre_AMEOperatorA_dbl  ( void *data, void *x, void *y );
void hypre_AMEOperatorA_long_dbl  ( void *data, void *x, void *y );
void hypre_AMEOperatorB_flt  ( void *data, void *x, void *y );
void hypre_AMEOperatorB_dbl  ( void *data, void *x, void *y );
void hypre_AMEOperatorB_long_dbl  ( void *data, void *x, void *y );
void hypre_AMEOperatorM_flt  ( void *data, void *x, void *y );
void hypre_AMEOperatorM_dbl  ( void *data, void *x, void *y );
void hypre_AMEOperatorM_long_dbl  ( void *data, void *x, void *y );
HYPRE_Int hypre_AMESetAMSSolver_flt  ( void *esolver, void *ams_solver );
HYPRE_Int hypre_AMESetAMSSolver_dbl  ( void *esolver, void *ams_solver );
HYPRE_Int hypre_AMESetAMSSolver_long_dbl  ( void *esolver, void *ams_solver );
HYPRE_Int hypre_AMESetBlockSize_flt  ( void *esolver, HYPRE_Int block_size );
HYPRE_Int hypre_AMESetBlockSize_dbl  ( void *esolver, HYPRE_Int block_size );
HYPRE_Int hypre_AMESetBlockSize_long_dbl  ( void *esolver, HYPRE_Int block_size );
HYPRE_Int hypre_AMESetMassMatrix_flt  ( void *esolver, hypre_ParCSRMatrix *M );
HYPRE_Int hypre_AMESetMassMatrix_dbl  ( void *esolver, hypre_ParCSRMatrix *M );
HYPRE_Int hypre_AMESetMassMatrix_long_dbl  ( void *esolver, hypre_ParCSRMatrix *M );
HYPRE_Int hypre_AMESetMaxIter_flt  ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetMaxIter_dbl  ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetMaxIter_long_dbl  ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetMaxPCGIter_flt  ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetMaxPCGIter_dbl  ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetMaxPCGIter_long_dbl  ( void *esolver, HYPRE_Int maxit );
HYPRE_Int hypre_AMESetPrintLevel_flt  ( void *esolver, HYPRE_Int print_level );
HYPRE_Int hypre_AMESetPrintLevel_dbl  ( void *esolver, HYPRE_Int print_level );
HYPRE_Int hypre_AMESetPrintLevel_long_dbl  ( void *esolver, HYPRE_Int print_level );
HYPRE_Int hypre_AMESetRTol_flt  ( void *esolver, hypre_float tol );
HYPRE_Int hypre_AMESetRTol_dbl  ( void *esolver, hypre_double tol );
HYPRE_Int hypre_AMESetRTol_long_dbl  ( void *esolver, hypre_long_double tol );
HYPRE_Int hypre_AMESetTol_flt  ( void *esolver, hypre_float tol );
HYPRE_Int hypre_AMESetTol_dbl  ( void *esolver, hypre_double tol );
HYPRE_Int hypre_AMESetTol_long_dbl  ( void *esolver, hypre_long_double tol );
HYPRE_Int hypre_AMESetup_flt  ( void *esolver );
HYPRE_Int hypre_AMESetup_dbl  ( void *esolver );
HYPRE_Int hypre_AMESetup_long_dbl  ( void *esolver );
HYPRE_Int hypre_AMESolve_flt  ( void *esolver );
HYPRE_Int hypre_AMESolve_dbl  ( void *esolver );
HYPRE_Int hypre_AMESolve_long_dbl  ( void *esolver );
void *hypre_AMGHybridCreate_flt  ( void );
void *hypre_AMGHybridCreate_dbl  ( void );
void *hypre_AMGHybridCreate_long_dbl  ( void );
HYPRE_Int hypre_AMGHybridDestroy_flt  ( void *AMGhybrid_vdata );
HYPRE_Int hypre_AMGHybridDestroy_dbl  ( void *AMGhybrid_vdata );
HYPRE_Int hypre_AMGHybridDestroy_long_dbl  ( void *AMGhybrid_vdata );
HYPRE_Int hypre_AMGHybridGetDSCGNumIterations_flt  ( void *AMGhybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_AMGHybridGetDSCGNumIterations_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_AMGHybridGetDSCGNumIterations_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_AMGHybridGetFinalRelativeResidualNorm_flt  ( void *AMGhybrid_vdata,
                                                        hypre_float *final_rel_res_norm );
HYPRE_Int hypre_AMGHybridGetFinalRelativeResidualNorm_dbl  ( void *AMGhybrid_vdata,
                                                        hypre_double *final_rel_res_norm );
HYPRE_Int hypre_AMGHybridGetFinalRelativeResidualNorm_long_dbl  ( void *AMGhybrid_vdata,
                                                        hypre_long_double *final_rel_res_norm );
HYPRE_Int hypre_AMGHybridGetNumIterations_flt  ( void *AMGhybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_AMGHybridGetNumIterations_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_AMGHybridGetNumIterations_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *num_its );
HYPRE_Int hypre_AMGHybridGetPCGNumIterations_flt  ( void *AMGhybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_AMGHybridGetPCGNumIterations_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_AMGHybridGetPCGNumIterations_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_AMGHybridGetRecomputeResidual_flt  ( void *AMGhybrid_vdata,
                                                HYPRE_Int *recompute_residual );
HYPRE_Int hypre_AMGHybridGetRecomputeResidual_dbl  ( void *AMGhybrid_vdata,
                                                HYPRE_Int *recompute_residual );
HYPRE_Int hypre_AMGHybridGetRecomputeResidual_long_dbl  ( void *AMGhybrid_vdata,
                                                HYPRE_Int *recompute_residual );
HYPRE_Int hypre_AMGHybridGetRecomputeResidualP_flt  ( void *AMGhybrid_vdata,
                                                 HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_AMGHybridGetRecomputeResidualP_dbl  ( void *AMGhybrid_vdata,
                                                 HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_AMGHybridGetRecomputeResidualP_long_dbl  ( void *AMGhybrid_vdata,
                                                 HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_AMGHybridGetSetupSolveTime_flt ( void *AMGhybrid_vdata, hypre_float *time );
HYPRE_Int hypre_AMGHybridGetSetupSolveTime_dbl ( void *AMGhybrid_vdata, hypre_double *time );
HYPRE_Int hypre_AMGHybridGetSetupSolveTime_long_dbl ( void *AMGhybrid_vdata, hypre_long_double *time );
HYPRE_Int hypre_AMGHybridSetAbsoluteTol_flt  ( void *AMGhybrid_vdata, hypre_float a_tol );
HYPRE_Int hypre_AMGHybridSetAbsoluteTol_dbl  ( void *AMGhybrid_vdata, hypre_double a_tol );
HYPRE_Int hypre_AMGHybridSetAbsoluteTol_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double a_tol );
HYPRE_Int hypre_AMGHybridSetAggInterpType_flt  ( void *AMGhybrid_vdata, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_AMGHybridSetAggInterpType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_AMGHybridSetAggInterpType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_AMGHybridSetAggNumLevels_flt  ( void *AMGhybrid_vdata, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_AMGHybridSetAggNumLevels_dbl  ( void *AMGhybrid_vdata, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_AMGHybridSetAggNumLevels_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_AMGHybridSetCoarsenType_flt  ( void *AMGhybrid_vdata, HYPRE_Int coarsen_type );
HYPRE_Int hypre_AMGHybridSetCoarsenType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int coarsen_type );
HYPRE_Int hypre_AMGHybridSetCoarsenType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int coarsen_type );
HYPRE_Int hypre_AMGHybridSetConvergenceTol_flt  ( void *AMGhybrid_vdata, hypre_float cf_tol );
HYPRE_Int hypre_AMGHybridSetConvergenceTol_dbl  ( void *AMGhybrid_vdata, hypre_double cf_tol );
HYPRE_Int hypre_AMGHybridSetConvergenceTol_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_AMGHybridSetCycleNumSweeps_flt  ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetCycleNumSweeps_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetCycleNumSweeps_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetCycleRelaxType_flt  ( void *AMGhybrid_vdata, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetCycleRelaxType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetCycleRelaxType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int hypre_AMGHybridSetCycleType_flt  ( void *AMGhybrid_vdata, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMGHybridSetCycleType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMGHybridSetCycleType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMGHybridSetDofFunc_flt  ( void *AMGhybrid_vdata, HYPRE_Int *dof_func );
HYPRE_Int hypre_AMGHybridSetDofFunc_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *dof_func );
HYPRE_Int hypre_AMGHybridSetDofFunc_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *dof_func );
HYPRE_Int hypre_AMGHybridSetDSCGMaxIter_flt  ( void *AMGhybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_AMGHybridSetDSCGMaxIter_dbl  ( void *AMGhybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_AMGHybridSetDSCGMaxIter_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int dscg_max_its );
HYPRE_Int hypre_AMGHybridSetGridRelaxPoints_flt  ( void *AMGhybrid_vdata,
                                              HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_AMGHybridSetGridRelaxPoints_dbl  ( void *AMGhybrid_vdata,
                                              HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_AMGHybridSetGridRelaxPoints_long_dbl  ( void *AMGhybrid_vdata,
                                              HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_AMGHybridSetGridRelaxType_flt  ( void *AMGhybrid_vdata, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_AMGHybridSetGridRelaxType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_AMGHybridSetGridRelaxType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_AMGHybridSetInterpType_flt  ( void *AMGhybrid_vdata, HYPRE_Int interp_type );
HYPRE_Int hypre_AMGHybridSetInterpType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int interp_type );
HYPRE_Int hypre_AMGHybridSetInterpType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int interp_type );
HYPRE_Int hypre_AMGHybridSetKDim_flt  ( void *AMGhybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_AMGHybridSetKDim_dbl  ( void *AMGhybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_AMGHybridSetKDim_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_AMGHybridSetKeepTranspose_flt  ( void *AMGhybrid_vdata, HYPRE_Int keepT );
HYPRE_Int hypre_AMGHybridSetKeepTranspose_dbl  ( void *AMGhybrid_vdata, HYPRE_Int keepT );
HYPRE_Int hypre_AMGHybridSetKeepTranspose_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int keepT );
HYPRE_Int hypre_AMGHybridSetLevelOuterWt_flt  ( void *AMGhybrid_vdata, hypre_float outer_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetLevelOuterWt_dbl  ( void *AMGhybrid_vdata, hypre_double outer_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetLevelOuterWt_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double outer_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetLevelRelaxWt_flt  ( void *AMGhybrid_vdata, hypre_float relax_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetLevelRelaxWt_dbl  ( void *AMGhybrid_vdata, hypre_double relax_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetLevelRelaxWt_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double relax_wt,
                                           HYPRE_Int level );
HYPRE_Int hypre_AMGHybridSetLogging_flt  ( void *AMGhybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_AMGHybridSetLogging_dbl  ( void *AMGhybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_AMGHybridSetLogging_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int logging );
HYPRE_Int hypre_AMGHybridSetMaxCoarseSize_flt  ( void *AMGhybrid_vdata, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_AMGHybridSetMaxCoarseSize_dbl  ( void *AMGhybrid_vdata, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_AMGHybridSetMaxCoarseSize_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_AMGHybridSetMaxLevels_flt  ( void *AMGhybrid_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_AMGHybridSetMaxLevels_dbl  ( void *AMGhybrid_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_AMGHybridSetMaxLevels_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int max_levels );
HYPRE_Int hypre_AMGHybridSetMaxRowSum_flt  ( void *AMGhybrid_vdata, hypre_float max_row_sum );
HYPRE_Int hypre_AMGHybridSetMaxRowSum_dbl  ( void *AMGhybrid_vdata, hypre_double max_row_sum );
HYPRE_Int hypre_AMGHybridSetMaxRowSum_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double max_row_sum );
HYPRE_Int hypre_AMGHybridSetMeasureType_flt  ( void *AMGhybrid_vdata, HYPRE_Int measure_type );
HYPRE_Int hypre_AMGHybridSetMeasureType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int measure_type );
HYPRE_Int hypre_AMGHybridSetMeasureType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int measure_type );
HYPRE_Int hypre_AMGHybridSetMinCoarseSize_flt  ( void *AMGhybrid_vdata, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_AMGHybridSetMinCoarseSize_dbl  ( void *AMGhybrid_vdata, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_AMGHybridSetMinCoarseSize_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_AMGHybridSetNodal_flt  ( void *AMGhybrid_vdata, HYPRE_Int nodal );
HYPRE_Int hypre_AMGHybridSetNodal_dbl  ( void *AMGhybrid_vdata, HYPRE_Int nodal );
HYPRE_Int hypre_AMGHybridSetNodal_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int nodal );
HYPRE_Int hypre_AMGHybridSetNonGalerkinTol_flt  ( void *AMGhybrid_vdata, HYPRE_Int nongalerk_num_tol,
                                             hypre_float *nongalerkin_tol );
HYPRE_Int hypre_AMGHybridSetNonGalerkinTol_dbl  ( void *AMGhybrid_vdata, HYPRE_Int nongalerk_num_tol,
                                             hypre_double *nongalerkin_tol );
HYPRE_Int hypre_AMGHybridSetNonGalerkinTol_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int nongalerk_num_tol,
                                             hypre_long_double *nongalerkin_tol );
HYPRE_Int hypre_AMGHybridSetNumFunctions_flt  ( void *AMGhybrid_vdata, HYPRE_Int num_functions );
HYPRE_Int hypre_AMGHybridSetNumFunctions_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_functions );
HYPRE_Int hypre_AMGHybridSetNumFunctions_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_functions );
HYPRE_Int hypre_AMGHybridSetNumGridSweeps_flt  ( void *AMGhybrid_vdata, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_AMGHybridSetNumGridSweeps_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_AMGHybridSetNumGridSweeps_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_AMGHybridSetNumPaths_flt  ( void *AMGhybrid_vdata, HYPRE_Int num_paths );
HYPRE_Int hypre_AMGHybridSetNumPaths_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_paths );
HYPRE_Int hypre_AMGHybridSetNumPaths_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_paths );
HYPRE_Int hypre_AMGHybridSetNumSweeps_flt  ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps );
HYPRE_Int hypre_AMGHybridSetNumSweeps_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps );
HYPRE_Int hypre_AMGHybridSetNumSweeps_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int num_sweeps );
HYPRE_Int hypre_AMGHybridSetOmega_flt  ( void *AMGhybrid_vdata, hypre_float *omega );
HYPRE_Int hypre_AMGHybridSetOmega_dbl  ( void *AMGhybrid_vdata, hypre_double *omega );
HYPRE_Int hypre_AMGHybridSetOmega_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double *omega );
HYPRE_Int hypre_AMGHybridSetOuterWt_flt  ( void *AMGhybrid_vdata, hypre_float outer_wt );
HYPRE_Int hypre_AMGHybridSetOuterWt_dbl  ( void *AMGhybrid_vdata, hypre_double outer_wt );
HYPRE_Int hypre_AMGHybridSetOuterWt_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double outer_wt );
HYPRE_Int hypre_AMGHybridSetPCGMaxIter_flt  ( void *AMGhybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_AMGHybridSetPCGMaxIter_dbl  ( void *AMGhybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_AMGHybridSetPCGMaxIter_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int pcg_max_its );
HYPRE_Int hypre_AMGHybridSetPMaxElmts_flt  ( void *AMGhybrid_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_AMGHybridSetPMaxElmts_dbl  ( void *AMGhybrid_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_AMGHybridSetPMaxElmts_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_AMGHybridSetPrecond_flt  ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                       void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_AMGHybridSetPrecond_dbl  ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                       void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_AMGHybridSetPrecond_long_dbl  ( void *pcg_vdata, HYPRE_Int (*pcg_precond_solve )(void*, void*,
                                                                                       void*, void*), HYPRE_Int (*pcg_precond_setup )(void*, void*, void*, void*), void *pcg_precond );
HYPRE_Int hypre_AMGHybridSetPrintLevel_flt  ( void *AMGhybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_AMGHybridSetPrintLevel_dbl  ( void *AMGhybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_AMGHybridSetPrintLevel_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_AMGHybridSetRecomputeResidual_flt  ( void *AMGhybrid_vdata,
                                                HYPRE_Int recompute_residual );
HYPRE_Int hypre_AMGHybridSetRecomputeResidual_dbl  ( void *AMGhybrid_vdata,
                                                HYPRE_Int recompute_residual );
HYPRE_Int hypre_AMGHybridSetRecomputeResidual_long_dbl  ( void *AMGhybrid_vdata,
                                                HYPRE_Int recompute_residual );
HYPRE_Int hypre_AMGHybridSetRecomputeResidualP_flt  ( void *AMGhybrid_vdata,
                                                 HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_AMGHybridSetRecomputeResidualP_dbl  ( void *AMGhybrid_vdata,
                                                 HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_AMGHybridSetRecomputeResidualP_long_dbl  ( void *AMGhybrid_vdata,
                                                 HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_AMGHybridSetRelaxOrder_flt  ( void *AMGhybrid_vdata, HYPRE_Int relax_order );
HYPRE_Int hypre_AMGHybridSetRelaxOrder_dbl  ( void *AMGhybrid_vdata, HYPRE_Int relax_order );
HYPRE_Int hypre_AMGHybridSetRelaxOrder_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int relax_order );
HYPRE_Int hypre_AMGHybridSetRelaxType_flt  ( void *AMGhybrid_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_AMGHybridSetRelaxType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_AMGHybridSetRelaxType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_AMGHybridSetRelaxWeight_flt  ( void *AMGhybrid_vdata, hypre_float *relax_weight );
HYPRE_Int hypre_AMGHybridSetRelaxWeight_dbl  ( void *AMGhybrid_vdata, hypre_double *relax_weight );
HYPRE_Int hypre_AMGHybridSetRelaxWeight_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double *relax_weight );
HYPRE_Int hypre_AMGHybridSetRelaxWt_flt  ( void *AMGhybrid_vdata, hypre_float relax_wt );
HYPRE_Int hypre_AMGHybridSetRelaxWt_dbl  ( void *AMGhybrid_vdata, hypre_double relax_wt );
HYPRE_Int hypre_AMGHybridSetRelaxWt_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double relax_wt );
HYPRE_Int hypre_AMGHybridSetRelChange_flt  ( void *AMGhybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_AMGHybridSetRelChange_dbl  ( void *AMGhybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_AMGHybridSetRelChange_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_AMGHybridSetSeqThreshold_flt  ( void *AMGhybrid_vdata, HYPRE_Int seq_threshold );
HYPRE_Int hypre_AMGHybridSetSeqThreshold_dbl  ( void *AMGhybrid_vdata, HYPRE_Int seq_threshold );
HYPRE_Int hypre_AMGHybridSetSeqThreshold_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int seq_threshold );
HYPRE_Int hypre_AMGHybridSetSetupType_flt  ( void *AMGhybrid_vdata, HYPRE_Int setup_type );
HYPRE_Int hypre_AMGHybridSetSetupType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int setup_type );
HYPRE_Int hypre_AMGHybridSetSetupType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int setup_type );
HYPRE_Int hypre_AMGHybridSetSolverType_flt  ( void *AMGhybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_AMGHybridSetSolverType_dbl  ( void *AMGhybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_AMGHybridSetSolverType_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int solver_type );
HYPRE_Int hypre_AMGHybridSetStopCrit_flt  ( void *AMGhybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_AMGHybridSetStopCrit_dbl  ( void *AMGhybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_AMGHybridSetStopCrit_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_AMGHybridSetStrongThreshold_flt  ( void *AMGhybrid_vdata, hypre_float strong_threshold );
HYPRE_Int hypre_AMGHybridSetStrongThreshold_dbl  ( void *AMGhybrid_vdata, hypre_double strong_threshold );
HYPRE_Int hypre_AMGHybridSetStrongThreshold_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double strong_threshold );
HYPRE_Int hypre_AMGHybridSetTol_flt  ( void *AMGhybrid_vdata, hypre_float tol );
HYPRE_Int hypre_AMGHybridSetTol_dbl  ( void *AMGhybrid_vdata, hypre_double tol );
HYPRE_Int hypre_AMGHybridSetTol_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double tol );
HYPRE_Int hypre_AMGHybridSetTruncFactor_flt  ( void *AMGhybrid_vdata, hypre_float trunc_factor );
HYPRE_Int hypre_AMGHybridSetTruncFactor_dbl  ( void *AMGhybrid_vdata, hypre_double trunc_factor );
HYPRE_Int hypre_AMGHybridSetTruncFactor_long_dbl  ( void *AMGhybrid_vdata, hypre_long_double trunc_factor );
HYPRE_Int hypre_AMGHybridSetTwoNorm_flt  ( void *AMGhybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_AMGHybridSetTwoNorm_dbl  ( void *AMGhybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_AMGHybridSetTwoNorm_long_dbl  ( void *AMGhybrid_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_AMGHybridSetup_flt  ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
HYPRE_Int hypre_AMGHybridSetup_dbl  ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
HYPRE_Int hypre_AMGHybridSetup_long_dbl  ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
HYPRE_Int hypre_AMGHybridSolve_flt  ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
HYPRE_Int hypre_AMGHybridSolve_dbl  ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
HYPRE_Int hypre_AMGHybridSolve_long_dbl  ( void *AMGhybrid_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                 hypre_ParVector *x );
HYPRE_Int hypre_AMSComputeGPi_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **GPi_ptr );
HYPRE_Int hypre_AMSComputeGPi_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **GPi_ptr );
HYPRE_Int hypre_AMSComputeGPi_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **GPi_ptr );
HYPRE_Int hypre_AMSComputePi_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                               hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_AMSComputePi_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                               hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_AMSComputePi_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                               hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_AMSComputePixyz_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                  hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pix_ptr,
                                  hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_AMSComputePixyz_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                  hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pix_ptr,
                                  hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_AMSComputePixyz_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *G, hypre_ParVector *Gx,
                                  hypre_ParVector *Gy, hypre_ParVector *Gz, HYPRE_Int dim, hypre_ParCSRMatrix **Pix_ptr,
                                  hypre_ParCSRMatrix **Piy_ptr, hypre_ParCSRMatrix **Piz_ptr );
HYPRE_Int hypre_AMSConstructDiscreteGradient_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, hypre_ParCSRMatrix **G_ptr );
HYPRE_Int hypre_AMSConstructDiscreteGradient_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, hypre_ParCSRMatrix **G_ptr );
HYPRE_Int hypre_AMSConstructDiscreteGradient_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, hypre_ParCSRMatrix **G_ptr );
void *hypre_AMSCreate_flt  ( void );
void *hypre_AMSCreate_dbl  ( void );
void *hypre_AMSCreate_long_dbl  ( void );
HYPRE_Int hypre_AMSDestroy_flt  ( void *solver );
HYPRE_Int hypre_AMSDestroy_dbl  ( void *solver );
HYPRE_Int hypre_AMSDestroy_long_dbl  ( void *solver );
HYPRE_Int hypre_AMSFEIDestroy_flt  ( void *solver );
HYPRE_Int hypre_AMSFEIDestroy_dbl  ( void *solver );
HYPRE_Int hypre_AMSFEIDestroy_long_dbl  ( void *solver );
HYPRE_Int hypre_AMSFEISetup_flt  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                              hypre_ParVector *x, HYPRE_Int num_vert, HYPRE_Int num_local_vert, HYPRE_BigInt *vert_number,
                              hypre_float *vert_coord, HYPRE_Int num_edges, HYPRE_BigInt *edge_vertex );
HYPRE_Int hypre_AMSFEISetup_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                              hypre_ParVector *x, HYPRE_Int num_vert, HYPRE_Int num_local_vert, HYPRE_BigInt *vert_number,
                              hypre_double *vert_coord, HYPRE_Int num_edges, HYPRE_BigInt *edge_vertex );
HYPRE_Int hypre_AMSFEISetup_long_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                              hypre_ParVector *x, HYPRE_Int num_vert, HYPRE_Int num_local_vert, HYPRE_BigInt *vert_number,
                              hypre_long_double *vert_coord, HYPRE_Int num_edges, HYPRE_BigInt *edge_vertex );
HYPRE_Int hypre_AMSGetFinalRelativeResidualNorm_flt  ( void *solver, hypre_float *rel_resid_norm );
HYPRE_Int hypre_AMSGetFinalRelativeResidualNorm_dbl  ( void *solver, hypre_double *rel_resid_norm );
HYPRE_Int hypre_AMSGetFinalRelativeResidualNorm_long_dbl  ( void *solver, hypre_long_double *rel_resid_norm );
HYPRE_Int hypre_AMSGetNumIterations_flt  ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_AMSGetNumIterations_dbl  ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_AMSGetNumIterations_long_dbl  ( void *solver, HYPRE_Int *num_iterations );
HYPRE_Int hypre_AMSProjectOutGradients_flt  ( void *solver, hypre_ParVector *x );
HYPRE_Int hypre_AMSProjectOutGradients_dbl  ( void *solver, hypre_ParVector *x );
HYPRE_Int hypre_AMSProjectOutGradients_long_dbl  ( void *solver, hypre_ParVector *x );
HYPRE_Int hypre_AMSSetAlphaAMGCoarseRelaxType_flt  ( void *solver, HYPRE_Int B_Pi_coarse_relax_type );
HYPRE_Int hypre_AMSSetAlphaAMGCoarseRelaxType_dbl  ( void *solver, HYPRE_Int B_Pi_coarse_relax_type );
HYPRE_Int hypre_AMSSetAlphaAMGCoarseRelaxType_long_dbl  ( void *solver, HYPRE_Int B_Pi_coarse_relax_type );
HYPRE_Int hypre_AMSSetAlphaAMGOptions_flt  ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                        HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, hypre_float B_Pi_theta,
                                        HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_AMSSetAlphaAMGOptions_dbl  ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                        HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, hypre_double B_Pi_theta,
                                        HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_AMSSetAlphaAMGOptions_long_dbl  ( void *solver, HYPRE_Int B_Pi_coarsen_type,
                                        HYPRE_Int B_Pi_agg_levels, HYPRE_Int B_Pi_relax_type, hypre_long_double B_Pi_theta,
                                        HYPRE_Int B_Pi_interp_type, HYPRE_Int B_Pi_Pmax );
HYPRE_Int hypre_AMSSetAlphaPoissonMatrix_flt  ( void *solver, hypre_ParCSRMatrix *A_Pi );
HYPRE_Int hypre_AMSSetAlphaPoissonMatrix_dbl  ( void *solver, hypre_ParCSRMatrix *A_Pi );
HYPRE_Int hypre_AMSSetAlphaPoissonMatrix_long_dbl  ( void *solver, hypre_ParCSRMatrix *A_Pi );
HYPRE_Int hypre_AMSSetBetaAMGCoarseRelaxType_flt  ( void *solver, HYPRE_Int B_G_coarse_relax_type );
HYPRE_Int hypre_AMSSetBetaAMGCoarseRelaxType_dbl  ( void *solver, HYPRE_Int B_G_coarse_relax_type );
HYPRE_Int hypre_AMSSetBetaAMGCoarseRelaxType_long_dbl  ( void *solver, HYPRE_Int B_G_coarse_relax_type );
HYPRE_Int hypre_AMSSetBetaAMGOptions_flt  ( void *solver, HYPRE_Int B_G_coarsen_type,
                                       HYPRE_Int B_G_agg_levels, HYPRE_Int B_G_relax_type, hypre_float B_G_theta, HYPRE_Int B_G_interp_type,
                                       HYPRE_Int B_G_Pmax );
HYPRE_Int hypre_AMSSetBetaAMGOptions_dbl  ( void *solver, HYPRE_Int B_G_coarsen_type,
                                       HYPRE_Int B_G_agg_levels, HYPRE_Int B_G_relax_type, hypre_double B_G_theta, HYPRE_Int B_G_interp_type,
                                       HYPRE_Int B_G_Pmax );
HYPRE_Int hypre_AMSSetBetaAMGOptions_long_dbl  ( void *solver, HYPRE_Int B_G_coarsen_type,
                                       HYPRE_Int B_G_agg_levels, HYPRE_Int B_G_relax_type, hypre_long_double B_G_theta, HYPRE_Int B_G_interp_type,
                                       HYPRE_Int B_G_Pmax );
HYPRE_Int hypre_AMSSetBetaPoissonMatrix_flt  ( void *solver, hypre_ParCSRMatrix *A_G );
HYPRE_Int hypre_AMSSetBetaPoissonMatrix_dbl  ( void *solver, hypre_ParCSRMatrix *A_G );
HYPRE_Int hypre_AMSSetBetaPoissonMatrix_long_dbl  ( void *solver, hypre_ParCSRMatrix *A_G );
HYPRE_Int hypre_AMSSetChebySmoothingOptions_flt  ( void *solver, HYPRE_Int A_cheby_order,
                                              hypre_float A_cheby_fraction );
HYPRE_Int hypre_AMSSetChebySmoothingOptions_dbl  ( void *solver, HYPRE_Int A_cheby_order,
                                              hypre_double A_cheby_fraction );
HYPRE_Int hypre_AMSSetChebySmoothingOptions_long_dbl  ( void *solver, HYPRE_Int A_cheby_order,
                                              hypre_long_double A_cheby_fraction );
HYPRE_Int hypre_AMSSetCoordinateVectors_flt  ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_AMSSetCoordinateVectors_dbl  ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_AMSSetCoordinateVectors_long_dbl  ( void *solver, hypre_ParVector *x, hypre_ParVector *y,
                                          hypre_ParVector *z );
HYPRE_Int hypre_AMSSetCycleType_flt  ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMSSetCycleType_dbl  ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMSSetCycleType_long_dbl  ( void *solver, HYPRE_Int cycle_type );
HYPRE_Int hypre_AMSSetDimension_flt  ( void *solver, HYPRE_Int dim );
HYPRE_Int hypre_AMSSetDimension_dbl  ( void *solver, HYPRE_Int dim );
HYPRE_Int hypre_AMSSetDimension_long_dbl  ( void *solver, HYPRE_Int dim );
HYPRE_Int hypre_AMSSetDiscreteGradient_flt  ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_AMSSetDiscreteGradient_dbl  ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_AMSSetDiscreteGradient_long_dbl  ( void *solver, hypre_ParCSRMatrix *G );
HYPRE_Int hypre_AMSSetEdgeConstantVectors_flt  ( void *solver, hypre_ParVector *Gx, hypre_ParVector *Gy,
                                            hypre_ParVector *Gz );
HYPRE_Int hypre_AMSSetEdgeConstantVectors_dbl  ( void *solver, hypre_ParVector *Gx, hypre_ParVector *Gy,
                                            hypre_ParVector *Gz );
HYPRE_Int hypre_AMSSetEdgeConstantVectors_long_dbl  ( void *solver, hypre_ParVector *Gx, hypre_ParVector *Gy,
                                            hypre_ParVector *Gz );
HYPRE_Int hypre_AMSSetInteriorNodes_flt  ( void *solver, hypre_ParVector *interior_nodes );
HYPRE_Int hypre_AMSSetInteriorNodes_dbl  ( void *solver, hypre_ParVector *interior_nodes );
HYPRE_Int hypre_AMSSetInteriorNodes_long_dbl  ( void *solver, hypre_ParVector *interior_nodes );
HYPRE_Int hypre_AMSSetInterpolations_flt  ( void *solver, hypre_ParCSRMatrix *Pi,
                                       hypre_ParCSRMatrix *Pix, hypre_ParCSRMatrix *Piy, hypre_ParCSRMatrix *Piz );
HYPRE_Int hypre_AMSSetInterpolations_dbl  ( void *solver, hypre_ParCSRMatrix *Pi,
                                       hypre_ParCSRMatrix *Pix, hypre_ParCSRMatrix *Piy, hypre_ParCSRMatrix *Piz );
HYPRE_Int hypre_AMSSetInterpolations_long_dbl  ( void *solver, hypre_ParCSRMatrix *Pi,
                                       hypre_ParCSRMatrix *Pix, hypre_ParCSRMatrix *Piy, hypre_ParCSRMatrix *Piz );
HYPRE_Int hypre_AMSSetMaxIter_flt  ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_AMSSetMaxIter_dbl  ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_AMSSetMaxIter_long_dbl  ( void *solver, HYPRE_Int maxit );
HYPRE_Int hypre_AMSSetPrintLevel_flt  ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_AMSSetPrintLevel_dbl  ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_AMSSetPrintLevel_long_dbl  ( void *solver, HYPRE_Int print_level );
HYPRE_Int hypre_AMSSetProjectionFrequency_flt  ( void *solver, HYPRE_Int projection_frequency );
HYPRE_Int hypre_AMSSetProjectionFrequency_dbl  ( void *solver, HYPRE_Int projection_frequency );
HYPRE_Int hypre_AMSSetProjectionFrequency_long_dbl  ( void *solver, HYPRE_Int projection_frequency );
HYPRE_Int hypre_AMSSetSmoothingOptions_flt  ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, hypre_float A_relax_weight, hypre_float A_omega );
HYPRE_Int hypre_AMSSetSmoothingOptions_dbl  ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, hypre_double A_relax_weight, hypre_double A_omega );
HYPRE_Int hypre_AMSSetSmoothingOptions_long_dbl  ( void *solver, HYPRE_Int A_relax_type,
                                         HYPRE_Int A_relax_times, hypre_long_double A_relax_weight, hypre_long_double A_omega );
HYPRE_Int hypre_AMSSetTol_flt  ( void *solver, hypre_float tol );
HYPRE_Int hypre_AMSSetTol_dbl  ( void *solver, hypre_double tol );
HYPRE_Int hypre_AMSSetTol_long_dbl  ( void *solver, hypre_long_double tol );
HYPRE_Int hypre_AMSSetup_flt  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_AMSSetup_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_AMSSetup_long_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_AMSSolve_flt  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_AMSSolve_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_AMSSolve_long_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                           hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGBlockSolve_flt  ( void *B, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                      hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGBlockSolve_dbl  ( void *B, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                      hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGBlockSolve_long_dbl  ( void *B, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                      hypre_ParVector *x );
HYPRE_Int hypre_ParCSRComputeL1Norms_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                       HYPRE_Int *cf_marker, hypre_float **l1_norm_ptr );
HYPRE_Int hypre_ParCSRComputeL1Norms_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                       HYPRE_Int *cf_marker, hypre_double **l1_norm_ptr );
HYPRE_Int hypre_ParCSRComputeL1Norms_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                       HYPRE_Int *cf_marker, hypre_long_double **l1_norm_ptr );
HYPRE_Int hypre_ParCSRComputeL1NormsThreads_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                              HYPRE_Int num_threads, HYPRE_Int *cf_marker, hypre_float **l1_norm_ptr );
HYPRE_Int hypre_ParCSRComputeL1NormsThreads_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                              HYPRE_Int num_threads, HYPRE_Int *cf_marker, hypre_double **l1_norm_ptr );
HYPRE_Int hypre_ParCSRComputeL1NormsThreads_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int option,
                                              HYPRE_Int num_threads, HYPRE_Int *cf_marker, hypre_long_double **l1_norm_ptr );
HYPRE_Int hypre_ParCSRMatrixFixZeroRows_flt  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixFixZeroRows_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixFixZeroRows_long_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ParCSRMatrixSetDiagRows_flt  ( hypre_ParCSRMatrix *A, hypre_float d );
HYPRE_Int hypre_ParCSRMatrixSetDiagRows_dbl  ( hypre_ParCSRMatrix *A, hypre_double d );
HYPRE_Int hypre_ParCSRMatrixSetDiagRows_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double d );
HYPRE_Int hypre_ParCSRRelax_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int relax_type,
                              HYPRE_Int relax_times, hypre_float *l1_norms, hypre_float relax_weight, hypre_float omega,
                              hypre_float max_eig_est, hypre_float min_eig_est, HYPRE_Int cheby_order, hypre_float cheby_fraction,
                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *z );
HYPRE_Int hypre_ParCSRRelax_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int relax_type,
                              HYPRE_Int relax_times, hypre_double *l1_norms, hypre_double relax_weight, hypre_double omega,
                              hypre_double max_eig_est, hypre_double min_eig_est, HYPRE_Int cheby_order, hypre_double cheby_fraction,
                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *z );
HYPRE_Int hypre_ParCSRRelax_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int relax_type,
                              HYPRE_Int relax_times, hypre_long_double *l1_norms, hypre_long_double relax_weight, hypre_long_double omega,
                              hypre_long_double max_eig_est, hypre_long_double min_eig_est, HYPRE_Int cheby_order, hypre_long_double cheby_fraction,
                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *z );
HYPRE_Int hypre_ParCSRSubspacePrec_flt  ( hypre_ParCSRMatrix *A0, HYPRE_Int A0_relax_type,
                                     HYPRE_Int A0_relax_times, hypre_float *A0_l1_norms, hypre_float A0_relax_weight, hypre_float A0_omega,
                                     hypre_float A0_max_eig_est, hypre_float A0_min_eig_est, HYPRE_Int A0_cheby_order,
                                     hypre_float A0_cheby_fraction, hypre_ParCSRMatrix **A, HYPRE_Solver *B, HYPRE_PtrToSolverFcn *HB,
                                     hypre_ParCSRMatrix **P, hypre_ParVector **r, hypre_ParVector **g, hypre_ParVector *x,
                                     hypre_ParVector *y, hypre_ParVector *r0, hypre_ParVector *g0, char *cycle, hypre_ParVector *z );
HYPRE_Int hypre_ParCSRSubspacePrec_dbl  ( hypre_ParCSRMatrix *A0, HYPRE_Int A0_relax_type,
                                     HYPRE_Int A0_relax_times, hypre_double *A0_l1_norms, hypre_double A0_relax_weight, hypre_double A0_omega,
                                     hypre_double A0_max_eig_est, hypre_double A0_min_eig_est, HYPRE_Int A0_cheby_order,
                                     hypre_double A0_cheby_fraction, hypre_ParCSRMatrix **A, HYPRE_Solver *B, HYPRE_PtrToSolverFcn *HB,
                                     hypre_ParCSRMatrix **P, hypre_ParVector **r, hypre_ParVector **g, hypre_ParVector *x,
                                     hypre_ParVector *y, hypre_ParVector *r0, hypre_ParVector *g0, char *cycle, hypre_ParVector *z );
HYPRE_Int hypre_ParCSRSubspacePrec_long_dbl  ( hypre_ParCSRMatrix *A0, HYPRE_Int A0_relax_type,
                                     HYPRE_Int A0_relax_times, hypre_long_double *A0_l1_norms, hypre_long_double A0_relax_weight, hypre_long_double A0_omega,
                                     hypre_long_double A0_max_eig_est, hypre_long_double A0_min_eig_est, HYPRE_Int A0_cheby_order,
                                     hypre_long_double A0_cheby_fraction, hypre_ParCSRMatrix **A, HYPRE_Solver *B, HYPRE_PtrToSolverFcn *HB,
                                     hypre_ParCSRMatrix **P, hypre_ParVector **r, hypre_ParVector **g, hypre_ParVector *x,
                                     hypre_ParVector *y, hypre_ParVector *r0, hypre_ParVector *g0, char *cycle, hypre_ParVector *z );
HYPRE_Int hypre_ParVectorBlockGather_flt  ( hypre_ParVector *x, hypre_ParVector *x_ [3 ],
                                       HYPRE_Int dim );
HYPRE_Int hypre_ParVectorBlockGather_dbl  ( hypre_ParVector *x, hypre_ParVector *x_ [3 ],
                                       HYPRE_Int dim );
HYPRE_Int hypre_ParVectorBlockGather_long_dbl  ( hypre_ParVector *x, hypre_ParVector *x_ [3 ],
                                       HYPRE_Int dim );
HYPRE_Int hypre_ParVectorBlockSplit_flt  ( hypre_ParVector *x, hypre_ParVector *x_ [3 ], HYPRE_Int dim );
HYPRE_Int hypre_ParVectorBlockSplit_dbl  ( hypre_ParVector *x, hypre_ParVector *x_ [3 ], HYPRE_Int dim );
HYPRE_Int hypre_ParVectorBlockSplit_long_dbl  ( hypre_ParVector *x, hypre_ParVector *x_ [3 ], HYPRE_Int dim );
hypre_ParVector *hypre_ParVectorInDomainOf_flt  ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInDomainOf_dbl  ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInDomainOf_long_dbl  ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInRangeOf_flt  ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInRangeOf_dbl  ( hypre_ParCSRMatrix *A );
hypre_ParVector *hypre_ParVectorInRangeOf_long_dbl  ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_alt_insert_new_nodes_flt  ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_Int *OUT_marker );
HYPRE_Int hypre_alt_insert_new_nodes_dbl  ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_Int *OUT_marker );
HYPRE_Int hypre_alt_insert_new_nodes_long_dbl  ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_Int *OUT_marker );
HYPRE_Int hypre_big_insert_new_nodes_flt  ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_BigInt offset, HYPRE_BigInt *OUT_marker );
HYPRE_Int hypre_big_insert_new_nodes_dbl  ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_BigInt offset, HYPRE_BigInt *OUT_marker );
HYPRE_Int hypre_big_insert_new_nodes_long_dbl  ( hypre_ParCSRCommPkg *comm_pkg,
                                       hypre_ParCSRCommPkg *extend_comm_pkg, HYPRE_Int *IN_marker, HYPRE_Int full_off_procNodes,
                                       HYPRE_BigInt offset, HYPRE_BigInt *OUT_marker );
void hypre_build_interp_colmap_flt (hypre_ParCSRMatrix *P, HYPRE_Int full_off_procNodes,
                               HYPRE_Int *tmp_CF_marker_offd, HYPRE_BigInt *fine_to_coarse_offd);
void hypre_build_interp_colmap_dbl (hypre_ParCSRMatrix *P, HYPRE_Int full_off_procNodes,
                               HYPRE_Int *tmp_CF_marker_offd, HYPRE_BigInt *fine_to_coarse_offd);
void hypre_build_interp_colmap_long_dbl (hypre_ParCSRMatrix *P, HYPRE_Int full_off_procNodes,
                               HYPRE_Int *tmp_CF_marker_offd, HYPRE_BigInt *fine_to_coarse_offd);
HYPRE_Int hypre_exchange_interp_data_flt ( HYPRE_Int **CF_marker_offd, HYPRE_Int **dof_func_offd,
                                      hypre_CSRMatrix **A_ext, HYPRE_Int *full_off_procNodes, hypre_CSRMatrix **Sop,
                                      hypre_ParCSRCommPkg **extend_comm_pkg, hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                      hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                      HYPRE_Int skip_fine_or_same_sign);
HYPRE_Int hypre_exchange_interp_data_dbl ( HYPRE_Int **CF_marker_offd, HYPRE_Int **dof_func_offd,
                                      hypre_CSRMatrix **A_ext, HYPRE_Int *full_off_procNodes, hypre_CSRMatrix **Sop,
                                      hypre_ParCSRCommPkg **extend_comm_pkg, hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                      hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                      HYPRE_Int skip_fine_or_same_sign);
HYPRE_Int hypre_exchange_interp_data_long_dbl ( HYPRE_Int **CF_marker_offd, HYPRE_Int **dof_func_offd,
                                      hypre_CSRMatrix **A_ext, HYPRE_Int *full_off_procNodes, hypre_CSRMatrix **Sop,
                                      hypre_ParCSRCommPkg **extend_comm_pkg, hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                      hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                      HYPRE_Int skip_fine_or_same_sign);
HYPRE_Int hypre_exchange_marker_flt (hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *IN_marker,
                                HYPRE_Int *OUT_marker);
HYPRE_Int hypre_exchange_marker_dbl (hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *IN_marker,
                                HYPRE_Int *OUT_marker);
HYPRE_Int hypre_exchange_marker_long_dbl (hypre_ParCSRCommPkg *comm_pkg, HYPRE_Int *IN_marker,
                                HYPRE_Int *OUT_marker);
HYPRE_Int hypre_index_of_minimum_flt  ( HYPRE_BigInt *data, HYPRE_Int n );
HYPRE_Int hypre_index_of_minimum_dbl  ( HYPRE_BigInt *data, HYPRE_Int n );
HYPRE_Int hypre_index_of_minimum_long_dbl  ( HYPRE_BigInt *data, HYPRE_Int n );
void hypre_initialize_vecs_flt  ( HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc,
                             HYPRE_BigInt *offd_ftc, HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF );
void hypre_initialize_vecs_dbl  ( HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc,
                             HYPRE_BigInt *offd_ftc, HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF );
void hypre_initialize_vecs_long_dbl  ( HYPRE_Int diag_n, HYPRE_Int offd_n, HYPRE_Int *diag_ftc,
                             HYPRE_BigInt *offd_ftc, HYPRE_Int *diag_pm, HYPRE_Int *offd_pm, HYPRE_Int *tmp_CF );
HYPRE_Int hypre_ssort_flt  ( HYPRE_BigInt *data, HYPRE_Int n );
HYPRE_Int hypre_ssort_dbl  ( HYPRE_BigInt *data, HYPRE_Int n );
HYPRE_Int hypre_ssort_long_dbl  ( HYPRE_BigInt *data, HYPRE_Int n );
void hypre_swap_int_flt  ( HYPRE_BigInt *data, HYPRE_Int a, HYPRE_Int b );
void hypre_swap_int_dbl  ( HYPRE_BigInt *data, HYPRE_Int a, HYPRE_Int b );
void hypre_swap_int_long_dbl  ( HYPRE_BigInt *data, HYPRE_Int a, HYPRE_Int b );
void *hypre_BlockTridiagCreate_flt  ( void );
void *hypre_BlockTridiagCreate_dbl  ( void );
void *hypre_BlockTridiagCreate_long_dbl  ( void );
HYPRE_Int hypre_BlockTridiagDestroy_flt  ( void *data );
HYPRE_Int hypre_BlockTridiagDestroy_dbl  ( void *data );
HYPRE_Int hypre_BlockTridiagDestroy_long_dbl  ( void *data );
HYPRE_Int hypre_BlockTridiagSetAMGNumSweeps_flt  ( void *data, HYPRE_Int nsweeps );
HYPRE_Int hypre_BlockTridiagSetAMGNumSweeps_dbl  ( void *data, HYPRE_Int nsweeps );
HYPRE_Int hypre_BlockTridiagSetAMGNumSweeps_long_dbl  ( void *data, HYPRE_Int nsweeps );
HYPRE_Int hypre_BlockTridiagSetAMGRelaxType_flt  ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BlockTridiagSetAMGRelaxType_dbl  ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BlockTridiagSetAMGRelaxType_long_dbl  ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BlockTridiagSetAMGStrengthThreshold_flt  ( void *data, hypre_float thresh );
HYPRE_Int hypre_BlockTridiagSetAMGStrengthThreshold_dbl  ( void *data, hypre_double thresh );
HYPRE_Int hypre_BlockTridiagSetAMGStrengthThreshold_long_dbl  ( void *data, hypre_long_double thresh );
HYPRE_Int hypre_BlockTridiagSetIndexSet_flt  ( void *data, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int hypre_BlockTridiagSetIndexSet_dbl  ( void *data, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int hypre_BlockTridiagSetIndexSet_long_dbl  ( void *data, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int hypre_BlockTridiagSetPrintLevel_flt  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_BlockTridiagSetPrintLevel_dbl  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_BlockTridiagSetPrintLevel_long_dbl  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_BlockTridiagSetup_flt  ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSetup_dbl  ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSetup_long_dbl  ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSolve_flt  ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSolve_dbl  ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_BlockTridiagSolve_long_dbl  ( void *data, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                    hypre_ParVector *x );
HYPRE_Int hypre_GenerateSubComm_flt  ( MPI_Comm comm, HYPRE_Int participate, MPI_Comm *new_comm_ptr );
HYPRE_Int hypre_GenerateSubComm_dbl  ( MPI_Comm comm, HYPRE_Int participate, MPI_Comm *new_comm_ptr );
HYPRE_Int hypre_GenerateSubComm_long_dbl  ( MPI_Comm comm, HYPRE_Int participate, MPI_Comm *new_comm_ptr );
void hypre_merge_lists_flt  ( HYPRE_Int *list1, HYPRE_Int *list2, hypre_int *np1,
                         hypre_MPI_Datatype *dptr );
void hypre_merge_lists_dbl  ( HYPRE_Int *list1, HYPRE_Int *list2, hypre_int *np1,
                         hypre_MPI_Datatype *dptr );
void hypre_merge_lists_long_dbl  ( HYPRE_Int *list1, HYPRE_Int *list2, hypre_int *np1,
                         hypre_MPI_Datatype *dptr );
HYPRE_Int hypre_seqAMGCycle_flt  ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              hypre_ParVector **Par_F_array, hypre_ParVector **Par_U_array );
HYPRE_Int hypre_seqAMGCycle_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              hypre_ParVector **Par_F_array, hypre_ParVector **Par_U_array );
HYPRE_Int hypre_seqAMGCycle_long_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              hypre_ParVector **Par_F_array, hypre_ParVector **Par_U_array );
HYPRE_Int hypre_seqAMGSetup_flt  ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              HYPRE_Int coarse_threshold );
HYPRE_Int hypre_seqAMGSetup_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              HYPRE_Int coarse_threshold );
HYPRE_Int hypre_seqAMGSetup_long_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int p_level,
                              HYPRE_Int coarse_threshold );
HYPRE_Int HYPRE_ADSCreate_flt  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ADSCreate_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ADSCreate_long_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ADSDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ADSDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ADSDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *rel_resid_norm );
HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *rel_resid_norm );
HYPRE_Int HYPRE_ADSGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *rel_resid_norm );
HYPRE_Int HYPRE_ADSGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ADSGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ADSGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ADSSetAMGOptions_flt  ( HYPRE_Solver solver, HYPRE_Int coarsen_type,
                                   HYPRE_Int agg_levels, HYPRE_Int relax_type, hypre_float strength_threshold, HYPRE_Int interp_type,
                                   HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetAMGOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int coarsen_type,
                                   HYPRE_Int agg_levels, HYPRE_Int relax_type, hypre_double strength_threshold, HYPRE_Int interp_type,
                                   HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetAMGOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int coarsen_type,
                                   HYPRE_Int agg_levels, HYPRE_Int relax_type, hypre_long_double strength_threshold, HYPRE_Int interp_type,
                                   HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetAMSOptions_flt  ( HYPRE_Solver solver, HYPRE_Int cycle_type,
                                   HYPRE_Int coarsen_type, HYPRE_Int agg_levels, HYPRE_Int relax_type, hypre_float strength_threshold,
                                   HYPRE_Int interp_type, HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetAMSOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type,
                                   HYPRE_Int coarsen_type, HYPRE_Int agg_levels, HYPRE_Int relax_type, hypre_double strength_threshold,
                                   HYPRE_Int interp_type, HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetAMSOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type,
                                   HYPRE_Int coarsen_type, HYPRE_Int agg_levels, HYPRE_Int relax_type, hypre_long_double strength_threshold,
                                   HYPRE_Int interp_type, HYPRE_Int Pmax );
HYPRE_Int HYPRE_ADSSetChebySmoothingOptions_flt  ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              hypre_float cheby_fraction );
HYPRE_Int HYPRE_ADSSetChebySmoothingOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              hypre_double cheby_fraction );
HYPRE_Int HYPRE_ADSSetChebySmoothingOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              hypre_long_double cheby_fraction );
HYPRE_Int HYPRE_ADSSetCoordinateVectors_flt  ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_ADSSetCoordinateVectors_dbl  ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_ADSSetCoordinateVectors_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_ADSSetCycleType_flt  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ADSSetCycleType_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ADSSetCycleType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ADSSetDiscreteCurl_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix C );
HYPRE_Int HYPRE_ADSSetDiscreteCurl_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix C );
HYPRE_Int HYPRE_ADSSetDiscreteCurl_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix C );
HYPRE_Int HYPRE_ADSSetDiscreteGradient_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_ADSSetDiscreteGradient_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_ADSSetDiscreteGradient_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_ADSSetInterpolations_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix RT_Pi,
                                       HYPRE_ParCSRMatrix RT_Pix, HYPRE_ParCSRMatrix RT_Piy, HYPRE_ParCSRMatrix RT_Piz,
                                       HYPRE_ParCSRMatrix ND_Pi, HYPRE_ParCSRMatrix ND_Pix, HYPRE_ParCSRMatrix ND_Piy,
                                       HYPRE_ParCSRMatrix ND_Piz );
HYPRE_Int HYPRE_ADSSetInterpolations_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix RT_Pi,
                                       HYPRE_ParCSRMatrix RT_Pix, HYPRE_ParCSRMatrix RT_Piy, HYPRE_ParCSRMatrix RT_Piz,
                                       HYPRE_ParCSRMatrix ND_Pi, HYPRE_ParCSRMatrix ND_Pix, HYPRE_ParCSRMatrix ND_Piy,
                                       HYPRE_ParCSRMatrix ND_Piz );
HYPRE_Int HYPRE_ADSSetInterpolations_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix RT_Pi,
                                       HYPRE_ParCSRMatrix RT_Pix, HYPRE_ParCSRMatrix RT_Piy, HYPRE_ParCSRMatrix RT_Piz,
                                       HYPRE_ParCSRMatrix ND_Pi, HYPRE_ParCSRMatrix ND_Pix, HYPRE_ParCSRMatrix ND_Piy,
                                       HYPRE_ParCSRMatrix ND_Piz );
HYPRE_Int HYPRE_ADSSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_ADSSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_ADSSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_ADSSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ADSSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ADSSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ADSSetSmoothingOptions_flt  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, hypre_float relax_weight, hypre_float omega );
HYPRE_Int HYPRE_ADSSetSmoothingOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, hypre_double relax_weight, hypre_double omega );
HYPRE_Int HYPRE_ADSSetSmoothingOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, hypre_long_double relax_weight, hypre_long_double omega );
HYPRE_Int HYPRE_ADSSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ADSSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ADSSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ADSSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_ADSSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMECreate_flt  ( HYPRE_Solver *esolver );
HYPRE_Int HYPRE_AMECreate_dbl  ( HYPRE_Solver *esolver );
HYPRE_Int HYPRE_AMECreate_long_dbl  ( HYPRE_Solver *esolver );
HYPRE_Int HYPRE_AMEDestroy_flt  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMEDestroy_dbl  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMEDestroy_long_dbl  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMEGetEigenvalues_flt  ( HYPRE_Solver esolver, hypre_float **eigenvalues );
HYPRE_Int HYPRE_AMEGetEigenvalues_dbl  ( HYPRE_Solver esolver, hypre_double **eigenvalues );
HYPRE_Int HYPRE_AMEGetEigenvalues_long_dbl  ( HYPRE_Solver esolver, hypre_long_double **eigenvalues );
HYPRE_Int HYPRE_AMEGetEigenvectors_flt  ( HYPRE_Solver esolver, HYPRE_ParVector **eigenvectors );
HYPRE_Int HYPRE_AMEGetEigenvectors_dbl  ( HYPRE_Solver esolver, HYPRE_ParVector **eigenvectors );
HYPRE_Int HYPRE_AMEGetEigenvectors_long_dbl  ( HYPRE_Solver esolver, HYPRE_ParVector **eigenvectors );
HYPRE_Int HYPRE_AMESetAMSSolver_flt  ( HYPRE_Solver esolver, HYPRE_Solver ams_solver );
HYPRE_Int HYPRE_AMESetAMSSolver_dbl  ( HYPRE_Solver esolver, HYPRE_Solver ams_solver );
HYPRE_Int HYPRE_AMESetAMSSolver_long_dbl  ( HYPRE_Solver esolver, HYPRE_Solver ams_solver );
HYPRE_Int HYPRE_AMESetBlockSize_flt  ( HYPRE_Solver esolver, HYPRE_Int block_size );
HYPRE_Int HYPRE_AMESetBlockSize_dbl  ( HYPRE_Solver esolver, HYPRE_Int block_size );
HYPRE_Int HYPRE_AMESetBlockSize_long_dbl  ( HYPRE_Solver esolver, HYPRE_Int block_size );
HYPRE_Int HYPRE_AMESetMassMatrix_flt  ( HYPRE_Solver esolver, HYPRE_ParCSRMatrix M );
HYPRE_Int HYPRE_AMESetMassMatrix_dbl  ( HYPRE_Solver esolver, HYPRE_ParCSRMatrix M );
HYPRE_Int HYPRE_AMESetMassMatrix_long_dbl  ( HYPRE_Solver esolver, HYPRE_ParCSRMatrix M );
HYPRE_Int HYPRE_AMESetMaxIter_flt  ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetMaxIter_dbl  ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetMaxIter_long_dbl  ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetMaxPCGIter_flt  ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetMaxPCGIter_dbl  ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetMaxPCGIter_long_dbl  ( HYPRE_Solver esolver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMESetPrintLevel_flt  ( HYPRE_Solver esolver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMESetPrintLevel_dbl  ( HYPRE_Solver esolver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMESetPrintLevel_long_dbl  ( HYPRE_Solver esolver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMESetRTol_flt  ( HYPRE_Solver esolver, hypre_float tol );
HYPRE_Int HYPRE_AMESetRTol_dbl  ( HYPRE_Solver esolver, hypre_double tol );
HYPRE_Int HYPRE_AMESetRTol_long_dbl  ( HYPRE_Solver esolver, hypre_long_double tol );
HYPRE_Int HYPRE_AMESetTol_flt  ( HYPRE_Solver esolver, hypre_float tol );
HYPRE_Int HYPRE_AMESetTol_dbl  ( HYPRE_Solver esolver, hypre_double tol );
HYPRE_Int HYPRE_AMESetTol_long_dbl  ( HYPRE_Solver esolver, hypre_long_double tol );
HYPRE_Int HYPRE_AMESetup_flt  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESetup_dbl  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESetup_long_dbl  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESolve_flt  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESolve_dbl  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMESolve_long_dbl  ( HYPRE_Solver esolver );
HYPRE_Int HYPRE_AMSConstructDiscreteGradient_flt  ( HYPRE_ParCSRMatrix A, HYPRE_ParVector x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, HYPRE_ParCSRMatrix *G );
HYPRE_Int HYPRE_AMSConstructDiscreteGradient_dbl  ( HYPRE_ParCSRMatrix A, HYPRE_ParVector x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, HYPRE_ParCSRMatrix *G );
HYPRE_Int HYPRE_AMSConstructDiscreteGradient_long_dbl  ( HYPRE_ParCSRMatrix A, HYPRE_ParVector x_coord,
                                               HYPRE_BigInt *edge_vertex, HYPRE_Int edge_orientation, HYPRE_ParCSRMatrix *G );
HYPRE_Int HYPRE_AMSCreate_flt  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_AMSCreate_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_AMSCreate_long_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_AMSDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSFEIDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSFEIDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSFEIDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_AMSFEISetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x, HYPRE_BigInt *EdgeNodeList_, HYPRE_BigInt *NodeNumbers_, HYPRE_Int numEdges_,
                              HYPRE_Int numLocalNodes_, HYPRE_Int numNodes_, hypre_float *NodalCoord_ );
HYPRE_Int HYPRE_AMSFEISetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x, HYPRE_BigInt *EdgeNodeList_, HYPRE_BigInt *NodeNumbers_, HYPRE_Int numEdges_,
                              HYPRE_Int numLocalNodes_, HYPRE_Int numNodes_, hypre_double *NodalCoord_ );
HYPRE_Int HYPRE_AMSFEISetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x, HYPRE_BigInt *EdgeNodeList_, HYPRE_BigInt *NodeNumbers_, HYPRE_Int numEdges_,
                              HYPRE_Int numLocalNodes_, HYPRE_Int numNodes_, hypre_long_double *NodalCoord_ );
HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *rel_resid_norm );
HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *rel_resid_norm );
HYPRE_Int HYPRE_AMSGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *rel_resid_norm );
HYPRE_Int HYPRE_AMSGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_AMSGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_AMSGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_AMSProjectOutGradients_flt  ( HYPRE_Solver solver, HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSProjectOutGradients_dbl  ( HYPRE_Solver solver, HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSProjectOutGradients_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSetAlphaAMGCoarseRelaxType_flt  ( HYPRE_Solver solver,
                                                HYPRE_Int alpha_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetAlphaAMGCoarseRelaxType_dbl  ( HYPRE_Solver solver,
                                                HYPRE_Int alpha_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetAlphaAMGCoarseRelaxType_long_dbl  ( HYPRE_Solver solver,
                                                HYPRE_Int alpha_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetAlphaAMGOptions_flt  ( HYPRE_Solver solver, HYPRE_Int alpha_coarsen_type,
                                        HYPRE_Int alpha_agg_levels, HYPRE_Int alpha_relax_type, hypre_float alpha_strength_threshold,
                                        HYPRE_Int alpha_interp_type, HYPRE_Int alpha_Pmax );
HYPRE_Int HYPRE_AMSSetAlphaAMGOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int alpha_coarsen_type,
                                        HYPRE_Int alpha_agg_levels, HYPRE_Int alpha_relax_type, hypre_double alpha_strength_threshold,
                                        HYPRE_Int alpha_interp_type, HYPRE_Int alpha_Pmax );
HYPRE_Int HYPRE_AMSSetAlphaAMGOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int alpha_coarsen_type,
                                        HYPRE_Int alpha_agg_levels, HYPRE_Int alpha_relax_type, hypre_long_double alpha_strength_threshold,
                                        HYPRE_Int alpha_interp_type, HYPRE_Int alpha_Pmax );
HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_alpha );
HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_alpha );
HYPRE_Int HYPRE_AMSSetAlphaPoissonMatrix_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_alpha );
HYPRE_Int HYPRE_AMSSetBetaAMGCoarseRelaxType_flt  ( HYPRE_Solver solver,
                                               HYPRE_Int beta_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetBetaAMGCoarseRelaxType_dbl  ( HYPRE_Solver solver,
                                               HYPRE_Int beta_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetBetaAMGCoarseRelaxType_long_dbl  ( HYPRE_Solver solver,
                                               HYPRE_Int beta_coarse_relax_type );
HYPRE_Int HYPRE_AMSSetBetaAMGOptions_flt  ( HYPRE_Solver solver, HYPRE_Int beta_coarsen_type,
                                       HYPRE_Int beta_agg_levels, HYPRE_Int beta_relax_type, hypre_float beta_strength_threshold,
                                       HYPRE_Int beta_interp_type, HYPRE_Int beta_Pmax );
HYPRE_Int HYPRE_AMSSetBetaAMGOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int beta_coarsen_type,
                                       HYPRE_Int beta_agg_levels, HYPRE_Int beta_relax_type, hypre_double beta_strength_threshold,
                                       HYPRE_Int beta_interp_type, HYPRE_Int beta_Pmax );
HYPRE_Int HYPRE_AMSSetBetaAMGOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int beta_coarsen_type,
                                       HYPRE_Int beta_agg_levels, HYPRE_Int beta_relax_type, hypre_long_double beta_strength_threshold,
                                       HYPRE_Int beta_interp_type, HYPRE_Int beta_Pmax );
HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_beta );
HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_beta );
HYPRE_Int HYPRE_AMSSetBetaPoissonMatrix_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A_beta );
HYPRE_Int HYPRE_AMSSetChebySmoothingOptions_flt  ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              hypre_float cheby_fraction );
HYPRE_Int HYPRE_AMSSetChebySmoothingOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              hypre_double cheby_fraction );
HYPRE_Int HYPRE_AMSSetChebySmoothingOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cheby_order,
                                              hypre_long_double cheby_fraction );
HYPRE_Int HYPRE_AMSSetCoordinateVectors_flt  ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_AMSSetCoordinateVectors_dbl  ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_AMSSetCoordinateVectors_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector x, HYPRE_ParVector y,
                                          HYPRE_ParVector z );
HYPRE_Int HYPRE_AMSSetCycleType_flt  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_AMSSetCycleType_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_AMSSetCycleType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_AMSSetDimension_flt  ( HYPRE_Solver solver, HYPRE_Int dim );
HYPRE_Int HYPRE_AMSSetDimension_dbl  ( HYPRE_Solver solver, HYPRE_Int dim );
HYPRE_Int HYPRE_AMSSetDimension_long_dbl  ( HYPRE_Solver solver, HYPRE_Int dim );
HYPRE_Int HYPRE_AMSSetDiscreteGradient_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_AMSSetDiscreteGradient_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_AMSSetDiscreteGradient_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix G );
HYPRE_Int HYPRE_AMSSetEdgeConstantVectors_flt  ( HYPRE_Solver solver, HYPRE_ParVector Gx,
                                            HYPRE_ParVector Gy, HYPRE_ParVector Gz );
HYPRE_Int HYPRE_AMSSetEdgeConstantVectors_dbl  ( HYPRE_Solver solver, HYPRE_ParVector Gx,
                                            HYPRE_ParVector Gy, HYPRE_ParVector Gz );
HYPRE_Int HYPRE_AMSSetEdgeConstantVectors_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector Gx,
                                            HYPRE_ParVector Gy, HYPRE_ParVector Gz );
HYPRE_Int HYPRE_AMSSetInteriorNodes_flt  ( HYPRE_Solver solver, HYPRE_ParVector interior_nodes );
HYPRE_Int HYPRE_AMSSetInteriorNodes_dbl  ( HYPRE_Solver solver, HYPRE_ParVector interior_nodes );
HYPRE_Int HYPRE_AMSSetInteriorNodes_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector interior_nodes );
HYPRE_Int HYPRE_AMSSetInterpolations_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix Pi,
                                       HYPRE_ParCSRMatrix Pix, HYPRE_ParCSRMatrix Piy, HYPRE_ParCSRMatrix Piz );
HYPRE_Int HYPRE_AMSSetInterpolations_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix Pi,
                                       HYPRE_ParCSRMatrix Pix, HYPRE_ParCSRMatrix Piy, HYPRE_ParCSRMatrix Piz );
HYPRE_Int HYPRE_AMSSetInterpolations_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix Pi,
                                       HYPRE_ParCSRMatrix Pix, HYPRE_ParCSRMatrix Piy, HYPRE_ParCSRMatrix Piz );
HYPRE_Int HYPRE_AMSSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMSSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMSSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int maxit );
HYPRE_Int HYPRE_AMSSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMSSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMSSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_AMSSetProjectionFrequency_flt  ( HYPRE_Solver solver, HYPRE_Int projection_frequency );
HYPRE_Int HYPRE_AMSSetProjectionFrequency_dbl  ( HYPRE_Solver solver, HYPRE_Int projection_frequency );
HYPRE_Int HYPRE_AMSSetProjectionFrequency_long_dbl  ( HYPRE_Solver solver, HYPRE_Int projection_frequency );
HYPRE_Int HYPRE_AMSSetSmoothingOptions_flt  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, hypre_float relax_weight, hypre_float omega );
HYPRE_Int HYPRE_AMSSetSmoothingOptions_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, hypre_double relax_weight, hypre_double omega );
HYPRE_Int HYPRE_AMSSetSmoothingOptions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                         HYPRE_Int relax_times, hypre_long_double relax_weight, hypre_long_double omega );
HYPRE_Int HYPRE_AMSSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_AMSSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_AMSSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_AMSSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_AMSSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDGetAMG_flt  ( HYPRE_Solver solver, HYPRE_Solver *amg_solver );
HYPRE_Int HYPRE_BoomerAMGDDGetAMG_dbl  ( HYPRE_Solver solver, HYPRE_Solver *amg_solver );
HYPRE_Int HYPRE_BoomerAMGDDGetAMG_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *amg_solver );
HYPRE_Int HYPRE_BoomerAMGDDGetFACCycleType_flt  ( HYPRE_Solver solver, HYPRE_Int *fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACCycleType_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACCycleType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumCycles_flt  ( HYPRE_Solver solver, HYPRE_Int *fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumCycles_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumCycles_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumRelax_flt  ( HYPRE_Solver solver, HYPRE_Int *fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumRelax_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDGetFACNumRelax_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int *fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxWeight_flt  ( HYPRE_Solver solver, hypre_float *fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxWeight_dbl  ( HYPRE_Solver solver, hypre_double *fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDGetFACRelaxWeight_long_dbl  ( HYPRE_Solver solver, hypre_long_double *fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDGetNumGhostLayers_flt  ( HYPRE_Solver solver, HYPRE_Int *num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDGetNumGhostLayers_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDGetNumGhostLayers_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDGetPadding_flt  ( HYPRE_Solver solver, HYPRE_Int *padding );
HYPRE_Int HYPRE_BoomerAMGDDGetPadding_dbl  ( HYPRE_Solver solver, HYPRE_Int *padding );
HYPRE_Int HYPRE_BoomerAMGDDGetPadding_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *padding );
HYPRE_Int HYPRE_BoomerAMGDDGetStartLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *start_level );
HYPRE_Int HYPRE_BoomerAMGDDGetStartLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *start_level );
HYPRE_Int HYPRE_BoomerAMGDDGetStartLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *start_level );
HYPRE_Int HYPRE_BoomerAMGDDSetFACCycleType_flt  ( HYPRE_Solver solver, HYPRE_Int fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACCycleType_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACCycleType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_cycle_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumCycles_flt  ( HYPRE_Solver solver, HYPRE_Int fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumCycles_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumCycles_long_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_num_cycles );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumRelax_flt  ( HYPRE_Solver solver, HYPRE_Int fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumRelax_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDSetFACNumRelax_long_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_num_relax );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int fac_relax_type );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxWeight_flt  ( HYPRE_Solver solver, hypre_float fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxWeight_dbl  ( HYPRE_Solver solver, hypre_double fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDSetFACRelaxWeight_long_dbl  ( HYPRE_Solver solver, hypre_long_double fac_relax_weight );
HYPRE_Int HYPRE_BoomerAMGDDSetNumGhostLayers_flt  ( HYPRE_Solver solver, HYPRE_Int num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDSetNumGhostLayers_dbl  ( HYPRE_Solver solver, HYPRE_Int num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDSetNumGhostLayers_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_ghost_layers );
HYPRE_Int HYPRE_BoomerAMGDDSetPadding_flt  ( HYPRE_Solver solver, HYPRE_Int padding );
HYPRE_Int HYPRE_BoomerAMGDDSetPadding_dbl  ( HYPRE_Solver solver, HYPRE_Int padding );
HYPRE_Int HYPRE_BoomerAMGDDSetPadding_long_dbl  ( HYPRE_Solver solver, HYPRE_Int padding );
HYPRE_Int HYPRE_BoomerAMGDDSetStartLevel_flt  ( HYPRE_Solver solver, HYPRE_Int start_level );
HYPRE_Int HYPRE_BoomerAMGDDSetStartLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int start_level );
HYPRE_Int HYPRE_BoomerAMGDDSetStartLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int start_level );
HYPRE_Int HYPRE_BoomerAMGDDSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDSetUserFACRelaxation_flt ( HYPRE_Solver solver,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int HYPRE_BoomerAMGDDSetUserFACRelaxation_dbl ( HYPRE_Solver solver,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int HYPRE_BoomerAMGDDSetUserFACRelaxation_long_dbl ( HYPRE_Solver solver,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int HYPRE_BoomerAMGDDSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGDDSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGCreate_flt  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BoomerAMGCreate_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BoomerAMGCreate_long_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BoomerAMGDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGGetAdditive_flt  ( HYPRE_Solver solver, HYPRE_Int *additive );
HYPRE_Int HYPRE_BoomerAMGGetAdditive_dbl  ( HYPRE_Solver solver, HYPRE_Int *additive );
HYPRE_Int HYPRE_BoomerAMGGetAdditive_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *additive );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenCutFactor_flt ( HYPRE_Solver solver, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenCutFactor_dbl ( HYPRE_Solver solver, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenCutFactor_long_dbl ( HYPRE_Solver solver, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenType_flt  ( HYPRE_Solver solver, HYPRE_Int *coarsen_type );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenType_dbl  ( HYPRE_Solver solver, HYPRE_Int *coarsen_type );
HYPRE_Int HYPRE_BoomerAMGGetCoarsenType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *coarsen_type );
HYPRE_Int HYPRE_BoomerAMGGetConvergeType_flt  ( HYPRE_Solver solver, HYPRE_Int *type );
HYPRE_Int HYPRE_BoomerAMGGetConvergeType_dbl  ( HYPRE_Solver solver, HYPRE_Int *type );
HYPRE_Int HYPRE_BoomerAMGGetConvergeType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *type );
HYPRE_Int HYPRE_BoomerAMGGetCumNnzAP_flt  ( HYPRE_Solver solver, hypre_float *cum_nnz_AP );
HYPRE_Int HYPRE_BoomerAMGGetCumNnzAP_dbl  ( HYPRE_Solver solver, hypre_double *cum_nnz_AP );
HYPRE_Int HYPRE_BoomerAMGGetCumNnzAP_long_dbl  ( HYPRE_Solver solver, hypre_long_double *cum_nnz_AP );
HYPRE_Int HYPRE_BoomerAMGGetCumNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *cum_num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetCumNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *cum_num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetCumNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *cum_num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetCycleNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int *num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int *relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int *relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGGetCycleType_flt  ( HYPRE_Solver solver, HYPRE_Int *cycle_type );
HYPRE_Int HYPRE_BoomerAMGGetCycleType_dbl  ( HYPRE_Solver solver, HYPRE_Int *cycle_type );
HYPRE_Int HYPRE_BoomerAMGGetCycleType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *cycle_type );
HYPRE_Int HYPRE_BoomerAMGGetDebugFlag_flt  ( HYPRE_Solver solver, HYPRE_Int *debug_flag );
HYPRE_Int HYPRE_BoomerAMGGetDebugFlag_dbl  ( HYPRE_Solver solver, HYPRE_Int *debug_flag );
HYPRE_Int HYPRE_BoomerAMGGetDebugFlag_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *debug_flag );
HYPRE_Int HYPRE_BoomerAMGGetDomainType_flt  ( HYPRE_Solver solver, HYPRE_Int *domain_type );
HYPRE_Int HYPRE_BoomerAMGGetDomainType_dbl  ( HYPRE_Solver solver, HYPRE_Int *domain_type );
HYPRE_Int HYPRE_BoomerAMGGetDomainType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *domain_type );
HYPRE_Int HYPRE_BoomerAMGGetFCycle_flt  ( HYPRE_Solver solver, HYPRE_Int *fcycle );
HYPRE_Int HYPRE_BoomerAMGGetFCycle_dbl  ( HYPRE_Solver solver, HYPRE_Int *fcycle );
HYPRE_Int HYPRE_BoomerAMGGetFCycle_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *fcycle );
HYPRE_Int HYPRE_BoomerAMGGetFilterThresholdR_flt  ( HYPRE_Solver solver, hypre_float *filter_threshold );
HYPRE_Int HYPRE_BoomerAMGGetFilterThresholdR_dbl  ( HYPRE_Solver solver, hypre_double *filter_threshold );
HYPRE_Int HYPRE_BoomerAMGGetFilterThresholdR_long_dbl  ( HYPRE_Solver solver, hypre_long_double *filter_threshold );
HYPRE_Int HYPRE_BoomerAMGGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver,
                                                        hypre_float *rel_resid_norm );
HYPRE_Int HYPRE_BoomerAMGGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver,
                                                        hypre_double *rel_resid_norm );
HYPRE_Int HYPRE_BoomerAMGGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver,
                                                        hypre_long_double *rel_resid_norm );
HYPRE_Int HYPRE_BoomerAMGGetGridHierarchy_flt (HYPRE_Solver solver, HYPRE_Int *cgrid );
HYPRE_Int HYPRE_BoomerAMGGetGridHierarchy_dbl (HYPRE_Solver solver, HYPRE_Int *cgrid );
HYPRE_Int HYPRE_BoomerAMGGetGridHierarchy_long_dbl (HYPRE_Solver solver, HYPRE_Int *cgrid );
HYPRE_Int HYPRE_BoomerAMGGetJacobiTruncThreshold_flt  ( HYPRE_Solver solver,
                                                   hypre_float *jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGGetJacobiTruncThreshold_dbl  ( HYPRE_Solver solver,
                                                   hypre_double *jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGGetJacobiTruncThreshold_long_dbl  ( HYPRE_Solver solver,
                                                   hypre_long_double *jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGGetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_BoomerAMGGetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_BoomerAMGGetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_BoomerAMGGetMaxCoarseSize_flt  ( HYPRE_Solver solver, HYPRE_Int *max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMaxCoarseSize_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMaxCoarseSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_BoomerAMGGetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_BoomerAMGGetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_BoomerAMGGetMaxLevels_flt  ( HYPRE_Solver solver, HYPRE_Int *max_levels );
HYPRE_Int HYPRE_BoomerAMGGetMaxLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_levels );
HYPRE_Int HYPRE_BoomerAMGGetMaxLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_levels );
HYPRE_Int HYPRE_BoomerAMGGetMaxRowSum_flt  ( HYPRE_Solver solver, hypre_float *max_row_sum );
HYPRE_Int HYPRE_BoomerAMGGetMaxRowSum_dbl  ( HYPRE_Solver solver, hypre_double *max_row_sum );
HYPRE_Int HYPRE_BoomerAMGGetMaxRowSum_long_dbl  ( HYPRE_Solver solver, hypre_long_double *max_row_sum );
HYPRE_Int HYPRE_BoomerAMGGetMeasureType_flt  ( HYPRE_Solver solver, HYPRE_Int *measure_type );
HYPRE_Int HYPRE_BoomerAMGGetMeasureType_dbl  ( HYPRE_Solver solver, HYPRE_Int *measure_type );
HYPRE_Int HYPRE_BoomerAMGGetMeasureType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *measure_type );
HYPRE_Int HYPRE_BoomerAMGGetMinCoarseSize_flt  ( HYPRE_Solver solver, HYPRE_Int *min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMinCoarseSize_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMinCoarseSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGGetMultAdditive_flt  ( HYPRE_Solver solver, HYPRE_Int *mult_additive );
HYPRE_Int HYPRE_BoomerAMGGetMultAdditive_dbl  ( HYPRE_Solver solver, HYPRE_Int *mult_additive );
HYPRE_Int HYPRE_BoomerAMGGetMultAdditive_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *mult_additive );
HYPRE_Int HYPRE_BoomerAMGGetNumFunctions_flt  ( HYPRE_Solver solver, HYPRE_Int *num_functions );
HYPRE_Int HYPRE_BoomerAMGGetNumFunctions_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_functions );
HYPRE_Int HYPRE_BoomerAMGGetNumFunctions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_functions );
HYPRE_Int HYPRE_BoomerAMGGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BoomerAMGGetOverlap_flt  ( HYPRE_Solver solver, HYPRE_Int *overlap );
HYPRE_Int HYPRE_BoomerAMGGetOverlap_dbl  ( HYPRE_Solver solver, HYPRE_Int *overlap );
HYPRE_Int HYPRE_BoomerAMGGetOverlap_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *overlap );
HYPRE_Int HYPRE_BoomerAMGGetPMaxElmts_flt  ( HYPRE_Solver solver, HYPRE_Int *P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGGetPMaxElmts_dbl  ( HYPRE_Solver solver, HYPRE_Int *P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGGetPMaxElmts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGGetPostInterpType_flt  ( HYPRE_Solver solver, HYPRE_Int *post_interp_type );
HYPRE_Int HYPRE_BoomerAMGGetPostInterpType_dbl  ( HYPRE_Solver solver, HYPRE_Int *post_interp_type );
HYPRE_Int HYPRE_BoomerAMGGetPostInterpType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *post_interp_type );
HYPRE_Int HYPRE_BoomerAMGGetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *print_level );
HYPRE_Int HYPRE_BoomerAMGGetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *print_level );
HYPRE_Int HYPRE_BoomerAMGGetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *print_level );
HYPRE_Int HYPRE_BoomerAMGGetRedundant_flt  ( HYPRE_Solver solver, HYPRE_Int *redundant );
HYPRE_Int HYPRE_BoomerAMGGetRedundant_dbl  ( HYPRE_Solver solver, HYPRE_Int *redundant );
HYPRE_Int HYPRE_BoomerAMGGetRedundant_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *redundant );
HYPRE_Int HYPRE_BoomerAMGGetResidual_flt  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_BoomerAMGGetResidual_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_BoomerAMGGetResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_BoomerAMGGetSchwarzRlxWeight_flt  ( HYPRE_Solver solver,
                                               hypre_float *schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGGetSchwarzRlxWeight_dbl  ( HYPRE_Solver solver,
                                               hypre_double *schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGGetSchwarzRlxWeight_long_dbl  ( HYPRE_Solver solver,
                                               hypre_long_double *schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGGetSeqThreshold_flt  ( HYPRE_Solver solver, HYPRE_Int *seq_threshold );
HYPRE_Int HYPRE_BoomerAMGGetSeqThreshold_dbl  ( HYPRE_Solver solver, HYPRE_Int *seq_threshold );
HYPRE_Int HYPRE_BoomerAMGGetSeqThreshold_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *seq_threshold );
HYPRE_Int HYPRE_BoomerAMGGetSimple_flt  ( HYPRE_Solver solver, HYPRE_Int *simple );
HYPRE_Int HYPRE_BoomerAMGGetSimple_dbl  ( HYPRE_Solver solver, HYPRE_Int *simple );
HYPRE_Int HYPRE_BoomerAMGGetSimple_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *simple );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumLevels_flt  ( HYPRE_Solver solver, HYPRE_Int *smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int *smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGGetSmoothNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGGetSmoothType_flt  ( HYPRE_Solver solver, HYPRE_Int *smooth_type );
HYPRE_Int HYPRE_BoomerAMGGetSmoothType_dbl  ( HYPRE_Solver solver, HYPRE_Int *smooth_type );
HYPRE_Int HYPRE_BoomerAMGGetSmoothType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *smooth_type );
HYPRE_Int HYPRE_BoomerAMGGetStrongThreshold_flt  ( HYPRE_Solver solver, hypre_float *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThreshold_dbl  ( HYPRE_Solver solver, hypre_double *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThresholdR_flt  ( HYPRE_Solver solver, hypre_float *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThresholdR_dbl  ( HYPRE_Solver solver, hypre_double *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetStrongThresholdR_long_dbl  ( HYPRE_Solver solver, hypre_long_double *strong_threshold );
HYPRE_Int HYPRE_BoomerAMGGetTol_flt  ( HYPRE_Solver solver, hypre_float *tol );
HYPRE_Int HYPRE_BoomerAMGGetTol_dbl  ( HYPRE_Solver solver, hypre_double *tol );
HYPRE_Int HYPRE_BoomerAMGGetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *tol );
HYPRE_Int HYPRE_BoomerAMGGetTruncFactor_flt  ( HYPRE_Solver solver, hypre_float *trunc_factor );
HYPRE_Int HYPRE_BoomerAMGGetTruncFactor_dbl  ( HYPRE_Solver solver, hypre_double *trunc_factor );
HYPRE_Int HYPRE_BoomerAMGGetTruncFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double *trunc_factor );
HYPRE_Int HYPRE_BoomerAMGGetVariant_flt  ( HYPRE_Solver solver, HYPRE_Int *variant );
HYPRE_Int HYPRE_BoomerAMGGetVariant_dbl  ( HYPRE_Solver solver, HYPRE_Int *variant );
HYPRE_Int HYPRE_BoomerAMGGetVariant_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *variant );
HYPRE_Int HYPRE_BoomerAMGInitGridRelaxation_flt  ( HYPRE_Int **num_grid_sweeps_ptr,
                                              HYPRE_Int **grid_relax_type_ptr, HYPRE_Int ***grid_relax_points_ptr, HYPRE_Int coarsen_type,
                                              hypre_float **relax_weights_ptr, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGInitGridRelaxation_dbl  ( HYPRE_Int **num_grid_sweeps_ptr,
                                              HYPRE_Int **grid_relax_type_ptr, HYPRE_Int ***grid_relax_points_ptr, HYPRE_Int coarsen_type,
                                              hypre_double **relax_weights_ptr, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGInitGridRelaxation_long_dbl  ( HYPRE_Int **num_grid_sweeps_ptr,
                                              HYPRE_Int **grid_relax_type_ptr, HYPRE_Int ***grid_relax_points_ptr, HYPRE_Int coarsen_type,
                                              hypre_long_double **relax_weights_ptr, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGSetAdditive_flt  ( HYPRE_Solver solver, HYPRE_Int additive );
HYPRE_Int HYPRE_BoomerAMGSetAdditive_dbl  ( HYPRE_Solver solver, HYPRE_Int additive );
HYPRE_Int HYPRE_BoomerAMGSetAdditive_long_dbl  ( HYPRE_Solver solver, HYPRE_Int additive );
HYPRE_Int HYPRE_BoomerAMGSetAddLastLvl_flt  ( HYPRE_Solver solver, HYPRE_Int add_last_lvl );
HYPRE_Int HYPRE_BoomerAMGSetAddLastLvl_dbl  ( HYPRE_Solver solver, HYPRE_Int add_last_lvl );
HYPRE_Int HYPRE_BoomerAMGSetAddLastLvl_long_dbl  ( HYPRE_Solver solver, HYPRE_Int add_last_lvl );
HYPRE_Int HYPRE_BoomerAMGSetAddPMaxElmts_flt  ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAddPMaxElmts_dbl  ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAddPMaxElmts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int add_rlx_type );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int add_rlx_type );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int add_rlx_type );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxWt_flt  ( HYPRE_Solver solver, hypre_float add_rlx_wt );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxWt_dbl  ( HYPRE_Solver solver, hypre_double add_rlx_wt );
HYPRE_Int HYPRE_BoomerAMGSetAddRelaxWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double add_rlx_wt );
HYPRE_Int HYPRE_BoomerAMGSetAddTruncFactor_flt  ( HYPRE_Solver solver, hypre_float add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAddTruncFactor_dbl  ( HYPRE_Solver solver, hypre_double add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAddTruncFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetADropTol_flt ( HYPRE_Solver solver, hypre_float A_drop_tol  );
HYPRE_Int HYPRE_BoomerAMGSetADropTol_dbl ( HYPRE_Solver solver, hypre_double A_drop_tol  );
HYPRE_Int HYPRE_BoomerAMGSetADropTol_long_dbl ( HYPRE_Solver solver, hypre_long_double A_drop_tol  );
HYPRE_Int HYPRE_BoomerAMGSetADropType_flt ( HYPRE_Solver solver, HYPRE_Int A_drop_type  );
HYPRE_Int HYPRE_BoomerAMGSetADropType_dbl ( HYPRE_Solver solver, HYPRE_Int A_drop_type  );
HYPRE_Int HYPRE_BoomerAMGSetADropType_long_dbl ( HYPRE_Solver solver, HYPRE_Int A_drop_type  );
HYPRE_Int HYPRE_BoomerAMGSetAggInterpType_flt  ( HYPRE_Solver solver, HYPRE_Int agg_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetAggInterpType_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetAggInterpType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetAggNumLevels_flt  ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetAggNumLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetAggNumLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetAggP12MaxElmts_flt  ( HYPRE_Solver solver, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAggP12MaxElmts_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAggP12MaxElmts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAggP12TruncFactor_flt  ( HYPRE_Solver solver,
                                                hypre_float agg_P12_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggP12TruncFactor_dbl  ( HYPRE_Solver solver,
                                                hypre_double agg_P12_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggP12TruncFactor_long_dbl  ( HYPRE_Solver solver,
                                                hypre_long_double agg_P12_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggPMaxElmts_flt  ( HYPRE_Solver solver, HYPRE_Int agg_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAggPMaxElmts_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAggPMaxElmts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetAggTruncFactor_flt  ( HYPRE_Solver solver, hypre_float agg_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggTruncFactor_dbl  ( HYPRE_Solver solver, hypre_double agg_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetAggTruncFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double agg_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetCGCIts_flt  ( HYPRE_Solver solver, HYPRE_Int its );
HYPRE_Int HYPRE_BoomerAMGSetCGCIts_dbl  ( HYPRE_Solver solver, HYPRE_Int its );
HYPRE_Int HYPRE_BoomerAMGSetCGCIts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int its );
HYPRE_Int HYPRE_BoomerAMGSetChebyEigEst_flt  ( HYPRE_Solver solver, HYPRE_Int eig_est );
HYPRE_Int HYPRE_BoomerAMGSetChebyEigEst_dbl  ( HYPRE_Solver solver, HYPRE_Int eig_est );
HYPRE_Int HYPRE_BoomerAMGSetChebyEigEst_long_dbl  ( HYPRE_Solver solver, HYPRE_Int eig_est );
HYPRE_Int HYPRE_BoomerAMGSetChebyFraction_flt  ( HYPRE_Solver solver, hypre_float ratio );
HYPRE_Int HYPRE_BoomerAMGSetChebyFraction_dbl  ( HYPRE_Solver solver, hypre_double ratio );
HYPRE_Int HYPRE_BoomerAMGSetChebyFraction_long_dbl  ( HYPRE_Solver solver, hypre_long_double ratio );
HYPRE_Int HYPRE_BoomerAMGSetChebyOrder_flt  ( HYPRE_Solver solver, HYPRE_Int order );
HYPRE_Int HYPRE_BoomerAMGSetChebyOrder_dbl  ( HYPRE_Solver solver, HYPRE_Int order );
HYPRE_Int HYPRE_BoomerAMGSetChebyOrder_long_dbl  ( HYPRE_Solver solver, HYPRE_Int order );
HYPRE_Int HYPRE_BoomerAMGSetChebyScale_flt  ( HYPRE_Solver solver, HYPRE_Int scale );
HYPRE_Int HYPRE_BoomerAMGSetChebyScale_dbl  ( HYPRE_Solver solver, HYPRE_Int scale );
HYPRE_Int HYPRE_BoomerAMGSetChebyScale_long_dbl  ( HYPRE_Solver solver, HYPRE_Int scale );
HYPRE_Int HYPRE_BoomerAMGSetChebyVariant_flt  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSetChebyVariant_dbl  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSetChebyVariant_long_dbl  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenCutFactor_flt ( HYPRE_Solver solver, HYPRE_Int coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenCutFactor_dbl ( HYPRE_Solver solver, HYPRE_Int coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenCutFactor_long_dbl ( HYPRE_Solver solver, HYPRE_Int coarsen_cut_factor );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenType_flt  ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenType_dbl  ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_BoomerAMGSetCoarsenType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_BoomerAMGSetConvergeType_flt  ( HYPRE_Solver solver, HYPRE_Int type );
HYPRE_Int HYPRE_BoomerAMGSetConvergeType_dbl  ( HYPRE_Solver solver, HYPRE_Int type );
HYPRE_Int HYPRE_BoomerAMGSetConvergeType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int type );
HYPRE_Int HYPRE_BoomerAMGSetCoordDim_flt  ( HYPRE_Solver solver, HYPRE_Int coorddim );
HYPRE_Int HYPRE_BoomerAMGSetCoordDim_dbl  ( HYPRE_Solver solver, HYPRE_Int coorddim );
HYPRE_Int HYPRE_BoomerAMGSetCoordDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int coorddim );
HYPRE_Int HYPRE_BoomerAMGSetCoordinates_flt  ( HYPRE_Solver solver, float *coordinates );
HYPRE_Int HYPRE_BoomerAMGSetCoordinates_dbl  ( HYPRE_Solver solver, float *coordinates );
HYPRE_Int HYPRE_BoomerAMGSetCoordinates_long_dbl  ( HYPRE_Solver solver, float *coordinates );
HYPRE_Int HYPRE_BoomerAMGSetCPoints_flt ( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetCPoints_dbl ( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetCPoints_long_dbl ( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetCpointsToKeep_flt ( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                           HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetCpointsToKeep_dbl ( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                           HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetCpointsToKeep_long_dbl ( HYPRE_Solver solver, HYPRE_Int cpt_coarse_level,
                                           HYPRE_Int num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index);
HYPRE_Int HYPRE_BoomerAMGSetCRRate_flt  ( HYPRE_Solver solver, hypre_float CR_rate );
HYPRE_Int HYPRE_BoomerAMGSetCRRate_dbl  ( HYPRE_Solver solver, hypre_double CR_rate );
HYPRE_Int HYPRE_BoomerAMGSetCRRate_long_dbl  ( HYPRE_Solver solver, hypre_long_double CR_rate );
HYPRE_Int HYPRE_BoomerAMGSetCRStrongTh_flt  ( HYPRE_Solver solver, hypre_float CR_strong_th );
HYPRE_Int HYPRE_BoomerAMGSetCRStrongTh_dbl  ( HYPRE_Solver solver, hypre_double CR_strong_th );
HYPRE_Int HYPRE_BoomerAMGSetCRStrongTh_long_dbl  ( HYPRE_Solver solver, hypre_long_double CR_strong_th );
HYPRE_Int HYPRE_BoomerAMGSetCRUseCG_flt  ( HYPRE_Solver solver, HYPRE_Int CR_use_CG );
HYPRE_Int HYPRE_BoomerAMGSetCRUseCG_dbl  ( HYPRE_Solver solver, HYPRE_Int CR_use_CG );
HYPRE_Int HYPRE_BoomerAMGSetCRUseCG_long_dbl  ( HYPRE_Solver solver, HYPRE_Int CR_use_CG );
HYPRE_Int HYPRE_BoomerAMGSetCumNnzAP_flt  ( HYPRE_Solver solver, hypre_float cum_nnz_AP );
HYPRE_Int HYPRE_BoomerAMGSetCumNnzAP_dbl  ( HYPRE_Solver solver, hypre_double cum_nnz_AP );
HYPRE_Int HYPRE_BoomerAMGSetCumNnzAP_long_dbl  ( HYPRE_Solver solver, hypre_long_double cum_nnz_AP );
HYPRE_Int HYPRE_BoomerAMGSetCycleNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetCycleNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetCycleNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetCycleRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetCycleRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetCycleRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                             HYPRE_Int k );
HYPRE_Int HYPRE_BoomerAMGSetCycleType_flt  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_BoomerAMGSetCycleType_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_BoomerAMGSetCycleType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_BoomerAMGSetDebugFlag_flt  ( HYPRE_Solver solver, HYPRE_Int debug_flag );
HYPRE_Int HYPRE_BoomerAMGSetDebugFlag_dbl  ( HYPRE_Solver solver, HYPRE_Int debug_flag );
HYPRE_Int HYPRE_BoomerAMGSetDebugFlag_long_dbl  ( HYPRE_Solver solver, HYPRE_Int debug_flag );
HYPRE_Int HYPRE_BoomerAMGSetDofFunc_flt  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_BoomerAMGSetDofFunc_dbl  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_BoomerAMGSetDofFunc_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_BoomerAMGSetDomainType_flt  ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_BoomerAMGSetDomainType_dbl  ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_BoomerAMGSetDomainType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_BoomerAMGSetDropTol_flt  ( HYPRE_Solver solver, hypre_float drop_tol );
HYPRE_Int HYPRE_BoomerAMGSetDropTol_dbl  ( HYPRE_Solver solver, hypre_double drop_tol );
HYPRE_Int HYPRE_BoomerAMGSetDropTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double drop_tol );
HYPRE_Int HYPRE_BoomerAMGSetEuBJ_flt  ( HYPRE_Solver solver, HYPRE_Int eu_bj );
HYPRE_Int HYPRE_BoomerAMGSetEuBJ_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_bj );
HYPRE_Int HYPRE_BoomerAMGSetEuBJ_long_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_bj );
HYPRE_Int HYPRE_BoomerAMGSetEuclidFile_flt  ( HYPRE_Solver solver, char *euclidfile );
HYPRE_Int HYPRE_BoomerAMGSetEuclidFile_dbl  ( HYPRE_Solver solver, char *euclidfile );
HYPRE_Int HYPRE_BoomerAMGSetEuclidFile_long_dbl  ( HYPRE_Solver solver, char *euclidfile );
HYPRE_Int HYPRE_BoomerAMGSetEuLevel_flt  ( HYPRE_Solver solver, HYPRE_Int eu_level );
HYPRE_Int HYPRE_BoomerAMGSetEuLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_level );
HYPRE_Int HYPRE_BoomerAMGSetEuLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_level );
HYPRE_Int HYPRE_BoomerAMGSetEuSparseA_flt  ( HYPRE_Solver solver, hypre_float eu_sparse_A );
HYPRE_Int HYPRE_BoomerAMGSetEuSparseA_dbl  ( HYPRE_Solver solver, hypre_double eu_sparse_A );
HYPRE_Int HYPRE_BoomerAMGSetEuSparseA_long_dbl  ( HYPRE_Solver solver, hypre_long_double eu_sparse_A );
HYPRE_Int HYPRE_BoomerAMGSetFCycle_flt  ( HYPRE_Solver solver, HYPRE_Int fcycle );
HYPRE_Int HYPRE_BoomerAMGSetFCycle_dbl  ( HYPRE_Solver solver, HYPRE_Int fcycle );
HYPRE_Int HYPRE_BoomerAMGSetFCycle_long_dbl  ( HYPRE_Solver solver, HYPRE_Int fcycle );
HYPRE_Int HYPRE_BoomerAMGSetFilter_flt  ( HYPRE_Solver solver, hypre_float filter );
HYPRE_Int HYPRE_BoomerAMGSetFilter_dbl  ( HYPRE_Solver solver, hypre_double filter );
HYPRE_Int HYPRE_BoomerAMGSetFilter_long_dbl  ( HYPRE_Solver solver, hypre_long_double filter );
HYPRE_Int HYPRE_BoomerAMGSetFilterThresholdR_flt  ( HYPRE_Solver solver, hypre_float filter_threshold );
HYPRE_Int HYPRE_BoomerAMGSetFilterThresholdR_dbl  ( HYPRE_Solver solver, hypre_double filter_threshold );
HYPRE_Int HYPRE_BoomerAMGSetFilterThresholdR_long_dbl  ( HYPRE_Solver solver, hypre_long_double filter_threshold );
HYPRE_Int HYPRE_BoomerAMGSetFPoints_flt ( HYPRE_Solver solver, HYPRE_Int num_fpt,
                                     HYPRE_BigInt *fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetFPoints_dbl ( HYPRE_Solver solver, HYPRE_Int num_fpt,
                                     HYPRE_BigInt *fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetFPoints_long_dbl ( HYPRE_Solver solver, HYPRE_Int num_fpt,
                                     HYPRE_BigInt *fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetFSAIAlgoType_flt  ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAIAlgoType_dbl  ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAIAlgoType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAIEigMaxIters_flt  ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_BoomerAMGSetFSAIEigMaxIters_dbl  ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_BoomerAMGSetFSAIEigMaxIters_long_dbl  ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_BoomerAMGSetFSAIKapTolerance_flt  ( HYPRE_Solver solver, hypre_float kap_tolerance );
HYPRE_Int HYPRE_BoomerAMGSetFSAIKapTolerance_dbl  ( HYPRE_Solver solver, hypre_double kap_tolerance );
HYPRE_Int HYPRE_BoomerAMGSetFSAIKapTolerance_long_dbl  ( HYPRE_Solver solver, hypre_long_double kap_tolerance );
HYPRE_Int HYPRE_BoomerAMGSetFSAILocalSolveType_flt  ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAILocalSolveType_dbl  ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAILocalSolveType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxNnzRow_flt  ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxNnzRow_dbl  ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxNnzRow_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxSteps_flt  ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxSteps_dbl  ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxSteps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxStepSize_flt  ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxStepSize_dbl  ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_BoomerAMGSetFSAIMaxStepSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_BoomerAMGSetFSAINumLevels_flt  ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_BoomerAMGSetFSAINumLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_BoomerAMGSetFSAINumLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_BoomerAMGSetFSAIThreshold_flt  ( HYPRE_Solver solver, hypre_float threshold );
HYPRE_Int HYPRE_BoomerAMGSetFSAIThreshold_dbl  ( HYPRE_Solver solver, hypre_double threshold );
HYPRE_Int HYPRE_BoomerAMGSetFSAIThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double threshold );
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR_flt  ( HYPRE_Solver solver, HYPRE_Int gmres_switch );
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR_dbl  ( HYPRE_Solver solver, HYPRE_Int gmres_switch );
HYPRE_Int HYPRE_BoomerAMGSetGMRESSwitchR_long_dbl  ( HYPRE_Solver solver, HYPRE_Int gmres_switch );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxPoints_flt  ( HYPRE_Solver solver, HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxPoints_dbl  ( HYPRE_Solver solver, HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxPoints_long_dbl  ( HYPRE_Solver solver, HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_BoomerAMGSetGridRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_BoomerAMGSetGSMG_flt  ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetGSMG_dbl  ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetGSMG_long_dbl  ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetILUDroptol_flt ( HYPRE_Solver solver, hypre_float ilu_droptol);
HYPRE_Int HYPRE_BoomerAMGSetILUDroptol_dbl ( HYPRE_Solver solver, hypre_double ilu_droptol);
HYPRE_Int HYPRE_BoomerAMGSetILUDroptol_long_dbl ( HYPRE_Solver solver, hypre_long_double ilu_droptol);
HYPRE_Int HYPRE_BoomerAMGSetILULevel_flt ( HYPRE_Solver solver, HYPRE_Int ilu_lfil);
HYPRE_Int HYPRE_BoomerAMGSetILULevel_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_lfil);
HYPRE_Int HYPRE_BoomerAMGSetILULevel_long_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_lfil);
HYPRE_Int HYPRE_BoomerAMGSetILULocalReordering_flt ( HYPRE_Solver solver, HYPRE_Int ilu_reordering_type);
HYPRE_Int HYPRE_BoomerAMGSetILULocalReordering_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_reordering_type);
HYPRE_Int HYPRE_BoomerAMGSetILULocalReordering_long_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_reordering_type);
HYPRE_Int HYPRE_BoomerAMGSetILULowerJacobiIters_flt ( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_lower_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetILULowerJacobiIters_dbl ( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_lower_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetILULowerJacobiIters_long_dbl ( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_lower_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxIter_flt ( HYPRE_Solver solver, HYPRE_Int ilu_max_iter);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxIter_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_max_iter);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxIter_long_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_max_iter);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxRowNnz_flt ( HYPRE_Solver  solver, HYPRE_Int ilu_max_row_nnz);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxRowNnz_dbl ( HYPRE_Solver  solver, HYPRE_Int ilu_max_row_nnz);
HYPRE_Int HYPRE_BoomerAMGSetILUMaxRowNnz_long_dbl ( HYPRE_Solver  solver, HYPRE_Int ilu_max_row_nnz);
HYPRE_Int HYPRE_BoomerAMGSetILUTriSolve_flt ( HYPRE_Solver solver, HYPRE_Int ilu_tri_solve);
HYPRE_Int HYPRE_BoomerAMGSetILUTriSolve_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_tri_solve);
HYPRE_Int HYPRE_BoomerAMGSetILUTriSolve_long_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_tri_solve);
HYPRE_Int HYPRE_BoomerAMGSetILUType_flt ( HYPRE_Solver solver, HYPRE_Int ilu_type);
HYPRE_Int HYPRE_BoomerAMGSetILUType_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_type);
HYPRE_Int HYPRE_BoomerAMGSetILUType_long_dbl ( HYPRE_Solver solver, HYPRE_Int ilu_type);
HYPRE_Int HYPRE_BoomerAMGSetILUUpperJacobiIters_flt ( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_upper_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetILUUpperJacobiIters_dbl ( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_upper_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetILUUpperJacobiIters_long_dbl ( HYPRE_Solver solver,
                                                 HYPRE_Int ilu_upper_jacobi_iters);
HYPRE_Int HYPRE_BoomerAMGSetInterpRefine_flt  ( HYPRE_Solver solver, HYPRE_Int num_refine );
HYPRE_Int HYPRE_BoomerAMGSetInterpRefine_dbl  ( HYPRE_Solver solver, HYPRE_Int num_refine );
HYPRE_Int HYPRE_BoomerAMGSetInterpRefine_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_refine );
HYPRE_Int HYPRE_BoomerAMGSetInterpType_flt  ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_BoomerAMGSetInterpType_dbl  ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_BoomerAMGSetInterpType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecAbsQTrunc_flt  ( HYPRE_Solver solver, hypre_float q_trunc );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecAbsQTrunc_dbl  ( HYPRE_Solver solver, hypre_double q_trunc );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecAbsQTrunc_long_dbl  ( HYPRE_Solver solver, hypre_long_double q_trunc );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecFirstLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecFirstLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecFirstLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecQMax_flt  ( HYPRE_Solver solver, HYPRE_Int q_max );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecQMax_dbl  ( HYPRE_Solver solver, HYPRE_Int q_max );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecQMax_long_dbl  ( HYPRE_Solver solver, HYPRE_Int q_max );
HYPRE_Int HYPRE_BoomerAMGSetInterpVectors_flt  ( HYPRE_Solver solver, HYPRE_Int num_vectors,
                                            HYPRE_ParVector *vectors );
HYPRE_Int HYPRE_BoomerAMGSetInterpVectors_dbl  ( HYPRE_Solver solver, HYPRE_Int num_vectors,
                                            HYPRE_ParVector *vectors );
HYPRE_Int HYPRE_BoomerAMGSetInterpVectors_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_vectors,
                                            HYPRE_ParVector *vectors );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecVariant_flt  ( HYPRE_Solver solver, HYPRE_Int num );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecVariant_dbl  ( HYPRE_Solver solver, HYPRE_Int num );
HYPRE_Int HYPRE_BoomerAMGSetInterpVecVariant_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num );
HYPRE_Int HYPRE_BoomerAMGSetIsolatedFPoints_flt ( HYPRE_Solver solver, HYPRE_Int num_isolated_fpt,
                                             HYPRE_BigInt *isolated_fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetIsolatedFPoints_dbl ( HYPRE_Solver solver, HYPRE_Int num_isolated_fpt,
                                             HYPRE_BigInt *isolated_fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetIsolatedFPoints_long_dbl ( HYPRE_Solver solver, HYPRE_Int num_isolated_fpt,
                                             HYPRE_BigInt *isolated_fpt_index );
HYPRE_Int HYPRE_BoomerAMGSetIsTriangular_flt  ( HYPRE_Solver solver, HYPRE_Int is_triangular );
HYPRE_Int HYPRE_BoomerAMGSetIsTriangular_dbl  ( HYPRE_Solver solver, HYPRE_Int is_triangular );
HYPRE_Int HYPRE_BoomerAMGSetIsTriangular_long_dbl  ( HYPRE_Solver solver, HYPRE_Int is_triangular );
HYPRE_Int HYPRE_BoomerAMGSetISType_flt  ( HYPRE_Solver solver, HYPRE_Int IS_type );
HYPRE_Int HYPRE_BoomerAMGSetISType_dbl  ( HYPRE_Solver solver, HYPRE_Int IS_type );
HYPRE_Int HYPRE_BoomerAMGSetISType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int IS_type );
HYPRE_Int HYPRE_BoomerAMGSetJacobiTruncThreshold_flt  ( HYPRE_Solver solver,
                                                   hypre_float jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGSetJacobiTruncThreshold_dbl  ( HYPRE_Solver solver,
                                                   hypre_double jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGSetJacobiTruncThreshold_long_dbl  ( HYPRE_Solver solver,
                                                   hypre_long_double jacobi_trunc_threshold );
HYPRE_Int HYPRE_BoomerAMGSetKeepSameSign_flt  ( HYPRE_Solver solver, HYPRE_Int keep_same_sign );
HYPRE_Int HYPRE_BoomerAMGSetKeepSameSign_dbl  ( HYPRE_Solver solver, HYPRE_Int keep_same_sign );
HYPRE_Int HYPRE_BoomerAMGSetKeepSameSign_long_dbl  ( HYPRE_Solver solver, HYPRE_Int keep_same_sign );
HYPRE_Int HYPRE_BoomerAMGSetKeepTranspose_flt  ( HYPRE_Solver solver, HYPRE_Int keepTranspose );
HYPRE_Int HYPRE_BoomerAMGSetKeepTranspose_dbl  ( HYPRE_Solver solver, HYPRE_Int keepTranspose );
HYPRE_Int HYPRE_BoomerAMGSetKeepTranspose_long_dbl  ( HYPRE_Solver solver, HYPRE_Int keepTranspose );
HYPRE_Int HYPRE_BoomerAMGSetLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelNonGalerkinTol_flt  ( HYPRE_Solver solver, hypre_float nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelNonGalerkinTol_dbl  ( HYPRE_Solver solver, hypre_double nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelNonGalerkinTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelOuterWt_flt  ( HYPRE_Solver solver, hypre_float outer_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelOuterWt_dbl  ( HYPRE_Solver solver, hypre_double outer_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelOuterWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double outer_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelRelaxWt_flt  ( HYPRE_Solver solver, hypre_float relax_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelRelaxWt_dbl  ( HYPRE_Solver solver, hypre_double relax_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLevelRelaxWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double relax_wt,
                                           HYPRE_Int level );
HYPRE_Int HYPRE_BoomerAMGSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BoomerAMGSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BoomerAMGSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BoomerAMGSetMaxCoarseSize_flt  ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMaxCoarseSize_dbl  ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMaxCoarseSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BoomerAMGSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BoomerAMGSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BoomerAMGSetMaxLevels_flt  ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGSetMaxLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGSetMaxLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_BoomerAMGSetMaxNzPerRow_flt  ( HYPRE_Solver solver, HYPRE_Int max_nz_per_row );
HYPRE_Int HYPRE_BoomerAMGSetMaxNzPerRow_dbl  ( HYPRE_Solver solver, HYPRE_Int max_nz_per_row );
HYPRE_Int HYPRE_BoomerAMGSetMaxNzPerRow_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_nz_per_row );
HYPRE_Int HYPRE_BoomerAMGSetMaxRowSum_flt  ( HYPRE_Solver solver, hypre_float max_row_sum );
HYPRE_Int HYPRE_BoomerAMGSetMaxRowSum_dbl  ( HYPRE_Solver solver, hypre_double max_row_sum );
HYPRE_Int HYPRE_BoomerAMGSetMaxRowSum_long_dbl  ( HYPRE_Solver solver, hypre_long_double max_row_sum );
HYPRE_Int HYPRE_BoomerAMGSetMeasureType_flt  ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_BoomerAMGSetMeasureType_dbl  ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_BoomerAMGSetMeasureType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_BoomerAMGSetMinCoarseSize_flt  ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMinCoarseSize_dbl  ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMinCoarseSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_BoomerAMGSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BoomerAMGSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BoomerAMGSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BoomerAMGSetModuleRAP2_flt  ( HYPRE_Solver solver, HYPRE_Int mod_rap2 );
HYPRE_Int HYPRE_BoomerAMGSetModuleRAP2_dbl  ( HYPRE_Solver solver, HYPRE_Int mod_rap2 );
HYPRE_Int HYPRE_BoomerAMGSetModuleRAP2_long_dbl  ( HYPRE_Solver solver, HYPRE_Int mod_rap2 );
HYPRE_Int HYPRE_BoomerAMGSetMultAdditive_flt  ( HYPRE_Solver solver, HYPRE_Int mult_additive );
HYPRE_Int HYPRE_BoomerAMGSetMultAdditive_dbl  ( HYPRE_Solver solver, HYPRE_Int mult_additive );
HYPRE_Int HYPRE_BoomerAMGSetMultAdditive_long_dbl  ( HYPRE_Solver solver, HYPRE_Int mult_additive );
HYPRE_Int HYPRE_BoomerAMGSetMultAddPMaxElmts_flt  ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetMultAddPMaxElmts_dbl  ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetMultAddPMaxElmts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int add_P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetMultAddTruncFactor_flt  ( HYPRE_Solver solver, hypre_float add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetMultAddTruncFactor_dbl  ( HYPRE_Solver solver, hypre_double add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetMultAddTruncFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double add_trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetNodal_flt  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodal_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodal_long_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodalDiag_flt  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodalDiag_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodalDiag_long_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_BoomerAMGSetNodalLevels_flt  ( HYPRE_Solver solver, HYPRE_Int nodal_levels );
HYPRE_Int HYPRE_BoomerAMGSetNodalLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal_levels );
HYPRE_Int HYPRE_BoomerAMGSetNodalLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal_levels );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkinTol_flt  ( HYPRE_Solver solver, hypre_float nongalerkin_tol );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkinTol_dbl  ( HYPRE_Solver solver, hypre_double nongalerkin_tol );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkinTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double nongalerkin_tol );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkTol_flt  ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                           hypre_float *nongalerk_tol );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkTol_dbl  ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                           hypre_double *nongalerk_tol );
HYPRE_Int HYPRE_BoomerAMGSetNonGalerkTol_long_dbl  ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                           hypre_long_double *nongalerk_tol );
HYPRE_Int HYPRE_BoomerAMGSetNumCRRelaxSteps_flt  ( HYPRE_Solver solver, HYPRE_Int num_CR_relax_steps );
HYPRE_Int HYPRE_BoomerAMGSetNumCRRelaxSteps_dbl  ( HYPRE_Solver solver, HYPRE_Int num_CR_relax_steps );
HYPRE_Int HYPRE_BoomerAMGSetNumCRRelaxSteps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_CR_relax_steps );
HYPRE_Int HYPRE_BoomerAMGSetNumFunctions_flt  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_BoomerAMGSetNumFunctions_dbl  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_BoomerAMGSetNumFunctions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_BoomerAMGSetNumGridSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetNumGridSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetNumGridSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetNumPaths_flt  ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_BoomerAMGSetNumPaths_dbl  ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_BoomerAMGSetNumPaths_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_BoomerAMGSetNumSamples_flt  ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetNumSamples_dbl  ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetNumSamples_long_dbl  ( HYPRE_Solver solver, HYPRE_Int gsmg );
HYPRE_Int HYPRE_BoomerAMGSetNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetOldDefault_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGSetOldDefault_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGSetOldDefault_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BoomerAMGSetOmega_flt  ( HYPRE_Solver solver, hypre_float *omega );
HYPRE_Int HYPRE_BoomerAMGSetOmega_dbl  ( HYPRE_Solver solver, hypre_double *omega );
HYPRE_Int HYPRE_BoomerAMGSetOmega_long_dbl  ( HYPRE_Solver solver, hypre_long_double *omega );
HYPRE_Int HYPRE_BoomerAMGSetOuterWt_flt  ( HYPRE_Solver solver, hypre_float outer_wt );
HYPRE_Int HYPRE_BoomerAMGSetOuterWt_dbl  ( HYPRE_Solver solver, hypre_double outer_wt );
HYPRE_Int HYPRE_BoomerAMGSetOuterWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double outer_wt );
HYPRE_Int HYPRE_BoomerAMGSetOverlap_flt  ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_BoomerAMGSetOverlap_dbl  ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_BoomerAMGSetOverlap_long_dbl  ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_BoomerAMGSetPlotFileName_flt  ( HYPRE_Solver solver, const char *plotfilename );
HYPRE_Int HYPRE_BoomerAMGSetPlotFileName_dbl  ( HYPRE_Solver solver, const char *plotfilename );
HYPRE_Int HYPRE_BoomerAMGSetPlotFileName_long_dbl  ( HYPRE_Solver solver, const char *plotfilename );
HYPRE_Int HYPRE_BoomerAMGSetPlotGrids_flt  ( HYPRE_Solver solver, HYPRE_Int plotgrids );
HYPRE_Int HYPRE_BoomerAMGSetPlotGrids_dbl  ( HYPRE_Solver solver, HYPRE_Int plotgrids );
HYPRE_Int HYPRE_BoomerAMGSetPlotGrids_long_dbl  ( HYPRE_Solver solver, HYPRE_Int plotgrids );
HYPRE_Int HYPRE_BoomerAMGSetPMaxElmts_flt  ( HYPRE_Solver solver, HYPRE_Int P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetPMaxElmts_dbl  ( HYPRE_Solver solver, HYPRE_Int P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetPMaxElmts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int P_max_elmts );
HYPRE_Int HYPRE_BoomerAMGSetPostInterpType_flt  ( HYPRE_Solver solver, HYPRE_Int post_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetPostInterpType_dbl  ( HYPRE_Solver solver, HYPRE_Int post_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetPostInterpType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int post_interp_type );
HYPRE_Int HYPRE_BoomerAMGSetPrintFileName_flt  ( HYPRE_Solver solver, const char *print_file_name );
HYPRE_Int HYPRE_BoomerAMGSetPrintFileName_dbl  ( HYPRE_Solver solver, const char *print_file_name );
HYPRE_Int HYPRE_BoomerAMGSetPrintFileName_long_dbl  ( HYPRE_Solver solver, const char *print_file_name );
HYPRE_Int HYPRE_BoomerAMGSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BoomerAMGSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BoomerAMGSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BoomerAMGSetRAP2_flt  ( HYPRE_Solver solver, HYPRE_Int rap2 );
HYPRE_Int HYPRE_BoomerAMGSetRAP2_dbl  ( HYPRE_Solver solver, HYPRE_Int rap2 );
HYPRE_Int HYPRE_BoomerAMGSetRAP2_long_dbl  ( HYPRE_Solver solver, HYPRE_Int rap2 );
HYPRE_Int HYPRE_BoomerAMGSetRedundant_flt  ( HYPRE_Solver solver, HYPRE_Int redundant );
HYPRE_Int HYPRE_BoomerAMGSetRedundant_dbl  ( HYPRE_Solver solver, HYPRE_Int redundant );
HYPRE_Int HYPRE_BoomerAMGSetRedundant_long_dbl  ( HYPRE_Solver solver, HYPRE_Int redundant );
HYPRE_Int HYPRE_BoomerAMGSetRelaxOrder_flt  ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_BoomerAMGSetRelaxOrder_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_BoomerAMGSetRelaxOrder_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_BoomerAMGSetRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BoomerAMGSetRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BoomerAMGSetRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWeight_flt  ( HYPRE_Solver solver, hypre_float *relax_weight );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWeight_dbl  ( HYPRE_Solver solver, hypre_double *relax_weight );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWeight_long_dbl  ( HYPRE_Solver solver, hypre_long_double *relax_weight );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWt_flt  ( HYPRE_Solver solver, hypre_float relax_wt );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWt_dbl  ( HYPRE_Solver solver, hypre_double relax_wt );
HYPRE_Int HYPRE_BoomerAMGSetRelaxWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double relax_wt );
HYPRE_Int HYPRE_BoomerAMGSetRestriction_flt  ( HYPRE_Solver solver, HYPRE_Int restr_par );
HYPRE_Int HYPRE_BoomerAMGSetRestriction_dbl  ( HYPRE_Solver solver, HYPRE_Int restr_par );
HYPRE_Int HYPRE_BoomerAMGSetRestriction_long_dbl  ( HYPRE_Solver solver, HYPRE_Int restr_par );
HYPRE_Int HYPRE_BoomerAMGSetSabs_flt  ( HYPRE_Solver solver, HYPRE_Int Sabs );
HYPRE_Int HYPRE_BoomerAMGSetSabs_dbl  ( HYPRE_Solver solver, HYPRE_Int Sabs );
HYPRE_Int HYPRE_BoomerAMGSetSabs_long_dbl  ( HYPRE_Solver solver, HYPRE_Int Sabs );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzRlxWeight_flt  ( HYPRE_Solver solver, hypre_float schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzRlxWeight_dbl  ( HYPRE_Solver solver, hypre_double schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzRlxWeight_long_dbl  ( HYPRE_Solver solver, hypre_long_double schwarz_rlx_weight );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzUseNonSymm_flt  ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzUseNonSymm_dbl  ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_BoomerAMGSetSchwarzUseNonSymm_long_dbl  ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_BoomerAMGSetSCommPkgSwitch_flt  ( HYPRE_Solver solver, hypre_float S_commpkg_switch );
HYPRE_Int HYPRE_BoomerAMGSetSCommPkgSwitch_dbl  ( HYPRE_Solver solver, hypre_double S_commpkg_switch );
HYPRE_Int HYPRE_BoomerAMGSetSCommPkgSwitch_long_dbl  ( HYPRE_Solver solver, hypre_long_double S_commpkg_switch );
HYPRE_Int HYPRE_BoomerAMGSetSepWeight_flt  ( HYPRE_Solver solver, HYPRE_Int sep_weight );
HYPRE_Int HYPRE_BoomerAMGSetSepWeight_dbl  ( HYPRE_Solver solver, HYPRE_Int sep_weight );
HYPRE_Int HYPRE_BoomerAMGSetSepWeight_long_dbl  ( HYPRE_Solver solver, HYPRE_Int sep_weight );
HYPRE_Int HYPRE_BoomerAMGSetSeqThreshold_flt  ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_BoomerAMGSetSeqThreshold_dbl  ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_BoomerAMGSetSeqThreshold_long_dbl  ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_BoomerAMGSetSetupType_flt  ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_BoomerAMGSetSetupType_dbl  ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_BoomerAMGSetSetupType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_BoomerAMGSetSimple_flt  ( HYPRE_Solver solver, HYPRE_Int simple );
HYPRE_Int HYPRE_BoomerAMGSetSimple_dbl  ( HYPRE_Solver solver, HYPRE_Int simple );
HYPRE_Int HYPRE_BoomerAMGSetSimple_long_dbl  ( HYPRE_Solver solver, HYPRE_Int simple );
HYPRE_Int HYPRE_BoomerAMGSetSmoothInterpVectors_flt  ( HYPRE_Solver solver,
                                                  HYPRE_Int smooth_interp_vectors );
HYPRE_Int HYPRE_BoomerAMGSetSmoothInterpVectors_dbl  ( HYPRE_Solver solver,
                                                  HYPRE_Int smooth_interp_vectors );
HYPRE_Int HYPRE_BoomerAMGSetSmoothInterpVectors_long_dbl  ( HYPRE_Solver solver,
                                                  HYPRE_Int smooth_interp_vectors );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumLevels_flt  ( HYPRE_Solver solver, HYPRE_Int smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int smooth_num_levels );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetSmoothNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int smooth_num_sweeps );
HYPRE_Int HYPRE_BoomerAMGSetSmoothType_flt  ( HYPRE_Solver solver, HYPRE_Int smooth_type );
HYPRE_Int HYPRE_BoomerAMGSetSmoothType_dbl  ( HYPRE_Solver solver, HYPRE_Int smooth_type );
HYPRE_Int HYPRE_BoomerAMGSetSmoothType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int smooth_type );
HYPRE_Int HYPRE_BoomerAMGSetStrongThreshold_flt  ( HYPRE_Solver solver, hypre_float strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetStrongThreshold_dbl  ( HYPRE_Solver solver, hypre_double strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetStrongThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetStrongThresholdR_flt  ( HYPRE_Solver solver, hypre_float strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetStrongThresholdR_dbl  ( HYPRE_Solver solver, hypre_double strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetStrongThresholdR_long_dbl  ( HYPRE_Solver solver, hypre_long_double strong_threshold );
HYPRE_Int HYPRE_BoomerAMGSetSym_flt  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_BoomerAMGSetSym_dbl  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_BoomerAMGSetSym_long_dbl  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_BoomerAMGSetThreshold_flt  ( HYPRE_Solver solver, hypre_float threshold );
HYPRE_Int HYPRE_BoomerAMGSetThreshold_dbl  ( HYPRE_Solver solver, hypre_double threshold );
HYPRE_Int HYPRE_BoomerAMGSetThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double threshold );
HYPRE_Int HYPRE_BoomerAMGSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_BoomerAMGSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_BoomerAMGSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_BoomerAMGSetTruncFactor_flt  ( HYPRE_Solver solver, hypre_float trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetTruncFactor_dbl  ( HYPRE_Solver solver, hypre_double trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetTruncFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double trunc_factor );
HYPRE_Int HYPRE_BoomerAMGSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSetVariant_flt  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSetVariant_dbl  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSetVariant_long_dbl  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_BoomerAMGSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolveT_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolveT_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_BoomerAMGSolveT_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver,
                                                             hypre_float *norm );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver,
                                                             hypre_double *norm );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver,
                                                             hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetResidual_flt  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetResidual_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRBiCGSTABGetResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                           HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                           HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                           HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRBiCGSTABSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                      HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagCreate_flt  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BlockTridiagCreate_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BlockTridiagCreate_long_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_BlockTridiagDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BlockTridiagDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BlockTridiagDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BlockTridiagSetAMGNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BlockTridiagSetAMGNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BlockTridiagSetAMGNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_BlockTridiagSetAMGRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BlockTridiagSetAMGRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BlockTridiagSetAMGRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_BlockTridiagSetAMGStrengthThreshold_flt  ( HYPRE_Solver solver, hypre_float thresh );
HYPRE_Int HYPRE_BlockTridiagSetAMGStrengthThreshold_dbl  ( HYPRE_Solver solver, hypre_double thresh );
HYPRE_Int HYPRE_BlockTridiagSetAMGStrengthThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double thresh );
HYPRE_Int HYPRE_BlockTridiagSetIndexSet_flt  ( HYPRE_Solver solver, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int HYPRE_BlockTridiagSetIndexSet_dbl  ( HYPRE_Solver solver, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int HYPRE_BlockTridiagSetIndexSet_long_dbl  ( HYPRE_Solver solver, HYPRE_Int n, HYPRE_Int *inds );
HYPRE_Int HYPRE_BlockTridiagSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BlockTridiagSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BlockTridiagSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BlockTridiagSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_BlockTridiagSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCGNRCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCGNRCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCGNRDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCGNRDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCGNRDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRCGNRGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCGNRGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCGNRGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCGNRGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCGNRGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCGNRGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCGNRSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCGNRSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCGNRSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCGNRSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCGNRSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                       HYPRE_PtrToParSolverFcn precondT, HYPRE_PtrToParSolverFcn precond_setup,
                                       HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCGNRSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                       HYPRE_PtrToParSolverFcn precondT, HYPRE_PtrToParSolverFcn precond_setup,
                                       HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCGNRSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                       HYPRE_PtrToParSolverFcn precondT, HYPRE_PtrToParSolverFcn precond_setup,
                                       HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCGNRSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRCGNRSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRCGNRSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRCGNRSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRCGNRSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRCGNRSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRCGNRSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCGNRSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                  HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCOGMRESCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCOGMRESCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRCOGMRESDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCOGMRESDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCOGMRESDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRCOGMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCOGMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCOGMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRCOGMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCOGMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCOGMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRCOGMRESGetResidual_flt  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRCOGMRESGetResidual_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRCOGMRESGetResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRCOGMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRCOGMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRCOGMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRCOGMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCOGMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCOGMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                          HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                          HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                          HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRCOGMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRCOGMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRCOGMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRCOGMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                     HYPRE_ParVector x );
HYPRE_Int HYPRE_EuclidCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_EuclidCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_EuclidCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_EuclidDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_EuclidDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_EuclidDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_EuclidSetBJ_flt  ( HYPRE_Solver solver, HYPRE_Int bj );
HYPRE_Int HYPRE_EuclidSetBJ_dbl  ( HYPRE_Solver solver, HYPRE_Int bj );
HYPRE_Int HYPRE_EuclidSetBJ_long_dbl  ( HYPRE_Solver solver, HYPRE_Int bj );
HYPRE_Int HYPRE_EuclidSetILUT_flt  ( HYPRE_Solver solver, hypre_float ilut );
HYPRE_Int HYPRE_EuclidSetILUT_dbl  ( HYPRE_Solver solver, hypre_double ilut );
HYPRE_Int HYPRE_EuclidSetILUT_long_dbl  ( HYPRE_Solver solver, hypre_long_double ilut );
HYPRE_Int HYPRE_EuclidSetLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_EuclidSetLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_EuclidSetLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_EuclidSetMem_flt  ( HYPRE_Solver solver, HYPRE_Int eu_mem );
HYPRE_Int HYPRE_EuclidSetMem_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_mem );
HYPRE_Int HYPRE_EuclidSetMem_long_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_mem );
HYPRE_Int HYPRE_EuclidSetParams_flt  ( HYPRE_Solver solver, HYPRE_Int argc, char *argv []);
HYPRE_Int HYPRE_EuclidSetParams_dbl  ( HYPRE_Solver solver, HYPRE_Int argc, char *argv []);
HYPRE_Int HYPRE_EuclidSetParams_long_dbl  ( HYPRE_Solver solver, HYPRE_Int argc, char *argv []);
HYPRE_Int HYPRE_EuclidSetParamsFromFile_flt  ( HYPRE_Solver solver, char *filename );
HYPRE_Int HYPRE_EuclidSetParamsFromFile_dbl  ( HYPRE_Solver solver, char *filename );
HYPRE_Int HYPRE_EuclidSetParamsFromFile_long_dbl  ( HYPRE_Solver solver, char *filename );
HYPRE_Int HYPRE_EuclidSetRowScale_flt  ( HYPRE_Solver solver, HYPRE_Int row_scale );
HYPRE_Int HYPRE_EuclidSetRowScale_dbl  ( HYPRE_Solver solver, HYPRE_Int row_scale );
HYPRE_Int HYPRE_EuclidSetRowScale_long_dbl  ( HYPRE_Solver solver, HYPRE_Int row_scale );
HYPRE_Int HYPRE_EuclidSetSparseA_flt  ( HYPRE_Solver solver, hypre_float sparse_A );
HYPRE_Int HYPRE_EuclidSetSparseA_dbl  ( HYPRE_Solver solver, hypre_double sparse_A );
HYPRE_Int HYPRE_EuclidSetSparseA_long_dbl  ( HYPRE_Solver solver, hypre_long_double sparse_A );
HYPRE_Int HYPRE_EuclidSetStats_flt  ( HYPRE_Solver solver, HYPRE_Int eu_stats );
HYPRE_Int HYPRE_EuclidSetStats_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_stats );
HYPRE_Int HYPRE_EuclidSetStats_long_dbl  ( HYPRE_Solver solver, HYPRE_Int eu_stats );
HYPRE_Int HYPRE_EuclidSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x );
HYPRE_Int HYPRE_EuclidSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x );
HYPRE_Int HYPRE_EuclidSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                              HYPRE_ParVector x );
HYPRE_Int HYPRE_EuclidSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector bb,
                              HYPRE_ParVector xx );
HYPRE_Int HYPRE_EuclidSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector bb,
                              HYPRE_ParVector xx );
HYPRE_Int HYPRE_EuclidSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector bb,
                              HYPRE_ParVector xx );
HYPRE_Int HYPRE_ParCSRFlexGMRESCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver,
                                                              hypre_float *norm );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver,
                                                              hypre_double *norm );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver,
                                                              hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetResidual_flt  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetResidual_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRFlexGMRESGetResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC_flt  ( HYPRE_Solver solver,
                                             HYPRE_PtrToModifyPCFcn modify_pc );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC_dbl  ( HYPRE_Solver solver,
                                             HYPRE_PtrToModifyPCFcn modify_pc );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC_long_dbl  ( HYPRE_Solver solver,
                                             HYPRE_PtrToModifyPCFcn modify_pc );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                            HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                            HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                            HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRFlexGMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAICreate_flt  ( HYPRE_Solver *solver);
HYPRE_Int HYPRE_FSAICreate_dbl  ( HYPRE_Solver *solver);
HYPRE_Int HYPRE_FSAICreate_long_dbl  ( HYPRE_Solver *solver);
HYPRE_Int HYPRE_FSAIDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_FSAIDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_FSAIDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_FSAIGetAlgoType_flt  ( HYPRE_Solver solver, HYPRE_Int *algo_type );
HYPRE_Int HYPRE_FSAIGetAlgoType_dbl  ( HYPRE_Solver solver, HYPRE_Int *algo_type );
HYPRE_Int HYPRE_FSAIGetAlgoType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *algo_type );
HYPRE_Int HYPRE_FSAIGetEigMaxIters_flt  ( HYPRE_Solver solver, HYPRE_Int *eig_max_iters );
HYPRE_Int HYPRE_FSAIGetEigMaxIters_dbl  ( HYPRE_Solver solver, HYPRE_Int *eig_max_iters );
HYPRE_Int HYPRE_FSAIGetEigMaxIters_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *eig_max_iters );
HYPRE_Int HYPRE_FSAIGetKapTolerance_flt  ( HYPRE_Solver solver, hypre_float *kap_tolerance );
HYPRE_Int HYPRE_FSAIGetKapTolerance_dbl  ( HYPRE_Solver solver, hypre_double *kap_tolerance );
HYPRE_Int HYPRE_FSAIGetKapTolerance_long_dbl  ( HYPRE_Solver solver, hypre_long_double *kap_tolerance );
HYPRE_Int HYPRE_FSAIGetLocalSolveType_flt  ( HYPRE_Solver solver, HYPRE_Int *local_solve_type );
HYPRE_Int HYPRE_FSAIGetLocalSolveType_dbl  ( HYPRE_Solver solver, HYPRE_Int *local_solve_type );
HYPRE_Int HYPRE_FSAIGetLocalSolveType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *local_solve_type );
HYPRE_Int HYPRE_FSAIGetMaxIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *max_iterations );
HYPRE_Int HYPRE_FSAIGetMaxIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iterations );
HYPRE_Int HYPRE_FSAIGetMaxIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iterations );
HYPRE_Int HYPRE_FSAIGetMaxNnzRow_flt  ( HYPRE_Solver solver, HYPRE_Int *max_nnz_row );
HYPRE_Int HYPRE_FSAIGetMaxNnzRow_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_nnz_row );
HYPRE_Int HYPRE_FSAIGetMaxNnzRow_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_nnz_row );
HYPRE_Int HYPRE_FSAIGetMaxSteps_flt  ( HYPRE_Solver solver, HYPRE_Int *max_steps );
HYPRE_Int HYPRE_FSAIGetMaxSteps_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_steps );
HYPRE_Int HYPRE_FSAIGetMaxSteps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_steps );
HYPRE_Int HYPRE_FSAIGetMaxStepSize_flt  ( HYPRE_Solver solver, HYPRE_Int *max_step_size );
HYPRE_Int HYPRE_FSAIGetMaxStepSize_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_step_size );
HYPRE_Int HYPRE_FSAIGetMaxStepSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_step_size );
HYPRE_Int HYPRE_FSAIGetNumLevels_flt  ( HYPRE_Solver solver, HYPRE_Int *num_levels );
HYPRE_Int HYPRE_FSAIGetNumLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_levels );
HYPRE_Int HYPRE_FSAIGetNumLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_levels );
HYPRE_Int HYPRE_FSAIGetOmega_flt  ( HYPRE_Solver solver, hypre_float *omega );
HYPRE_Int HYPRE_FSAIGetOmega_dbl  ( HYPRE_Solver solver, hypre_double *omega );
HYPRE_Int HYPRE_FSAIGetOmega_long_dbl  ( HYPRE_Solver solver, hypre_long_double *omega );
HYPRE_Int HYPRE_FSAIGetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *print_level );
HYPRE_Int HYPRE_FSAIGetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *print_level );
HYPRE_Int HYPRE_FSAIGetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *print_level );
HYPRE_Int HYPRE_FSAIGetThreshold_flt  ( HYPRE_Solver solver, hypre_float *threshold );
HYPRE_Int HYPRE_FSAIGetThreshold_dbl  ( HYPRE_Solver solver, hypre_double *threshold );
HYPRE_Int HYPRE_FSAIGetThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double *threshold );
HYPRE_Int HYPRE_FSAIGetTolerance_flt  ( HYPRE_Solver solver, hypre_float *tolerance );
HYPRE_Int HYPRE_FSAIGetTolerance_dbl  ( HYPRE_Solver solver, hypre_double *tolerance );
HYPRE_Int HYPRE_FSAIGetTolerance_long_dbl  ( HYPRE_Solver solver, hypre_long_double *tolerance );
HYPRE_Int HYPRE_FSAIGetZeroGuess_flt  ( HYPRE_Solver solver, HYPRE_Int *zero_guess );
HYPRE_Int HYPRE_FSAIGetZeroGuess_dbl  ( HYPRE_Solver solver, HYPRE_Int *zero_guess );
HYPRE_Int HYPRE_FSAIGetZeroGuess_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *zero_guess );
HYPRE_Int HYPRE_FSAISetAlgoType_flt  ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_FSAISetAlgoType_dbl  ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_FSAISetAlgoType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int algo_type );
HYPRE_Int HYPRE_FSAISetEigMaxIters_flt  ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_FSAISetEigMaxIters_dbl  ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_FSAISetEigMaxIters_long_dbl  ( HYPRE_Solver solver, HYPRE_Int eig_max_iters );
HYPRE_Int HYPRE_FSAISetKapTolerance_flt  ( HYPRE_Solver solver, hypre_float kap_tolerance );
HYPRE_Int HYPRE_FSAISetKapTolerance_dbl  ( HYPRE_Solver solver, hypre_double kap_tolerance );
HYPRE_Int HYPRE_FSAISetKapTolerance_long_dbl  ( HYPRE_Solver solver, hypre_long_double kap_tolerance );
HYPRE_Int HYPRE_FSAISetLocalSolveType_flt  ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_FSAISetLocalSolveType_dbl  ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_FSAISetLocalSolveType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int local_solve_type );
HYPRE_Int HYPRE_FSAISetMaxIterations_flt  ( HYPRE_Solver solver, HYPRE_Int max_iterations );
HYPRE_Int HYPRE_FSAISetMaxIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iterations );
HYPRE_Int HYPRE_FSAISetMaxIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iterations );
HYPRE_Int HYPRE_FSAISetMaxNnzRow_flt  ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_FSAISetMaxNnzRow_dbl  ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_FSAISetMaxNnzRow_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_nnz_row );
HYPRE_Int HYPRE_FSAISetMaxSteps_flt  ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_FSAISetMaxSteps_dbl  ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_FSAISetMaxSteps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_steps );
HYPRE_Int HYPRE_FSAISetMaxStepSize_flt  ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_FSAISetMaxStepSize_dbl  ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_FSAISetMaxStepSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_step_size );
HYPRE_Int HYPRE_FSAISetNumLevels_flt  ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_FSAISetNumLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_FSAISetNumLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_levels );
HYPRE_Int HYPRE_FSAISetOmega_flt  ( HYPRE_Solver solver, hypre_float omega );
HYPRE_Int HYPRE_FSAISetOmega_dbl  ( HYPRE_Solver solver, hypre_double omega );
HYPRE_Int HYPRE_FSAISetOmega_long_dbl  ( HYPRE_Solver solver, hypre_long_double omega );
HYPRE_Int HYPRE_FSAISetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_FSAISetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_FSAISetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_FSAISetThreshold_flt  ( HYPRE_Solver solver, hypre_float threshold );
HYPRE_Int HYPRE_FSAISetThreshold_dbl  ( HYPRE_Solver solver, hypre_double threshold );
HYPRE_Int HYPRE_FSAISetThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double threshold );
HYPRE_Int HYPRE_FSAISetTolerance_flt  ( HYPRE_Solver solver, hypre_float tolerance );
HYPRE_Int HYPRE_FSAISetTolerance_dbl  ( HYPRE_Solver solver, hypre_double tolerance );
HYPRE_Int HYPRE_FSAISetTolerance_long_dbl  ( HYPRE_Solver solver, hypre_long_double tolerance );
HYPRE_Int HYPRE_FSAISetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAISetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAISetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAISetZeroGuess_flt  ( HYPRE_Solver solver, HYPRE_Int zero_guess );
HYPRE_Int HYPRE_FSAISetZeroGuess_dbl  ( HYPRE_Solver solver, HYPRE_Int zero_guess );
HYPRE_Int HYPRE_FSAISetZeroGuess_long_dbl  ( HYPRE_Solver solver, HYPRE_Int zero_guess );
HYPRE_Int HYPRE_FSAISolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAISolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_FSAISolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                            HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRGMRESCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRGMRESCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRGMRESDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRGMRESDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRGMRESDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRGMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRGMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRGMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRGMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRGMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRGMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRGMRESGetResidual_flt  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRGMRESGetResidual_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRGMRESGetResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRGMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_ParCSRGMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_ParCSRGMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_ParCSRGMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRGMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRGMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRGMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRGMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRGMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRGMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRGMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                        HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRGMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                        HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRGMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                        HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRGMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRGMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRGMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRGMRESSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRGMRESSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRGMRESSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRGMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRGMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRGMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRGMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRGMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSROnProcTriSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSROnProcTriSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA,
                                       HYPRE_ParVector Hy, HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSRHybridCreate_flt  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRHybridCreate_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRHybridCreate_long_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRHybridDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRHybridDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRHybridDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRHybridGetDSCGNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *dscg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetDSCGNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *dscg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetDSCGNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *dscg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRHybridGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_its );
HYPRE_Int HYPRE_ParCSRHybridGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_its );
HYPRE_Int HYPRE_ParCSRHybridGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_its );
HYPRE_Int HYPRE_ParCSRHybridGetPCGNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *pcg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetPCGNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *pcg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetPCGNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *pcg_num_its );
HYPRE_Int HYPRE_ParCSRHybridGetSetupSolveTime_flt ( HYPRE_Solver solver, hypre_float *time );
HYPRE_Int HYPRE_ParCSRHybridGetSetupSolveTime_dbl ( HYPRE_Solver solver, hypre_double *time );
HYPRE_Int HYPRE_ParCSRHybridGetSetupSolveTime_long_dbl ( HYPRE_Solver solver, hypre_long_double *time );
HYPRE_Int HYPRE_ParCSRHybridSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRHybridSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRHybridSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRHybridSetAggNumLevels_flt  ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_ParCSRHybridSetAggNumLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_ParCSRHybridSetAggNumLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int agg_num_levels );
HYPRE_Int HYPRE_ParCSRHybridSetCoarsenType_flt  ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_ParCSRHybridSetCoarsenType_dbl  ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_ParCSRHybridSetCoarsenType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int coarsen_type );
HYPRE_Int HYPRE_ParCSRHybridSetConvergenceTol_flt  ( HYPRE_Solver solver, hypre_float cf_tol );
HYPRE_Int HYPRE_ParCSRHybridSetConvergenceTol_dbl  ( HYPRE_Solver solver, hypre_double cf_tol );
HYPRE_Int HYPRE_ParCSRHybridSetConvergenceTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double cf_tol );
HYPRE_Int HYPRE_ParCSRHybridSetCycleNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetCycleNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetCycleNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetCycleRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetCycleRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetCycleRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type,
                                                HYPRE_Int k );
HYPRE_Int HYPRE_ParCSRHybridSetCycleType_flt  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ParCSRHybridSetCycleType_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ParCSRHybridSetCycleType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cycle_type );
HYPRE_Int HYPRE_ParCSRHybridSetDofFunc_flt  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_ParCSRHybridSetDofFunc_dbl  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_ParCSRHybridSetDofFunc_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_ParCSRHybridSetDSCGMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int dscg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetDSCGMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int dscg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetDSCGMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int dscg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxPoints_flt  ( HYPRE_Solver solver,
                                                 HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxPoints_dbl  ( HYPRE_Solver solver,
                                                 HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxPoints_long_dbl  ( HYPRE_Solver solver,
                                                 HYPRE_Int **grid_relax_points );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetGridRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *grid_relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetInterpType_flt  ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_ParCSRHybridSetInterpType_dbl  ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_ParCSRHybridSetInterpType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int interp_type );
HYPRE_Int HYPRE_ParCSRHybridSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRHybridSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRHybridSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRHybridSetKeepTranspose_flt  ( HYPRE_Solver solver, HYPRE_Int keepT );
HYPRE_Int HYPRE_ParCSRHybridSetKeepTranspose_dbl  ( HYPRE_Solver solver, HYPRE_Int keepT );
HYPRE_Int HYPRE_ParCSRHybridSetKeepTranspose_long_dbl  ( HYPRE_Solver solver, HYPRE_Int keepT );
HYPRE_Int HYPRE_ParCSRHybridSetLevelOuterWt_flt  ( HYPRE_Solver solver, hypre_float outer_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetLevelOuterWt_dbl  ( HYPRE_Solver solver, hypre_double outer_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetLevelOuterWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double outer_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetLevelRelaxWt_flt  ( HYPRE_Solver solver, hypre_float relax_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetLevelRelaxWt_dbl  ( HYPRE_Solver solver, hypre_double relax_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetLevelRelaxWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double relax_wt,
                                              HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRHybridSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRHybridSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRHybridSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRHybridSetMaxCoarseSize_flt  ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetMaxCoarseSize_dbl  ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetMaxCoarseSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetMaxLevels_flt  ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_ParCSRHybridSetMaxLevels_dbl  ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_ParCSRHybridSetMaxLevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_levels );
HYPRE_Int HYPRE_ParCSRHybridSetMaxRowSum_flt  ( HYPRE_Solver solver, hypre_float max_row_sum );
HYPRE_Int HYPRE_ParCSRHybridSetMaxRowSum_dbl  ( HYPRE_Solver solver, hypre_double max_row_sum );
HYPRE_Int HYPRE_ParCSRHybridSetMaxRowSum_long_dbl  ( HYPRE_Solver solver, hypre_long_double max_row_sum );
HYPRE_Int HYPRE_ParCSRHybridSetMeasureType_flt  ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_ParCSRHybridSetMeasureType_dbl  ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_ParCSRHybridSetMeasureType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int measure_type );
HYPRE_Int HYPRE_ParCSRHybridSetMinCoarseSize_flt  ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetMinCoarseSize_dbl  ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetMinCoarseSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_coarse_size );
HYPRE_Int HYPRE_ParCSRHybridSetNodal_flt  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_ParCSRHybridSetNodal_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_ParCSRHybridSetNodal_long_dbl  ( HYPRE_Solver solver, HYPRE_Int nodal );
HYPRE_Int HYPRE_ParCSRHybridSetNonGalerkinTol_flt  ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                                hypre_float *nongalerkin_tol );
HYPRE_Int HYPRE_ParCSRHybridSetNonGalerkinTol_dbl  ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                                hypre_double *nongalerkin_tol );
HYPRE_Int HYPRE_ParCSRHybridSetNonGalerkinTol_long_dbl  ( HYPRE_Solver solver, HYPRE_Int nongalerk_num_tol,
                                                hypre_long_double *nongalerkin_tol );
HYPRE_Int HYPRE_ParCSRHybridSetNumFunctions_flt  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_ParCSRHybridSetNumFunctions_dbl  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_ParCSRHybridSetNumFunctions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_ParCSRHybridSetNumGridSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetNumGridSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetNumGridSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_grid_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetNumPaths_flt  ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_ParCSRHybridSetNumPaths_dbl  ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_ParCSRHybridSetNumPaths_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_paths );
HYPRE_Int HYPRE_ParCSRHybridSetNumSweeps_flt  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetNumSweeps_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetNumSweeps_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_sweeps );
HYPRE_Int HYPRE_ParCSRHybridSetOmega_flt  ( HYPRE_Solver solver, hypre_float *omega );
HYPRE_Int HYPRE_ParCSRHybridSetOmega_dbl  ( HYPRE_Solver solver, hypre_double *omega );
HYPRE_Int HYPRE_ParCSRHybridSetOmega_long_dbl  ( HYPRE_Solver solver, hypre_long_double *omega );
HYPRE_Int HYPRE_ParCSRHybridSetOuterWt_flt  ( HYPRE_Solver solver, hypre_float outer_wt );
HYPRE_Int HYPRE_ParCSRHybridSetOuterWt_dbl  ( HYPRE_Solver solver, hypre_double outer_wt );
HYPRE_Int HYPRE_ParCSRHybridSetOuterWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double outer_wt );
HYPRE_Int HYPRE_ParCSRHybridSetPCGMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int pcg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetPCGMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int pcg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetPCGMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int pcg_max_its );
HYPRE_Int HYPRE_ParCSRHybridSetPMaxElmts_flt  ( HYPRE_Solver solver, HYPRE_Int p_max );
HYPRE_Int HYPRE_ParCSRHybridSetPMaxElmts_dbl  ( HYPRE_Solver solver, HYPRE_Int p_max );
HYPRE_Int HYPRE_ParCSRHybridSetPMaxElmts_long_dbl  ( HYPRE_Solver solver, HYPRE_Int p_max );
HYPRE_Int HYPRE_ParCSRHybridSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRHybridSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRHybridSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRHybridSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRHybridSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRHybridSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxOrder_flt  ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxOrder_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxOrder_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_order );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxType_flt  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxType_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int relax_type );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWeight_flt  ( HYPRE_Solver solver, hypre_float *relax_weight );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWeight_dbl  ( HYPRE_Solver solver, hypre_double *relax_weight );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWeight_long_dbl  ( HYPRE_Solver solver, hypre_long_double *relax_weight );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWt_flt  ( HYPRE_Solver solver, hypre_float relax_wt );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWt_dbl  ( HYPRE_Solver solver, hypre_double relax_wt );
HYPRE_Int HYPRE_ParCSRHybridSetRelaxWt_long_dbl  ( HYPRE_Solver solver, hypre_long_double relax_wt );
HYPRE_Int HYPRE_ParCSRHybridSetRelChange_flt  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRHybridSetRelChange_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRHybridSetRelChange_long_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRHybridSetSeqThreshold_flt  ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetSeqThreshold_dbl  ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetSeqThreshold_long_dbl  ( HYPRE_Solver solver, HYPRE_Int seq_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetSetupType_flt  ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_ParCSRHybridSetSetupType_dbl  ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_ParCSRHybridSetSetupType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int setup_type );
HYPRE_Int HYPRE_ParCSRHybridSetSolverType_flt  ( HYPRE_Solver solver, HYPRE_Int solver_type );
HYPRE_Int HYPRE_ParCSRHybridSetSolverType_dbl  ( HYPRE_Solver solver, HYPRE_Int solver_type );
HYPRE_Int HYPRE_ParCSRHybridSetSolverType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int solver_type );
HYPRE_Int HYPRE_ParCSRHybridSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRHybridSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRHybridSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRHybridSetStrongThreshold_flt  ( HYPRE_Solver solver, hypre_float strong_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetStrongThreshold_dbl  ( HYPRE_Solver solver, hypre_double strong_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetStrongThreshold_long_dbl  ( HYPRE_Solver solver, hypre_long_double strong_threshold );
HYPRE_Int HYPRE_ParCSRHybridSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRHybridSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRHybridSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRHybridSetTruncFactor_flt  ( HYPRE_Solver solver, hypre_float trunc_factor );
HYPRE_Int HYPRE_ParCSRHybridSetTruncFactor_dbl  ( HYPRE_Solver solver, hypre_double trunc_factor );
HYPRE_Int HYPRE_ParCSRHybridSetTruncFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double trunc_factor );
HYPRE_Int HYPRE_ParCSRHybridSetTwoNorm_flt  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRHybridSetTwoNorm_dbl  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRHybridSetTwoNorm_long_dbl  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRHybridSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRHybridSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
void aux_indexFromMask_flt  ( HYPRE_Int n, HYPRE_Int *mask, HYPRE_Int *index );
void aux_indexFromMask_dbl  ( HYPRE_Int n, HYPRE_Int *mask, HYPRE_Int *index );
void aux_indexFromMask_long_dbl  ( HYPRE_Int n, HYPRE_Int *mask, HYPRE_Int *index );
HYPRE_Int aux_maskCount_flt  ( HYPRE_Int n, HYPRE_Int *mask );
HYPRE_Int aux_maskCount_dbl  ( HYPRE_Int n, HYPRE_Int *mask );
HYPRE_Int aux_maskCount_long_dbl  ( HYPRE_Int n, HYPRE_Int *mask );
HYPRE_Int HYPRE_ParCSRMultiVectorPrint_flt  ( void *x_, const char *fileName );
HYPRE_Int HYPRE_ParCSRMultiVectorPrint_dbl  ( void *x_, const char *fileName );
HYPRE_Int HYPRE_ParCSRMultiVectorPrint_long_dbl  ( void *x_, const char *fileName );
void *HYPRE_ParCSRMultiVectorRead_flt  ( MPI_Comm comm, void *ii_, const char *fileName );
void *HYPRE_ParCSRMultiVectorRead_dbl  ( MPI_Comm comm, void *ii_, const char *fileName );
void *HYPRE_ParCSRMultiVectorRead_long_dbl  ( MPI_Comm comm, void *ii_, const char *fileName );
HYPRE_Int HYPRE_ParCSRSetupInterpreter_flt  ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRSetupInterpreter_dbl  ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRSetupInterpreter_long_dbl  ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRSetupMatvec_flt  ( HYPRE_MatvecFunctions *mv );
HYPRE_Int HYPRE_ParCSRSetupMatvec_dbl  ( HYPRE_MatvecFunctions *mv );
HYPRE_Int HYPRE_ParCSRSetupMatvec_long_dbl  ( HYPRE_MatvecFunctions *mv );
HYPRE_Int hypre_ParPrintVector_flt  ( void *v, const char *file );
HYPRE_Int hypre_ParPrintVector_dbl  ( void *v, const char *file );
HYPRE_Int hypre_ParPrintVector_long_dbl  ( void *v, const char *file );
void *hypre_ParReadVector_flt  ( MPI_Comm comm, const char *file );
void *hypre_ParReadVector_dbl  ( MPI_Comm comm, const char *file );
void *hypre_ParReadVector_long_dbl  ( MPI_Comm comm, const char *file );
HYPRE_Int hypre_ParSetRandomValues_flt  ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_ParSetRandomValues_dbl  ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_ParSetRandomValues_long_dbl  ( void *v, HYPRE_Int seed );
HYPRE_Int hypre_ParVectorSize_flt  ( void *x );
HYPRE_Int hypre_ParVectorSize_dbl  ( void *x );
HYPRE_Int hypre_ParVectorSize_long_dbl  ( void *x );
HYPRE_Int HYPRE_TempParCSRSetupInterpreter_flt  ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_TempParCSRSetupInterpreter_dbl  ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_TempParCSRSetupInterpreter_long_dbl  ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_ParCSRLGMRESCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRLGMRESCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRLGMRESCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRLGMRESDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRLGMRESDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRLGMRESDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRLGMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRLGMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRLGMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRLGMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRLGMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRLGMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRLGMRESGetResidual_flt  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRLGMRESGetResidual_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRLGMRESGetResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRLGMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetAugDim_flt  ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetAugDim_dbl  ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetAugDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_ParCSRLGMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRLGMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRLGMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRLGMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                         HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRLGMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_ParCSRLGMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRLGMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRLGMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                    HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsBuildIJMatrix_flt  ( HYPRE_Solver solver, HYPRE_IJMatrix *pij_A );
HYPRE_Int HYPRE_ParaSailsBuildIJMatrix_dbl  ( HYPRE_Solver solver, HYPRE_IJMatrix *pij_A );
HYPRE_Int HYPRE_ParaSailsBuildIJMatrix_long_dbl  ( HYPRE_Solver solver, HYPRE_IJMatrix *pij_A );
HYPRE_Int HYPRE_ParaSailsCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParaSailsCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParaSailsCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParaSailsDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParaSailsDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParaSailsDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParaSailsGetFilter_flt  ( HYPRE_Solver solver, hypre_float *filter );
HYPRE_Int HYPRE_ParaSailsGetFilter_dbl  ( HYPRE_Solver solver, hypre_double *filter );
HYPRE_Int HYPRE_ParaSailsGetFilter_long_dbl  ( HYPRE_Solver solver, hypre_long_double *filter );
HYPRE_Int HYPRE_ParaSailsGetLoadbal_flt  ( HYPRE_Solver solver, hypre_float *loadbal );
HYPRE_Int HYPRE_ParaSailsGetLoadbal_dbl  ( HYPRE_Solver solver, hypre_double *loadbal );
HYPRE_Int HYPRE_ParaSailsGetLoadbal_long_dbl  ( HYPRE_Solver solver, hypre_long_double *loadbal );
HYPRE_Int HYPRE_ParaSailsGetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_ParaSailsGetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_ParaSailsGetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *logging );
HYPRE_Int HYPRE_ParaSailsGetNlevels_flt  ( HYPRE_Solver solver, HYPRE_Int *nlevels );
HYPRE_Int HYPRE_ParaSailsGetNlevels_dbl  ( HYPRE_Solver solver, HYPRE_Int *nlevels );
HYPRE_Int HYPRE_ParaSailsGetNlevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *nlevels );
HYPRE_Int HYPRE_ParaSailsGetReuse_flt  ( HYPRE_Solver solver, HYPRE_Int *reuse );
HYPRE_Int HYPRE_ParaSailsGetReuse_dbl  ( HYPRE_Solver solver, HYPRE_Int *reuse );
HYPRE_Int HYPRE_ParaSailsGetReuse_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *reuse );
HYPRE_Int HYPRE_ParaSailsGetSym_flt  ( HYPRE_Solver solver, HYPRE_Int *sym );
HYPRE_Int HYPRE_ParaSailsGetSym_dbl  ( HYPRE_Solver solver, HYPRE_Int *sym );
HYPRE_Int HYPRE_ParaSailsGetSym_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *sym );
HYPRE_Int HYPRE_ParaSailsGetThresh_flt  ( HYPRE_Solver solver, hypre_float *thresh );
HYPRE_Int HYPRE_ParaSailsGetThresh_dbl  ( HYPRE_Solver solver, hypre_double *thresh );
HYPRE_Int HYPRE_ParaSailsGetThresh_long_dbl  ( HYPRE_Solver solver, hypre_long_double *thresh );
HYPRE_Int HYPRE_ParaSailsSetFilter_flt  ( HYPRE_Solver solver, hypre_float filter );
HYPRE_Int HYPRE_ParaSailsSetFilter_dbl  ( HYPRE_Solver solver, hypre_double filter );
HYPRE_Int HYPRE_ParaSailsSetFilter_long_dbl  ( HYPRE_Solver solver, hypre_long_double filter );
HYPRE_Int HYPRE_ParaSailsSetLoadbal_flt  ( HYPRE_Solver solver, hypre_float loadbal );
HYPRE_Int HYPRE_ParaSailsSetLoadbal_dbl  ( HYPRE_Solver solver, hypre_double loadbal );
HYPRE_Int HYPRE_ParaSailsSetLoadbal_long_dbl  ( HYPRE_Solver solver, hypre_long_double loadbal );
HYPRE_Int HYPRE_ParaSailsSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParaSailsSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParaSailsSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParaSailsSetNlevels_flt  ( HYPRE_Solver solver, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetNlevels_dbl  ( HYPRE_Solver solver, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetNlevels_long_dbl  ( HYPRE_Solver solver, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetParams_flt  ( HYPRE_Solver solver, hypre_float thresh, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetParams_dbl  ( HYPRE_Solver solver, hypre_double thresh, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetParams_long_dbl  ( HYPRE_Solver solver, hypre_long_double thresh, HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParaSailsSetReuse_flt  ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParaSailsSetReuse_dbl  ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParaSailsSetReuse_long_dbl  ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParaSailsSetSym_flt  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParaSailsSetSym_dbl  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParaSailsSetSym_long_dbl  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParaSailsSetThresh_flt  ( HYPRE_Solver solver, hypre_float thresh );
HYPRE_Int HYPRE_ParaSailsSetThresh_dbl  ( HYPRE_Solver solver, hypre_double thresh );
HYPRE_Int HYPRE_ParaSailsSetThresh_long_dbl  ( HYPRE_Solver solver, hypre_long_double thresh );
HYPRE_Int HYPRE_ParaSailsSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParaSailsSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRParaSailsCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRParaSailsCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRParaSailsDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRParaSailsDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRParaSailsDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRParaSailsGetFilter_flt  ( HYPRE_Solver solver, hypre_float *filter );
HYPRE_Int HYPRE_ParCSRParaSailsGetFilter_dbl  ( HYPRE_Solver solver, hypre_double *filter );
HYPRE_Int HYPRE_ParCSRParaSailsGetFilter_long_dbl  ( HYPRE_Solver solver, hypre_long_double *filter );
HYPRE_Int HYPRE_ParCSRParaSailsGetLoadbal_flt  ( HYPRE_Solver solver, hypre_float *loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsGetLoadbal_dbl  ( HYPRE_Solver solver, hypre_double *loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsGetLoadbal_long_dbl  ( HYPRE_Solver solver, hypre_long_double *loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsSetFilter_flt  ( HYPRE_Solver solver, hypre_float filter );
HYPRE_Int HYPRE_ParCSRParaSailsSetFilter_dbl  ( HYPRE_Solver solver, hypre_double filter );
HYPRE_Int HYPRE_ParCSRParaSailsSetFilter_long_dbl  ( HYPRE_Solver solver, hypre_long_double filter );
HYPRE_Int HYPRE_ParCSRParaSailsSetLoadbal_flt  ( HYPRE_Solver solver, hypre_float loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsSetLoadbal_dbl  ( HYPRE_Solver solver, hypre_double loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsSetLoadbal_long_dbl  ( HYPRE_Solver solver, hypre_long_double loadbal );
HYPRE_Int HYPRE_ParCSRParaSailsSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRParaSailsSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRParaSailsSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_ParCSRParaSailsSetParams_flt  ( HYPRE_Solver solver, hypre_float thresh,
                                           HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParCSRParaSailsSetParams_dbl  ( HYPRE_Solver solver, hypre_double thresh,
                                           HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParCSRParaSailsSetParams_long_dbl  ( HYPRE_Solver solver, hypre_long_double thresh,
                                           HYPRE_Int nlevels );
HYPRE_Int HYPRE_ParCSRParaSailsSetReuse_flt  ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParCSRParaSailsSetReuse_dbl  ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParCSRParaSailsSetReuse_long_dbl  ( HYPRE_Solver solver, HYPRE_Int reuse );
HYPRE_Int HYPRE_ParCSRParaSailsSetSym_flt  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParCSRParaSailsSetSym_dbl  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParCSRParaSailsSetSym_long_dbl  ( HYPRE_Solver solver, HYPRE_Int sym );
HYPRE_Int HYPRE_ParCSRParaSailsSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRParaSailsSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRDiagScale_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA, HYPRE_ParVector Hy,
                                  HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSRDiagScale_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA, HYPRE_ParVector Hy,
                                  HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSRDiagScale_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix HA, HYPRE_ParVector Hy,
                                  HYPRE_ParVector Hx );
HYPRE_Int HYPRE_ParCSRDiagScaleSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector y,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRDiagScaleSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector y,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRDiagScaleSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector y,
                                       HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPCGCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPCGCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPCGDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPCGDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPCGDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_ParCSRPCGGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRPCGGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRPCGGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_ParCSRPCGGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRPCGGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRPCGGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_ParCSRPCGGetResidual_flt  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRPCGGetResidual_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRPCGGetResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_ParVector *residual );
HYPRE_Int HYPRE_ParCSRPCGSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_ParCSRPCGSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_ParCSRPCGSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_ParCSRPCGSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPCGSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPCGSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPCGSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                      HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRPCGSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                      HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRPCGSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToParSolverFcn precond,
                                      HYPRE_PtrToParSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_ParCSRPCGSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_ParCSRPCGSetRelChange_flt  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRPCGSetRelChange_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRPCGSetRelChange_long_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_ParCSRPCGSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRPCGSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRPCGSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_ParCSRPCGSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRPCGSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRPCGSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRPCGSetTwoNorm_flt  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRPCGSetTwoNorm_dbl  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRPCGSetTwoNorm_long_dbl  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_ParCSRPCGSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPCGSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                 HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutCreate_flt  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPilutCreate_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPilutCreate_long_dbl  ( MPI_Comm comm, HYPRE_Solver *solver );
HYPRE_Int HYPRE_ParCSRPilutDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPilutDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPilutDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_ParCSRPilutSetDropTolerance_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_ParCSRPilutSetDropTolerance_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_ParCSRPilutSetDropTolerance_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_ParCSRPilutSetFactorRowSize_flt  ( HYPRE_Solver solver, HYPRE_Int size );
HYPRE_Int HYPRE_ParCSRPilutSetFactorRowSize_dbl  ( HYPRE_Solver solver, HYPRE_Int size );
HYPRE_Int HYPRE_ParCSRPilutSetFactorRowSize_long_dbl  ( HYPRE_Solver solver, HYPRE_Int size );
HYPRE_Int HYPRE_ParCSRPilutSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPilutSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPilutSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_ParCSRPilutSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_ParCSRPilutSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                                   HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzCreate_flt  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_SchwarzCreate_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_SchwarzCreate_long_dbl  ( HYPRE_Solver *solver );
HYPRE_Int HYPRE_SchwarzDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_SchwarzDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_SchwarzDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_SchwarzSetDofFunc_flt  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_SchwarzSetDofFunc_dbl  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_SchwarzSetDofFunc_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *dof_func );
HYPRE_Int HYPRE_SchwarzSetDomainStructure_flt  ( HYPRE_Solver solver, HYPRE_CSRMatrix domain_structure );
HYPRE_Int HYPRE_SchwarzSetDomainStructure_dbl  ( HYPRE_Solver solver, HYPRE_CSRMatrix domain_structure );
HYPRE_Int HYPRE_SchwarzSetDomainStructure_long_dbl  ( HYPRE_Solver solver, HYPRE_CSRMatrix domain_structure );
HYPRE_Int HYPRE_SchwarzSetDomainType_flt  ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_SchwarzSetDomainType_dbl  ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_SchwarzSetDomainType_long_dbl  ( HYPRE_Solver solver, HYPRE_Int domain_type );
HYPRE_Int HYPRE_SchwarzSetNonSymm_flt  ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_SchwarzSetNonSymm_dbl  ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_SchwarzSetNonSymm_long_dbl  ( HYPRE_Solver solver, HYPRE_Int use_nonsymm );
HYPRE_Int HYPRE_SchwarzSetNumFunctions_flt  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_SchwarzSetNumFunctions_dbl  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_SchwarzSetNumFunctions_long_dbl  ( HYPRE_Solver solver, HYPRE_Int num_functions );
HYPRE_Int HYPRE_SchwarzSetOverlap_flt  ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_SchwarzSetOverlap_dbl  ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_SchwarzSetOverlap_long_dbl  ( HYPRE_Solver solver, HYPRE_Int overlap );
HYPRE_Int HYPRE_SchwarzSetRelaxWeight_flt  ( HYPRE_Solver solver, hypre_float relax_weight );
HYPRE_Int HYPRE_SchwarzSetRelaxWeight_dbl  ( HYPRE_Solver solver, hypre_double relax_weight );
HYPRE_Int HYPRE_SchwarzSetRelaxWeight_long_dbl  ( HYPRE_Solver solver, hypre_long_double relax_weight );
HYPRE_Int HYPRE_SchwarzSetup_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSetup_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSetVariant_flt  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_SchwarzSetVariant_dbl  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_SchwarzSetVariant_long_dbl  ( HYPRE_Solver solver, HYPRE_Int variant );
HYPRE_Int HYPRE_SchwarzSolve_flt  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSolve_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int HYPRE_SchwarzSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                               HYPRE_ParVector x );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                    hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                    HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_float trunc_factor,
                                                    HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                    hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                    HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_double trunc_factor,
                                                    HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                    hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                    HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                                    HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterpHost_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                        HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_float trunc_factor,
                                                        HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterpHost_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                        HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_double trunc_factor,
                                                        HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtInterpHost_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                        HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                                        HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                      hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                      HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_float trunc_factor,
                                                      HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                      hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                      HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_double trunc_factor,
                                                      HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                      hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                      HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                                      HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterpHost_flt  ( hypre_ParCSRMatrix *A,
                                                          HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                          HYPRE_BigInt *num_old_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                          HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterpHost_dbl  ( hypre_ParCSRMatrix *A,
                                                          HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                          HYPRE_BigInt *num_old_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                          HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModPartialExtPEInterpHost_long_dbl  ( hypre_ParCSRMatrix *A,
                                                          HYPRE_Int *CF_marker, hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                          HYPRE_BigInt *num_old_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                          HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGAdditiveCycle_flt  ( void *amg_vdata );
HYPRE_Int hypre_BoomerAMGAdditiveCycle_dbl  ( void *amg_vdata );
HYPRE_Int hypre_BoomerAMGAdditiveCycle_long_dbl  ( void *amg_vdata );
HYPRE_Int hypre_CreateDinv_flt  ( void *amg_vdata );
HYPRE_Int hypre_CreateDinv_dbl  ( void *amg_vdata );
HYPRE_Int hypre_CreateDinv_long_dbl  ( void *amg_vdata );
HYPRE_Int hypre_CreateLambda_flt  ( void *amg_vdata );
HYPRE_Int hypre_CreateLambda_dbl  ( void *amg_vdata );
HYPRE_Int hypre_CreateLambda_long_dbl  ( void *amg_vdata );
hypre_AMGDDCommPkg *hypre_AMGDDCommPkgCreate_flt  ( HYPRE_Int num_levels );
hypre_AMGDDCommPkg *hypre_AMGDDCommPkgCreate_dbl  ( HYPRE_Int num_levels );
hypre_AMGDDCommPkg *hypre_AMGDDCommPkgCreate_long_dbl  ( HYPRE_Int num_levels );
HYPRE_Int hypre_AMGDDCommPkgDestroy_flt  ( hypre_AMGDDCommPkg *compGridCommPkg );
HYPRE_Int hypre_AMGDDCommPkgDestroy_dbl  ( hypre_AMGDDCommPkg *compGridCommPkg );
HYPRE_Int hypre_AMGDDCommPkgDestroy_long_dbl  ( hypre_AMGDDCommPkg *compGridCommPkg );
HYPRE_Int hypre_AMGDDCommPkgRecvLevelDestroy_flt  ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
HYPRE_Int hypre_AMGDDCommPkgRecvLevelDestroy_dbl  ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
HYPRE_Int hypre_AMGDDCommPkgRecvLevelDestroy_long_dbl  ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
HYPRE_Int hypre_AMGDDCommPkgSendLevelDestroy_flt  ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
HYPRE_Int hypre_AMGDDCommPkgSendLevelDestroy_dbl  ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
HYPRE_Int hypre_AMGDDCommPkgSendLevelDestroy_long_dbl  ( hypre_AMGDDCommPkg *amgddCommPkg, HYPRE_Int level,
                                               HYPRE_Int proc );
hypre_AMGDDCompGrid *hypre_AMGDDCompGridCreate_flt ( void );
hypre_AMGDDCompGrid *hypre_AMGDDCompGridCreate_dbl ( void );
hypre_AMGDDCompGrid *hypre_AMGDDCompGridCreate_long_dbl ( void );
HYPRE_Int hypre_AMGDDCompGridDestroy_flt  ( hypre_AMGDDCompGrid *compGrid );
HYPRE_Int hypre_AMGDDCompGridDestroy_dbl  ( hypre_AMGDDCompGrid *compGrid );
HYPRE_Int hypre_AMGDDCompGridDestroy_long_dbl  ( hypre_AMGDDCompGrid *compGrid );
HYPRE_Int hypre_AMGDDCompGridFinalize_flt  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridFinalize_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridFinalize_long_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridInitialize_flt  ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int padding,
                                          HYPRE_Int level );
HYPRE_Int hypre_AMGDDCompGridInitialize_dbl  ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int padding,
                                          HYPRE_Int level );
HYPRE_Int hypre_AMGDDCompGridInitialize_long_dbl  ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int padding,
                                          HYPRE_Int level );
hypre_AMGDDCompGridMatrix* hypre_AMGDDCompGridMatrixCreate_flt ( void );
hypre_AMGDDCompGridMatrix* hypre_AMGDDCompGridMatrixCreate_dbl ( void );
hypre_AMGDDCompGridMatrix* hypre_AMGDDCompGridMatrixCreate_long_dbl ( void );
HYPRE_Int hypre_AMGDDCompGridMatrixDestroy_flt  ( hypre_AMGDDCompGridMatrix *matrix );
HYPRE_Int hypre_AMGDDCompGridMatrixDestroy_dbl  ( hypre_AMGDDCompGridMatrix *matrix );
HYPRE_Int hypre_AMGDDCompGridMatrixDestroy_long_dbl  ( hypre_AMGDDCompGridMatrix *matrix );
HYPRE_Int hypre_AMGDDCompGridMatvec_flt  ( hypre_float alpha, hypre_AMGDDCompGridMatrix *A,
                                      hypre_AMGDDCompGridVector *x, hypre_float beta, hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridMatvec_dbl  ( hypre_double alpha, hypre_AMGDDCompGridMatrix *A,
                                      hypre_AMGDDCompGridVector *x, hypre_double beta, hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridMatvec_long_dbl  ( hypre_long_double alpha, hypre_AMGDDCompGridMatrix *A,
                                      hypre_AMGDDCompGridVector *x, hypre_long_double beta, hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridRealMatvec_flt  ( hypre_float alpha, hypre_AMGDDCompGridMatrix *A,
                                          hypre_AMGDDCompGridVector *x, hypre_float beta, hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridRealMatvec_dbl  ( hypre_double alpha, hypre_AMGDDCompGridMatrix *A,
                                          hypre_AMGDDCompGridVector *x, hypre_double beta, hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridRealMatvec_long_dbl  ( hypre_long_double alpha, hypre_AMGDDCompGridMatrix *A,
                                          hypre_AMGDDCompGridVector *x, hypre_long_double beta, hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridResize_flt  ( hypre_AMGDDCompGrid *compGrid, HYPRE_Int new_size,
                                      HYPRE_Int need_coarse_info );
HYPRE_Int hypre_AMGDDCompGridResize_dbl  ( hypre_AMGDDCompGrid *compGrid, HYPRE_Int new_size,
                                      HYPRE_Int need_coarse_info );
HYPRE_Int hypre_AMGDDCompGridResize_long_dbl  ( hypre_AMGDDCompGrid *compGrid, HYPRE_Int new_size,
                                      HYPRE_Int need_coarse_info );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndices_flt  ( hypre_AMGDDCompGrid **compGrid,
                                                 HYPRE_Int *num_added_nodes, HYPRE_Int ****recv_map, HYPRE_Int num_recv_procs,
                                                 HYPRE_Int **A_tmp_info, HYPRE_Int start_level, HYPRE_Int num_levels );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndices_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                                 HYPRE_Int *num_added_nodes, HYPRE_Int ****recv_map, HYPRE_Int num_recv_procs,
                                                 HYPRE_Int **A_tmp_info, HYPRE_Int start_level, HYPRE_Int num_levels );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndices_long_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                                 HYPRE_Int *num_added_nodes, HYPRE_Int ****recv_map, HYPRE_Int num_recv_procs,
                                                 HYPRE_Int **A_tmp_info, HYPRE_Int start_level, HYPRE_Int num_levels );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndicesP_flt  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndicesP_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridSetupLocalIndicesP_long_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridSetupRelax_flt  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridSetupRelax_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridSetupRelax_long_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_AMGDDCompGridVectorAxpy_flt  ( hypre_float alpha, hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorAxpy_dbl  ( hypre_double alpha, hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorAxpy_long_dbl  ( hypre_long_double alpha, hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorCopy_flt  ( hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorCopy_dbl  ( hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorCopy_long_dbl  ( hypre_AMGDDCompGridVector *x,
                                          hypre_AMGDDCompGridVector *y );
hypre_AMGDDCompGridVector* hypre_AMGDDCompGridVectorCreate_flt ( void );
hypre_AMGDDCompGridVector* hypre_AMGDDCompGridVectorCreate_dbl ( void );
hypre_AMGDDCompGridVector* hypre_AMGDDCompGridVectorCreate_long_dbl ( void );
HYPRE_Int hypre_AMGDDCompGridVectorDestroy_flt  ( hypre_AMGDDCompGridVector *vector );
HYPRE_Int hypre_AMGDDCompGridVectorDestroy_dbl  ( hypre_AMGDDCompGridVector *vector );
HYPRE_Int hypre_AMGDDCompGridVectorDestroy_long_dbl  ( hypre_AMGDDCompGridVector *vector );
HYPRE_Int hypre_AMGDDCompGridVectorInitialize_flt  ( hypre_AMGDDCompGridVector *vector,
                                                HYPRE_Int num_owned, HYPRE_Int num_nonowned, HYPRE_Int num_real );
HYPRE_Int hypre_AMGDDCompGridVectorInitialize_dbl  ( hypre_AMGDDCompGridVector *vector,
                                                HYPRE_Int num_owned, HYPRE_Int num_nonowned, HYPRE_Int num_real );
HYPRE_Int hypre_AMGDDCompGridVectorInitialize_long_dbl  ( hypre_AMGDDCompGridVector *vector,
                                                HYPRE_Int num_owned, HYPRE_Int num_nonowned, HYPRE_Int num_real );
hypre_float hypre_AMGDDCompGridVectorInnerProd_flt  ( hypre_AMGDDCompGridVector *x,
                                                hypre_AMGDDCompGridVector *y );
hypre_double hypre_AMGDDCompGridVectorInnerProd_dbl  ( hypre_AMGDDCompGridVector *x,
                                                hypre_AMGDDCompGridVector *y );
hypre_long_double hypre_AMGDDCompGridVectorInnerProd_long_dbl  ( hypre_AMGDDCompGridVector *x,
                                                hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealAxpy_flt  ( hypre_float alpha, hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealAxpy_dbl  ( hypre_double alpha, hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealAxpy_long_dbl  ( hypre_long_double alpha, hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealCopy_flt  ( hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealCopy_dbl  ( hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealCopy_long_dbl  ( hypre_AMGDDCompGridVector *x,
                                              hypre_AMGDDCompGridVector *y );
hypre_float hypre_AMGDDCompGridVectorRealInnerProd_flt  ( hypre_AMGDDCompGridVector *x,
                                                    hypre_AMGDDCompGridVector *y );
hypre_double hypre_AMGDDCompGridVectorRealInnerProd_dbl  ( hypre_AMGDDCompGridVector *x,
                                                    hypre_AMGDDCompGridVector *y );
hypre_long_double hypre_AMGDDCompGridVectorRealInnerProd_long_dbl  ( hypre_AMGDDCompGridVector *x,
                                                    hypre_AMGDDCompGridVector *y );
HYPRE_Int hypre_AMGDDCompGridVectorRealScale_flt  ( hypre_float alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorRealScale_dbl  ( hypre_double alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorRealScale_long_dbl  ( hypre_long_double alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorRealSetConstantValues_flt  ( hypre_AMGDDCompGridVector *vector,
                                                           hypre_float value );
HYPRE_Int hypre_AMGDDCompGridVectorRealSetConstantValues_dbl  ( hypre_AMGDDCompGridVector *vector,
                                                           hypre_double value );
HYPRE_Int hypre_AMGDDCompGridVectorRealSetConstantValues_long_dbl  ( hypre_AMGDDCompGridVector *vector,
                                                           hypre_long_double value );
HYPRE_Int hypre_AMGDDCompGridVectorScale_flt  ( hypre_float alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorScale_dbl  ( hypre_double alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorScale_long_dbl  ( hypre_long_double alpha, hypre_AMGDDCompGridVector *x );
HYPRE_Int hypre_AMGDDCompGridVectorSetConstantValues_flt  ( hypre_AMGDDCompGridVector *vector,
                                                       hypre_float value );
HYPRE_Int hypre_AMGDDCompGridVectorSetConstantValues_dbl  ( hypre_AMGDDCompGridVector *vector,
                                                       hypre_double value );
HYPRE_Int hypre_AMGDDCompGridVectorSetConstantValues_long_dbl  ( hypre_AMGDDCompGridVector *vector,
                                                       hypre_long_double value );
HYPRE_Int hypre_BoomerAMGDD_FAC_flt  ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_dbl  ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_long_dbl  ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1Jacobi_flt  ( void *amgdd_vdata, HYPRE_Int level,
                                             HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1Jacobi_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                             HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1Jacobi_long_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                             HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1JacobiHost_flt  ( void *amgdd_vdata, HYPRE_Int level,
                                                 HYPRE_Int relax_set );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1JacobiHost_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                                 HYPRE_Int relax_set );
HYPRE_Int hypre_BoomerAMGDD_FAC_CFL1JacobiHost_long_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                                 HYPRE_Int relax_set );
HYPRE_Int hypre_BoomerAMGDD_FAC_Cycle_flt  ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_type,
                                        HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_Cycle_dbl  ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_type,
                                        HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_Cycle_long_dbl  ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_type,
                                        HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_FCycle_flt  ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_FCycle_dbl  ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_FCycle_long_dbl  ( void *amgdd_vdata, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_GaussSeidel_flt  ( void *amgdd_vdata, HYPRE_Int level,
                                              HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_GaussSeidel_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                              HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_GaussSeidel_long_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                              HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_Interpolate_flt  ( hypre_AMGDDCompGrid *compGrid_f,
                                              hypre_AMGDDCompGrid *compGrid_c );
HYPRE_Int hypre_BoomerAMGDD_FAC_Interpolate_dbl  ( hypre_AMGDDCompGrid *compGrid_f,
                                              hypre_AMGDDCompGrid *compGrid_c );
HYPRE_Int hypre_BoomerAMGDD_FAC_Interpolate_long_dbl  ( hypre_AMGDDCompGrid *compGrid_f,
                                              hypre_AMGDDCompGrid *compGrid_c );
HYPRE_Int hypre_BoomerAMGDD_FAC_Jacobi_flt  ( void *amgdd_vdata, HYPRE_Int level,
                                         HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_Jacobi_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                         HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_Jacobi_long_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                         HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_JacobiHost_flt  ( void *amgdd_vdata, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGDD_FAC_JacobiHost_dbl  ( void *amgdd_vdata, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGDD_FAC_JacobiHost_long_dbl  ( void *amgdd_vdata, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGDD_FAC_OrderedGaussSeidel_flt  ( void *amgdd_vdata, HYPRE_Int level,
                                                     HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_OrderedGaussSeidel_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                                     HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_OrderedGaussSeidel_long_dbl  ( void *amgdd_vdata, HYPRE_Int level,
                                                     HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_Relax_flt  ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_Relax_dbl  ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_Relax_long_dbl  ( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param );
HYPRE_Int hypre_BoomerAMGDD_FAC_Restrict_flt  ( hypre_AMGDDCompGrid *compGrid_f,
                                           hypre_AMGDDCompGrid *compGrid_c, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_Restrict_dbl  ( hypre_AMGDDCompGrid *compGrid_f,
                                           hypre_AMGDDCompGrid *compGrid_c, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_FAC_Restrict_long_dbl  ( hypre_AMGDDCompGrid *compGrid_f,
                                           hypre_AMGDDCompGrid *compGrid_c, HYPRE_Int first_iteration );
HYPRE_Int hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo_flt  ( hypre_ParAMGDDData* amgdd_data );
HYPRE_Int hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo_dbl  ( hypre_ParAMGDDData* amgdd_data );
HYPRE_Int hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo_long_dbl  ( hypre_ParAMGDDData* amgdd_data );
HYPRE_Int hypre_BoomerAMGDD_FixUpRecvMaps_flt  ( hypre_AMGDDCompGrid **compGrid,
                                            hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int start_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_FixUpRecvMaps_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                            hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int start_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_FixUpRecvMaps_long_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                            hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int start_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_MarkCoarse_flt  ( HYPRE_Int *list, HYPRE_Int *marker,
                                         HYPRE_Int *owned_coarse_indices, HYPRE_Int *nonowned_coarse_indices, HYPRE_Int *sort_map,
                                         HYPRE_Int num_owned, HYPRE_Int total_num_nodes, HYPRE_Int num_owned_coarse, HYPRE_Int list_size,
                                         HYPRE_Int dist, HYPRE_Int use_sort, HYPRE_Int *nodes_to_add );
HYPRE_Int hypre_BoomerAMGDD_MarkCoarse_dbl  ( HYPRE_Int *list, HYPRE_Int *marker,
                                         HYPRE_Int *owned_coarse_indices, HYPRE_Int *nonowned_coarse_indices, HYPRE_Int *sort_map,
                                         HYPRE_Int num_owned, HYPRE_Int total_num_nodes, HYPRE_Int num_owned_coarse, HYPRE_Int list_size,
                                         HYPRE_Int dist, HYPRE_Int use_sort, HYPRE_Int *nodes_to_add );
HYPRE_Int hypre_BoomerAMGDD_MarkCoarse_long_dbl  ( HYPRE_Int *list, HYPRE_Int *marker,
                                         HYPRE_Int *owned_coarse_indices, HYPRE_Int *nonowned_coarse_indices, HYPRE_Int *sort_map,
                                         HYPRE_Int num_owned, HYPRE_Int total_num_nodes, HYPRE_Int num_owned_coarse, HYPRE_Int list_size,
                                         HYPRE_Int dist, HYPRE_Int use_sort, HYPRE_Int *nodes_to_add );
HYPRE_Int hypre_BoomerAMGDD_PackRecvMapSendBuffer_flt  ( HYPRE_Int *recv_map_send_buffer,
                                                    HYPRE_Int **recv_red_marker, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size,
                                                    HYPRE_Int current_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_PackRecvMapSendBuffer_dbl  ( HYPRE_Int *recv_map_send_buffer,
                                                    HYPRE_Int **recv_red_marker, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size,
                                                    HYPRE_Int current_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_PackRecvMapSendBuffer_long_dbl  ( HYPRE_Int *recv_map_send_buffer,
                                                    HYPRE_Int **recv_red_marker, HYPRE_Int *num_recv_nodes, HYPRE_Int *recv_buffer_size,
                                                    HYPRE_Int current_level, HYPRE_Int num_levels );
HYPRE_Int* hypre_BoomerAMGDD_PackSendBuffer_flt  ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int proc,
                                              HYPRE_Int current_level, HYPRE_Int *padding, HYPRE_Int *send_flag_buffer_size );
HYPRE_Int* hypre_BoomerAMGDD_PackSendBuffer_dbl  ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int proc,
                                              HYPRE_Int current_level, HYPRE_Int *padding, HYPRE_Int *send_flag_buffer_size );
HYPRE_Int* hypre_BoomerAMGDD_PackSendBuffer_long_dbl  ( hypre_ParAMGDDData *amgdd_data, HYPRE_Int proc,
                                              HYPRE_Int current_level, HYPRE_Int *padding, HYPRE_Int *send_flag_buffer_size );
HYPRE_Int hypre_BoomerAMGDD_RecursivelyBuildPsiComposite_flt  ( HYPRE_Int node, HYPRE_Int m,
                                                           hypre_AMGDDCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int use_sort );
HYPRE_Int hypre_BoomerAMGDD_RecursivelyBuildPsiComposite_dbl  ( HYPRE_Int node, HYPRE_Int m,
                                                           hypre_AMGDDCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int use_sort );
HYPRE_Int hypre_BoomerAMGDD_RecursivelyBuildPsiComposite_long_dbl  ( HYPRE_Int node, HYPRE_Int m,
                                                           hypre_AMGDDCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int use_sort );
HYPRE_Int hypre_BoomerAMGDD_SetupNearestProcessorNeighbors_flt  ( hypre_ParCSRMatrix *A,
                                                             hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding,
                                                             HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDD_SetupNearestProcessorNeighbors_dbl  ( hypre_ParCSRMatrix *A,
                                                             hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding,
                                                             HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDD_SetupNearestProcessorNeighbors_long_dbl  ( hypre_ParCSRMatrix *A,
                                                             hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding,
                                                             HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDD_UnpackRecvBuffer_flt  ( hypre_ParAMGDDData *amgdd_data,
                                               HYPRE_Int *recv_buffer, HYPRE_Int **A_tmp_info, HYPRE_Int *recv_map_send_buffer_size,
                                               HYPRE_Int *nodes_added_on_level, HYPRE_Int current_level, HYPRE_Int buffer_number );
HYPRE_Int hypre_BoomerAMGDD_UnpackRecvBuffer_dbl  ( hypre_ParAMGDDData *amgdd_data,
                                               HYPRE_Int *recv_buffer, HYPRE_Int **A_tmp_info, HYPRE_Int *recv_map_send_buffer_size,
                                               HYPRE_Int *nodes_added_on_level, HYPRE_Int current_level, HYPRE_Int buffer_number );
HYPRE_Int hypre_BoomerAMGDD_UnpackRecvBuffer_long_dbl  ( hypre_ParAMGDDData *amgdd_data,
                                               HYPRE_Int *recv_buffer, HYPRE_Int **A_tmp_info, HYPRE_Int *recv_map_send_buffer_size,
                                               HYPRE_Int *nodes_added_on_level, HYPRE_Int current_level, HYPRE_Int buffer_number );
HYPRE_Int hypre_BoomerAMGDD_UnpackSendFlagBuffer_flt  ( hypre_AMGDDCompGrid **compGrid,
                                                   HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes,
                                                   HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_UnpackSendFlagBuffer_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                                   HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes,
                                                   HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels );
HYPRE_Int hypre_BoomerAMGDD_UnpackSendFlagBuffer_long_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                                   HYPRE_Int *send_flag_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes,
                                                   HYPRE_Int *send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels );
void *hypre_BoomerAMGDDCreate_flt  ( void );
void *hypre_BoomerAMGDDCreate_dbl  ( void );
void *hypre_BoomerAMGDDCreate_long_dbl  ( void );
HYPRE_Int hypre_BoomerAMGDDDestroy_flt  ( void *data );
HYPRE_Int hypre_BoomerAMGDDDestroy_dbl  ( void *data );
HYPRE_Int hypre_BoomerAMGDDDestroy_long_dbl  ( void *data );
HYPRE_Int hypre_BoomerAMGDDGetAMG_flt  ( void *data, void **amg_solver );
HYPRE_Int hypre_BoomerAMGDDGetAMG_dbl  ( void *data, void **amg_solver );
HYPRE_Int hypre_BoomerAMGDDGetAMG_long_dbl  ( void *data, void **amg_solver );
HYPRE_Int hypre_BoomerAMGDDGetFACCycleType_flt  ( void *data, HYPRE_Int *fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDGetFACCycleType_dbl  ( void *data, HYPRE_Int *fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDGetFACCycleType_long_dbl  ( void *data, HYPRE_Int *fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDGetFACNumCycles_flt  ( void *data, HYPRE_Int *fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDGetFACNumCycles_dbl  ( void *data, HYPRE_Int *fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDGetFACNumCycles_long_dbl  ( void *data, HYPRE_Int *fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDGetFACNumRelax_flt  ( void *data, HYPRE_Int *fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDGetFACNumRelax_dbl  ( void *data, HYPRE_Int *fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDGetFACNumRelax_long_dbl  ( void *data, HYPRE_Int *fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxType_flt  ( void *data, HYPRE_Int *fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxType_dbl  ( void *data, HYPRE_Int *fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxType_long_dbl  ( void *data, HYPRE_Int *fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxWeight_flt  ( void *data, hypre_float *fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxWeight_dbl  ( void *data, hypre_double *fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDGetFACRelaxWeight_long_dbl  ( void *data, hypre_long_double *fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDGetNumGhostLayers_flt  ( void *data, HYPRE_Int *num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDGetNumGhostLayers_dbl  ( void *data, HYPRE_Int *num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDGetNumGhostLayers_long_dbl  ( void *data, HYPRE_Int *num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDGetPadding_flt  ( void *data, HYPRE_Int *padding );
HYPRE_Int hypre_BoomerAMGDDGetPadding_dbl  ( void *data, HYPRE_Int *padding );
HYPRE_Int hypre_BoomerAMGDDGetPadding_long_dbl  ( void *data, HYPRE_Int *padding );
HYPRE_Int hypre_BoomerAMGDDGetStartLevel_flt  ( void *data, HYPRE_Int *start_level );
HYPRE_Int hypre_BoomerAMGDDGetStartLevel_dbl  ( void *data, HYPRE_Int *start_level );
HYPRE_Int hypre_BoomerAMGDDGetStartLevel_long_dbl  ( void *data, HYPRE_Int *start_level );
HYPRE_Int hypre_BoomerAMGDDSetFACCycleType_flt  ( void *data, HYPRE_Int fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDSetFACCycleType_dbl  ( void *data, HYPRE_Int fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDSetFACCycleType_long_dbl  ( void *data, HYPRE_Int fac_cycle_type );
HYPRE_Int hypre_BoomerAMGDDSetFACNumCycles_flt  ( void *data, HYPRE_Int fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDSetFACNumCycles_dbl  ( void *data, HYPRE_Int fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDSetFACNumCycles_long_dbl  ( void *data, HYPRE_Int fac_num_cycles );
HYPRE_Int hypre_BoomerAMGDDSetFACNumRelax_flt  ( void *data, HYPRE_Int fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDSetFACNumRelax_dbl  ( void *data, HYPRE_Int fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDSetFACNumRelax_long_dbl  ( void *data, HYPRE_Int fac_num_relax );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxType_flt  ( void *data, HYPRE_Int fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxType_dbl  ( void *data, HYPRE_Int fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxType_long_dbl  ( void *data, HYPRE_Int fac_relax_type );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxWeight_flt  ( void *data, hypre_float fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxWeight_dbl  ( void *data, hypre_double fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDSetFACRelaxWeight_long_dbl  ( void *data, hypre_long_double fac_relax_weight );
HYPRE_Int hypre_BoomerAMGDDSetNumGhostLayers_flt  ( void *data, HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDSetNumGhostLayers_dbl  ( void *data, HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDSetNumGhostLayers_long_dbl  ( void *data, HYPRE_Int num_ghost_layers );
HYPRE_Int hypre_BoomerAMGDDSetPadding_flt  ( void *data, HYPRE_Int padding );
HYPRE_Int hypre_BoomerAMGDDSetPadding_dbl  ( void *data, HYPRE_Int padding );
HYPRE_Int hypre_BoomerAMGDDSetPadding_long_dbl  ( void *data, HYPRE_Int padding );
HYPRE_Int hypre_BoomerAMGDDSetStartLevel_flt  ( void *data, HYPRE_Int start_level );
HYPRE_Int hypre_BoomerAMGDDSetStartLevel_dbl  ( void *data, HYPRE_Int start_level );
HYPRE_Int hypre_BoomerAMGDDSetStartLevel_long_dbl  ( void *data, HYPRE_Int start_level );
HYPRE_Int hypre_BoomerAMGDDSetUserFACRelaxation_flt ( void *data,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int hypre_BoomerAMGDDSetUserFACRelaxation_dbl ( void *data,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int hypre_BoomerAMGDDSetUserFACRelaxation_long_dbl ( void *data,
                                                 HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ) );
HYPRE_Int hypre_BoomerAMGDDSetup_flt  ( void *amgdd_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGDDSetup_dbl  ( void *amgdd_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGDDSetup_long_dbl  ( void *amgdd_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
hypre_float* hypre_BoomerAMGDD_PackResidualBuffer_flt  ( hypre_AMGDDCompGrid **compGrid,
                                                      hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );
hypre_double* hypre_BoomerAMGDD_PackResidualBuffer_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                                      hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );
hypre_long_double* hypre_BoomerAMGDD_PackResidualBuffer_long_dbl  ( hypre_AMGDDCompGrid **compGrid,
                                                      hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level, HYPRE_Int proc );
HYPRE_Int hypre_BoomerAMGDD_ResidualCommunication_flt  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_BoomerAMGDD_ResidualCommunication_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_BoomerAMGDD_ResidualCommunication_long_dbl  ( hypre_ParAMGDDData *amgdd_data );
HYPRE_Int hypre_BoomerAMGDDSolve_flt  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGDDSolve_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGDDSolve_long_dbl  ( void *solver, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                                   hypre_ParVector *x );
HYPRE_Int hypre_BoomerAMGDD_UnpackResidualBuffer_flt  ( hypre_float *buffer,
                                                   hypre_AMGDDCompGrid **compGrid, hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level,
                                                   HYPRE_Int proc );
HYPRE_Int hypre_BoomerAMGDD_UnpackResidualBuffer_dbl  ( hypre_double *buffer,
                                                   hypre_AMGDDCompGrid **compGrid, hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level,
                                                   HYPRE_Int proc );
HYPRE_Int hypre_BoomerAMGDD_UnpackResidualBuffer_long_dbl  ( hypre_long_double *buffer,
                                                   hypre_AMGDDCompGrid **compGrid, hypre_AMGDDCommPkg *compGridCommPkg, HYPRE_Int current_level,
                                                   HYPRE_Int proc );
void *hypre_BoomerAMGCreate_flt  ( void );
void *hypre_BoomerAMGCreate_dbl  ( void );
void *hypre_BoomerAMGCreate_long_dbl  ( void );
HYPRE_Int hypre_BoomerAMGDestroy_flt  ( void *data );
HYPRE_Int hypre_BoomerAMGDestroy_dbl  ( void *data );
HYPRE_Int hypre_BoomerAMGDestroy_long_dbl  ( void *data );
HYPRE_Int hypre_BoomerAMGGetAdditive_flt  ( void *data, HYPRE_Int *additive );
HYPRE_Int hypre_BoomerAMGGetAdditive_dbl  ( void *data, HYPRE_Int *additive );
HYPRE_Int hypre_BoomerAMGGetAdditive_long_dbl  ( void *data, HYPRE_Int *additive );
HYPRE_Int hypre_BoomerAMGGetCoarsenCutFactor_flt ( void *data, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGGetCoarsenCutFactor_dbl ( void *data, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGGetCoarsenCutFactor_long_dbl ( void *data, HYPRE_Int *coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGGetCoarsenType_flt  ( void *data, HYPRE_Int *coarsen_type );
HYPRE_Int hypre_BoomerAMGGetCoarsenType_dbl  ( void *data, HYPRE_Int *coarsen_type );
HYPRE_Int hypre_BoomerAMGGetCoarsenType_long_dbl  ( void *data, HYPRE_Int *coarsen_type );
HYPRE_Int hypre_BoomerAMGGetConvergeType_flt  ( void *data, HYPRE_Int *type );
HYPRE_Int hypre_BoomerAMGGetConvergeType_dbl  ( void *data, HYPRE_Int *type );
HYPRE_Int hypre_BoomerAMGGetConvergeType_long_dbl  ( void *data, HYPRE_Int *type );
HYPRE_Int hypre_BoomerAMGGetCumNnzAP_flt  ( void *data, hypre_float *cum_nnz_AP );
HYPRE_Int hypre_BoomerAMGGetCumNnzAP_dbl  ( void *data, hypre_double *cum_nnz_AP );
HYPRE_Int hypre_BoomerAMGGetCumNnzAP_long_dbl  ( void *data, hypre_long_double *cum_nnz_AP );
HYPRE_Int hypre_BoomerAMGGetCumNumIterations_flt  ( void *data, HYPRE_Int *cum_num_iterations );
HYPRE_Int hypre_BoomerAMGGetCumNumIterations_dbl  ( void *data, HYPRE_Int *cum_num_iterations );
HYPRE_Int hypre_BoomerAMGGetCumNumIterations_long_dbl  ( void *data, HYPRE_Int *cum_num_iterations );
HYPRE_Int hypre_BoomerAMGGetCycleNumSweeps_flt  ( void *data, HYPRE_Int *num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleNumSweeps_dbl  ( void *data, HYPRE_Int *num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleNumSweeps_long_dbl  ( void *data, HYPRE_Int *num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleRelaxType_flt  ( void *data, HYPRE_Int *relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleRelaxType_dbl  ( void *data, HYPRE_Int *relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleRelaxType_long_dbl  ( void *data, HYPRE_Int *relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGGetCycleType_flt  ( void *data, HYPRE_Int *cycle_type );
HYPRE_Int hypre_BoomerAMGGetCycleType_dbl  ( void *data, HYPRE_Int *cycle_type );
HYPRE_Int hypre_BoomerAMGGetCycleType_long_dbl  ( void *data, HYPRE_Int *cycle_type );
HYPRE_Int hypre_BoomerAMGGetDebugFlag_flt  ( void *data, HYPRE_Int *debug_flag );
HYPRE_Int hypre_BoomerAMGGetDebugFlag_dbl  ( void *data, HYPRE_Int *debug_flag );
HYPRE_Int hypre_BoomerAMGGetDebugFlag_long_dbl  ( void *data, HYPRE_Int *debug_flag );
HYPRE_Int hypre_BoomerAMGGetDomainType_flt  ( void *data, HYPRE_Int *domain_type );
HYPRE_Int hypre_BoomerAMGGetDomainType_dbl  ( void *data, HYPRE_Int *domain_type );
HYPRE_Int hypre_BoomerAMGGetDomainType_long_dbl  ( void *data, HYPRE_Int *domain_type );
HYPRE_Int hypre_BoomerAMGGetFCycle_flt  ( void *data, HYPRE_Int *fcycle );
HYPRE_Int hypre_BoomerAMGGetFCycle_dbl  ( void *data, HYPRE_Int *fcycle );
HYPRE_Int hypre_BoomerAMGGetFCycle_long_dbl  ( void *data, HYPRE_Int *fcycle );
HYPRE_Int hypre_BoomerAMGGetFilterThresholdR_flt  ( void *data, hypre_float *filter_threshold );
HYPRE_Int hypre_BoomerAMGGetFilterThresholdR_dbl  ( void *data, hypre_double *filter_threshold );
HYPRE_Int hypre_BoomerAMGGetFilterThresholdR_long_dbl  ( void *data, hypre_long_double *filter_threshold );
HYPRE_Int hypre_BoomerAMGGetGridHierarchy_flt (void *data, HYPRE_Int *cgrid );
HYPRE_Int hypre_BoomerAMGGetGridHierarchy_dbl (void *data, HYPRE_Int *cgrid );
HYPRE_Int hypre_BoomerAMGGetGridHierarchy_long_dbl (void *data, HYPRE_Int *cgrid );
HYPRE_Int hypre_BoomerAMGGetGridRelaxPoints_flt  ( void *data, HYPRE_Int ***grid_relax_points );
HYPRE_Int hypre_BoomerAMGGetGridRelaxPoints_dbl  ( void *data, HYPRE_Int ***grid_relax_points );
HYPRE_Int hypre_BoomerAMGGetGridRelaxPoints_long_dbl  ( void *data, HYPRE_Int ***grid_relax_points );
HYPRE_Int hypre_BoomerAMGGetGridRelaxType_flt  ( void *data, HYPRE_Int **grid_relax_type );
HYPRE_Int hypre_BoomerAMGGetGridRelaxType_dbl  ( void *data, HYPRE_Int **grid_relax_type );
HYPRE_Int hypre_BoomerAMGGetGridRelaxType_long_dbl  ( void *data, HYPRE_Int **grid_relax_type );
HYPRE_Int hypre_BoomerAMGGetInterpType_flt  ( void *data, HYPRE_Int *interp_type );
HYPRE_Int hypre_BoomerAMGGetInterpType_dbl  ( void *data, HYPRE_Int *interp_type );
HYPRE_Int hypre_BoomerAMGGetInterpType_long_dbl  ( void *data, HYPRE_Int *interp_type );
HYPRE_Int hypre_BoomerAMGGetJacobiTruncThreshold_flt  ( void *data, hypre_float *jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGGetJacobiTruncThreshold_dbl  ( void *data, hypre_double *jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGGetJacobiTruncThreshold_long_dbl  ( void *data, hypre_long_double *jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGGetLevelOuterWt_flt  ( void *data, hypre_float *omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelOuterWt_dbl  ( void *data, hypre_double *omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelOuterWt_long_dbl  ( void *data, hypre_long_double *omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelRelaxWt_flt  ( void *data, hypre_float *relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelRelaxWt_dbl  ( void *data, hypre_double *relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLevelRelaxWt_long_dbl  ( void *data, hypre_long_double *relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGGetLogging_flt  ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_BoomerAMGGetLogging_dbl  ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_BoomerAMGGetLogging_long_dbl  ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_BoomerAMGGetMaxCoarseSize_flt  ( void *data, HYPRE_Int *max_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMaxCoarseSize_dbl  ( void *data, HYPRE_Int *max_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMaxCoarseSize_long_dbl  ( void *data, HYPRE_Int *max_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMaxIter_flt  ( void *data, HYPRE_Int *max_iter );
HYPRE_Int hypre_BoomerAMGGetMaxIter_dbl  ( void *data, HYPRE_Int *max_iter );
HYPRE_Int hypre_BoomerAMGGetMaxIter_long_dbl  ( void *data, HYPRE_Int *max_iter );
HYPRE_Int hypre_BoomerAMGGetMaxLevels_flt  ( void *data, HYPRE_Int *max_levels );
HYPRE_Int hypre_BoomerAMGGetMaxLevels_dbl  ( void *data, HYPRE_Int *max_levels );
HYPRE_Int hypre_BoomerAMGGetMaxLevels_long_dbl  ( void *data, HYPRE_Int *max_levels );
HYPRE_Int hypre_BoomerAMGGetMaxRowSum_flt  ( void *data, hypre_float *max_row_sum );
HYPRE_Int hypre_BoomerAMGGetMaxRowSum_dbl  ( void *data, hypre_double *max_row_sum );
HYPRE_Int hypre_BoomerAMGGetMaxRowSum_long_dbl  ( void *data, hypre_long_double *max_row_sum );
HYPRE_Int hypre_BoomerAMGGetMeasureType_flt  ( void *data, HYPRE_Int *measure_type );
HYPRE_Int hypre_BoomerAMGGetMeasureType_dbl  ( void *data, HYPRE_Int *measure_type );
HYPRE_Int hypre_BoomerAMGGetMeasureType_long_dbl  ( void *data, HYPRE_Int *measure_type );
HYPRE_Int hypre_BoomerAMGGetMinCoarseSize_flt  ( void *data, HYPRE_Int *min_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMinCoarseSize_dbl  ( void *data, HYPRE_Int *min_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMinCoarseSize_long_dbl  ( void *data, HYPRE_Int *min_coarse_size );
HYPRE_Int hypre_BoomerAMGGetMinIter_flt  ( void *data, HYPRE_Int *min_iter );
HYPRE_Int hypre_BoomerAMGGetMinIter_dbl  ( void *data, HYPRE_Int *min_iter );
HYPRE_Int hypre_BoomerAMGGetMinIter_long_dbl  ( void *data, HYPRE_Int *min_iter );
HYPRE_Int hypre_BoomerAMGGetMultAdditive_flt  ( void *data, HYPRE_Int *mult_additive );
HYPRE_Int hypre_BoomerAMGGetMultAdditive_dbl  ( void *data, HYPRE_Int *mult_additive );
HYPRE_Int hypre_BoomerAMGGetMultAdditive_long_dbl  ( void *data, HYPRE_Int *mult_additive );
HYPRE_Int hypre_BoomerAMGGetNumFunctions_flt  ( void *data, HYPRE_Int *num_functions );
HYPRE_Int hypre_BoomerAMGGetNumFunctions_dbl  ( void *data, HYPRE_Int *num_functions );
HYPRE_Int hypre_BoomerAMGGetNumFunctions_long_dbl  ( void *data, HYPRE_Int *num_functions );
HYPRE_Int hypre_BoomerAMGGetNumGridSweeps_flt  ( void *data, HYPRE_Int **num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGGetNumGridSweeps_dbl  ( void *data, HYPRE_Int **num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGGetNumGridSweeps_long_dbl  ( void *data, HYPRE_Int **num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGGetNumIterations_flt  ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BoomerAMGGetNumIterations_dbl  ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BoomerAMGGetNumIterations_long_dbl  ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BoomerAMGGetOmega_flt  ( void *data, hypre_float **omega );
HYPRE_Int hypre_BoomerAMGGetOmega_dbl  ( void *data, hypre_double **omega );
HYPRE_Int hypre_BoomerAMGGetOmega_long_dbl  ( void *data, hypre_long_double **omega );
HYPRE_Int hypre_BoomerAMGGetOverlap_flt  ( void *data, HYPRE_Int *overlap );
HYPRE_Int hypre_BoomerAMGGetOverlap_dbl  ( void *data, HYPRE_Int *overlap );
HYPRE_Int hypre_BoomerAMGGetOverlap_long_dbl  ( void *data, HYPRE_Int *overlap );
HYPRE_Int hypre_BoomerAMGGetPMaxElmts_flt  ( void *data, HYPRE_Int *P_max_elmts );
HYPRE_Int hypre_BoomerAMGGetPMaxElmts_dbl  ( void *data, HYPRE_Int *P_max_elmts );
HYPRE_Int hypre_BoomerAMGGetPMaxElmts_long_dbl  ( void *data, HYPRE_Int *P_max_elmts );
HYPRE_Int hypre_BoomerAMGGetPostInterpType_flt  ( void *data, HYPRE_Int *post_interp_type );
HYPRE_Int hypre_BoomerAMGGetPostInterpType_dbl  ( void *data, HYPRE_Int *post_interp_type );
HYPRE_Int hypre_BoomerAMGGetPostInterpType_long_dbl  ( void *data, HYPRE_Int *post_interp_type );
HYPRE_Int hypre_BoomerAMGGetPrintFileName_flt  ( void *data, char **print_file_name );
HYPRE_Int hypre_BoomerAMGGetPrintFileName_dbl  ( void *data, char **print_file_name );
HYPRE_Int hypre_BoomerAMGGetPrintFileName_long_dbl  ( void *data, char **print_file_name );
HYPRE_Int hypre_BoomerAMGGetPrintLevel_flt  ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_BoomerAMGGetPrintLevel_dbl  ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_BoomerAMGGetPrintLevel_long_dbl  ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_BoomerAMGGetRedundant_flt  ( void *data, HYPRE_Int *redundant );
HYPRE_Int hypre_BoomerAMGGetRedundant_dbl  ( void *data, HYPRE_Int *redundant );
HYPRE_Int hypre_BoomerAMGGetRedundant_long_dbl  ( void *data, HYPRE_Int *redundant );
HYPRE_Int hypre_BoomerAMGGetRelaxOrder_flt  ( void *data, HYPRE_Int *relax_order );
HYPRE_Int hypre_BoomerAMGGetRelaxOrder_dbl  ( void *data, HYPRE_Int *relax_order );
HYPRE_Int hypre_BoomerAMGGetRelaxOrder_long_dbl  ( void *data, HYPRE_Int *relax_order );
HYPRE_Int hypre_BoomerAMGGetRelaxWeight_flt  ( void *data, hypre_float **relax_weight );
HYPRE_Int hypre_BoomerAMGGetRelaxWeight_dbl  ( void *data, hypre_double **relax_weight );
HYPRE_Int hypre_BoomerAMGGetRelaxWeight_long_dbl  ( void *data, hypre_long_double **relax_weight );
HYPRE_Int hypre_BoomerAMGGetRelResidualNorm_flt  ( void *data, hypre_float *rel_resid_norm );
HYPRE_Int hypre_BoomerAMGGetRelResidualNorm_dbl  ( void *data, hypre_double *rel_resid_norm );
HYPRE_Int hypre_BoomerAMGGetRelResidualNorm_long_dbl  ( void *data, hypre_long_double *rel_resid_norm );
HYPRE_Int hypre_BoomerAMGGetResidual_flt  ( void *data, hypre_ParVector **resid );
HYPRE_Int hypre_BoomerAMGGetResidual_dbl  ( void *data, hypre_ParVector **resid );
HYPRE_Int hypre_BoomerAMGGetResidual_long_dbl  ( void *data, hypre_ParVector **resid );
HYPRE_Int hypre_BoomerAMGGetSchwarzRlxWeight_flt  ( void *data, hypre_float *schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGGetSchwarzRlxWeight_dbl  ( void *data, hypre_double *schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGGetSchwarzRlxWeight_long_dbl  ( void *data, hypre_long_double *schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGGetSeqThreshold_flt  ( void *data, HYPRE_Int *seq_threshold );
HYPRE_Int hypre_BoomerAMGGetSeqThreshold_dbl  ( void *data, HYPRE_Int *seq_threshold );
HYPRE_Int hypre_BoomerAMGGetSeqThreshold_long_dbl  ( void *data, HYPRE_Int *seq_threshold );
HYPRE_Int hypre_BoomerAMGGetSetupType_flt  ( void *data, HYPRE_Int *setup_type );
HYPRE_Int hypre_BoomerAMGGetSetupType_dbl  ( void *data, HYPRE_Int *setup_type );
HYPRE_Int hypre_BoomerAMGGetSetupType_long_dbl  ( void *data, HYPRE_Int *setup_type );
HYPRE_Int hypre_BoomerAMGGetSimple_flt  ( void *data, HYPRE_Int *simple );
HYPRE_Int hypre_BoomerAMGGetSimple_dbl  ( void *data, HYPRE_Int *simple );
HYPRE_Int hypre_BoomerAMGGetSimple_long_dbl  ( void *data, HYPRE_Int *simple );
HYPRE_Int hypre_BoomerAMGGetSmoothNumLevels_flt  ( void *data, HYPRE_Int *smooth_num_levels );
HYPRE_Int hypre_BoomerAMGGetSmoothNumLevels_dbl  ( void *data, HYPRE_Int *smooth_num_levels );
HYPRE_Int hypre_BoomerAMGGetSmoothNumLevels_long_dbl  ( void *data, HYPRE_Int *smooth_num_levels );
HYPRE_Int hypre_BoomerAMGGetSmoothNumSweeps_flt  ( void *data, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGGetSmoothNumSweeps_dbl  ( void *data, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGGetSmoothNumSweeps_long_dbl  ( void *data, HYPRE_Int *smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGGetSmoothType_flt  ( void *data, HYPRE_Int *smooth_type );
HYPRE_Int hypre_BoomerAMGGetSmoothType_dbl  ( void *data, HYPRE_Int *smooth_type );
HYPRE_Int hypre_BoomerAMGGetSmoothType_long_dbl  ( void *data, HYPRE_Int *smooth_type );
HYPRE_Int hypre_BoomerAMGGetStrongThreshold_flt  ( void *data, hypre_float *strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThreshold_dbl  ( void *data, hypre_double *strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThreshold_long_dbl  ( void *data, hypre_long_double *strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThresholdR_flt  ( void *data, hypre_float *strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThresholdR_dbl  ( void *data, hypre_double *strong_threshold );
HYPRE_Int hypre_BoomerAMGGetStrongThresholdR_long_dbl  ( void *data, hypre_long_double *strong_threshold );
HYPRE_Int hypre_BoomerAMGGetTol_flt  ( void *data, hypre_float *tol );
HYPRE_Int hypre_BoomerAMGGetTol_dbl  ( void *data, hypre_double *tol );
HYPRE_Int hypre_BoomerAMGGetTol_long_dbl  ( void *data, hypre_long_double *tol );
HYPRE_Int hypre_BoomerAMGGetTruncFactor_flt  ( void *data, hypre_float *trunc_factor );
HYPRE_Int hypre_BoomerAMGGetTruncFactor_dbl  ( void *data, hypre_double *trunc_factor );
HYPRE_Int hypre_BoomerAMGGetTruncFactor_long_dbl  ( void *data, hypre_long_double *trunc_factor );
HYPRE_Int hypre_BoomerAMGGetVariant_flt  ( void *data, HYPRE_Int *variant );
HYPRE_Int hypre_BoomerAMGGetVariant_dbl  ( void *data, HYPRE_Int *variant );
HYPRE_Int hypre_BoomerAMGGetVariant_long_dbl  ( void *data, HYPRE_Int *variant );
HYPRE_Int hypre_BoomerAMGSetAdditive_flt  ( void *data, HYPRE_Int additive );
HYPRE_Int hypre_BoomerAMGSetAdditive_dbl  ( void *data, HYPRE_Int additive );
HYPRE_Int hypre_BoomerAMGSetAdditive_long_dbl  ( void *data, HYPRE_Int additive );
HYPRE_Int hypre_BoomerAMGSetAddLastLvl_flt  ( void *data, HYPRE_Int add_last_lvl );
HYPRE_Int hypre_BoomerAMGSetAddLastLvl_dbl  ( void *data, HYPRE_Int add_last_lvl );
HYPRE_Int hypre_BoomerAMGSetAddLastLvl_long_dbl  ( void *data, HYPRE_Int add_last_lvl );
HYPRE_Int hypre_BoomerAMGSetAddRelaxType_flt  ( void *data, HYPRE_Int add_rlx_type );
HYPRE_Int hypre_BoomerAMGSetAddRelaxType_dbl  ( void *data, HYPRE_Int add_rlx_type );
HYPRE_Int hypre_BoomerAMGSetAddRelaxType_long_dbl  ( void *data, HYPRE_Int add_rlx_type );
HYPRE_Int hypre_BoomerAMGSetAddRelaxWt_flt  ( void *data, hypre_float add_rlx_wt );
HYPRE_Int hypre_BoomerAMGSetAddRelaxWt_dbl  ( void *data, hypre_double add_rlx_wt );
HYPRE_Int hypre_BoomerAMGSetAddRelaxWt_long_dbl  ( void *data, hypre_long_double add_rlx_wt );
HYPRE_Int hypre_BoomerAMGSetADropTol_flt ( void     *data, hypre_float  A_drop_tol );
HYPRE_Int hypre_BoomerAMGSetADropTol_dbl ( void     *data, hypre_double  A_drop_tol );
HYPRE_Int hypre_BoomerAMGSetADropTol_long_dbl ( void     *data, hypre_long_double  A_drop_tol );
HYPRE_Int hypre_BoomerAMGSetADropType_flt ( void     *data, HYPRE_Int  A_drop_type );
HYPRE_Int hypre_BoomerAMGSetADropType_dbl ( void     *data, HYPRE_Int  A_drop_type );
HYPRE_Int hypre_BoomerAMGSetADropType_long_dbl ( void     *data, HYPRE_Int  A_drop_type );
HYPRE_Int hypre_BoomerAMGSetAggInterpType_flt  ( void *data, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_BoomerAMGSetAggInterpType_dbl  ( void *data, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_BoomerAMGSetAggInterpType_long_dbl  ( void *data, HYPRE_Int agg_interp_type );
HYPRE_Int hypre_BoomerAMGSetAggNumLevels_flt  ( void *data, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_BoomerAMGSetAggNumLevels_dbl  ( void *data, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_BoomerAMGSetAggNumLevels_long_dbl  ( void *data, HYPRE_Int agg_num_levels );
HYPRE_Int hypre_BoomerAMGSetAggP12MaxElmts_flt  ( void *data, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggP12MaxElmts_dbl  ( void *data, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggP12MaxElmts_long_dbl  ( void *data, HYPRE_Int agg_P12_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggP12TruncFactor_flt  ( void *data, hypre_float agg_P12_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetAggP12TruncFactor_dbl  ( void *data, hypre_double agg_P12_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetAggP12TruncFactor_long_dbl  ( void *data, hypre_long_double agg_P12_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetAggPMaxElmts_flt  ( void *data, HYPRE_Int agg_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggPMaxElmts_dbl  ( void *data, HYPRE_Int agg_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggPMaxElmts_long_dbl  ( void *data, HYPRE_Int agg_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetAggTruncFactor_flt  ( void *data, hypre_float agg_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetAggTruncFactor_dbl  ( void *data, hypre_double agg_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetAggTruncFactor_long_dbl  ( void *data, hypre_long_double agg_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetCGCIts_flt  ( void *data, HYPRE_Int its );
HYPRE_Int hypre_BoomerAMGSetCGCIts_dbl  ( void *data, HYPRE_Int its );
HYPRE_Int hypre_BoomerAMGSetCGCIts_long_dbl  ( void *data, HYPRE_Int its );
HYPRE_Int hypre_BoomerAMGSetChebyEigEst_flt  ( void *data, HYPRE_Int eig_est );
HYPRE_Int hypre_BoomerAMGSetChebyEigEst_dbl  ( void *data, HYPRE_Int eig_est );
HYPRE_Int hypre_BoomerAMGSetChebyEigEst_long_dbl  ( void *data, HYPRE_Int eig_est );
HYPRE_Int hypre_BoomerAMGSetChebyFraction_flt  ( void *data, hypre_float ratio );
HYPRE_Int hypre_BoomerAMGSetChebyFraction_dbl  ( void *data, hypre_double ratio );
HYPRE_Int hypre_BoomerAMGSetChebyFraction_long_dbl  ( void *data, hypre_long_double ratio );
HYPRE_Int hypre_BoomerAMGSetChebyOrder_flt  ( void *data, HYPRE_Int order );
HYPRE_Int hypre_BoomerAMGSetChebyOrder_dbl  ( void *data, HYPRE_Int order );
HYPRE_Int hypre_BoomerAMGSetChebyOrder_long_dbl  ( void *data, HYPRE_Int order );
HYPRE_Int hypre_BoomerAMGSetChebyScale_flt  ( void *data, HYPRE_Int scale );
HYPRE_Int hypre_BoomerAMGSetChebyScale_dbl  ( void *data, HYPRE_Int scale );
HYPRE_Int hypre_BoomerAMGSetChebyScale_long_dbl  ( void *data, HYPRE_Int scale );
HYPRE_Int hypre_BoomerAMGSetChebyVariant_flt  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetChebyVariant_dbl  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetChebyVariant_long_dbl  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetCoarsenCutFactor_flt ( void *data, HYPRE_Int coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGSetCoarsenCutFactor_dbl ( void *data, HYPRE_Int coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGSetCoarsenCutFactor_long_dbl ( void *data, HYPRE_Int coarsen_cut_factor );
HYPRE_Int hypre_BoomerAMGSetCoarsenType_flt  ( void *data, HYPRE_Int coarsen_type );
HYPRE_Int hypre_BoomerAMGSetCoarsenType_dbl  ( void *data, HYPRE_Int coarsen_type );
HYPRE_Int hypre_BoomerAMGSetCoarsenType_long_dbl  ( void *data, HYPRE_Int coarsen_type );
HYPRE_Int hypre_BoomerAMGSetConvergeType_flt  ( void *data, HYPRE_Int type );
HYPRE_Int hypre_BoomerAMGSetConvergeType_dbl  ( void *data, HYPRE_Int type );
HYPRE_Int hypre_BoomerAMGSetConvergeType_long_dbl  ( void *data, HYPRE_Int type );
HYPRE_Int hypre_BoomerAMGSetCoordDim_flt  ( void *data, HYPRE_Int coorddim );
HYPRE_Int hypre_BoomerAMGSetCoordDim_dbl  ( void *data, HYPRE_Int coorddim );
HYPRE_Int hypre_BoomerAMGSetCoordDim_long_dbl  ( void *data, HYPRE_Int coorddim );
HYPRE_Int hypre_BoomerAMGSetCoordinates_flt  ( void *data, float *coordinates );
HYPRE_Int hypre_BoomerAMGSetCoordinates_dbl  ( void *data, float *coordinates );
HYPRE_Int hypre_BoomerAMGSetCoordinates_long_dbl  ( void *data, float *coordinates );
HYPRE_Int hypre_BoomerAMGSetCPoints_flt ( void *data, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int  num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index );
HYPRE_Int hypre_BoomerAMGSetCPoints_dbl ( void *data, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int  num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index );
HYPRE_Int hypre_BoomerAMGSetCPoints_long_dbl ( void *data, HYPRE_Int cpt_coarse_level,
                                     HYPRE_Int  num_cpt_coarse, HYPRE_BigInt *cpt_coarse_index );
HYPRE_Int hypre_BoomerAMGSetCRRate_flt  ( void *data, hypre_float CR_rate );
HYPRE_Int hypre_BoomerAMGSetCRRate_dbl  ( void *data, hypre_double CR_rate );
HYPRE_Int hypre_BoomerAMGSetCRRate_long_dbl  ( void *data, hypre_long_double CR_rate );
HYPRE_Int hypre_BoomerAMGSetCRStrongTh_flt  ( void *data, hypre_float CR_strong_th );
HYPRE_Int hypre_BoomerAMGSetCRStrongTh_dbl  ( void *data, hypre_double CR_strong_th );
HYPRE_Int hypre_BoomerAMGSetCRStrongTh_long_dbl  ( void *data, hypre_long_double CR_strong_th );
HYPRE_Int hypre_BoomerAMGSetCRUseCG_flt  ( void *data, HYPRE_Int CR_use_CG );
HYPRE_Int hypre_BoomerAMGSetCRUseCG_dbl  ( void *data, HYPRE_Int CR_use_CG );
HYPRE_Int hypre_BoomerAMGSetCRUseCG_long_dbl  ( void *data, HYPRE_Int CR_use_CG );
HYPRE_Int hypre_BoomerAMGSetCumNnzAP_flt  ( void *data, hypre_float cum_nnz_AP );
HYPRE_Int hypre_BoomerAMGSetCumNnzAP_dbl  ( void *data, hypre_double cum_nnz_AP );
HYPRE_Int hypre_BoomerAMGSetCumNnzAP_long_dbl  ( void *data, hypre_long_double cum_nnz_AP );
HYPRE_Int hypre_BoomerAMGSetCycleNumSweeps_flt  ( void *data, HYPRE_Int num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetCycleNumSweeps_dbl  ( void *data, HYPRE_Int num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetCycleNumSweeps_long_dbl  ( void *data, HYPRE_Int num_sweeps, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetCycleRelaxType_flt  ( void *data, HYPRE_Int relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetCycleRelaxType_dbl  ( void *data, HYPRE_Int relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetCycleRelaxType_long_dbl  ( void *data, HYPRE_Int relax_type, HYPRE_Int k );
HYPRE_Int hypre_BoomerAMGSetCycleType_flt  ( void *data, HYPRE_Int cycle_type );
HYPRE_Int hypre_BoomerAMGSetCycleType_dbl  ( void *data, HYPRE_Int cycle_type );
HYPRE_Int hypre_BoomerAMGSetCycleType_long_dbl  ( void *data, HYPRE_Int cycle_type );
HYPRE_Int hypre_BoomerAMGSetDebugFlag_flt  ( void *data, HYPRE_Int debug_flag );
HYPRE_Int hypre_BoomerAMGSetDebugFlag_dbl  ( void *data, HYPRE_Int debug_flag );
HYPRE_Int hypre_BoomerAMGSetDebugFlag_long_dbl  ( void *data, HYPRE_Int debug_flag );
HYPRE_Int hypre_BoomerAMGSetDofFunc_flt  ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_BoomerAMGSetDofFunc_dbl  ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_BoomerAMGSetDofFunc_long_dbl  ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_BoomerAMGSetDofPoint_flt  ( void *data, HYPRE_Int *dof_point );
HYPRE_Int hypre_BoomerAMGSetDofPoint_dbl  ( void *data, HYPRE_Int *dof_point );
HYPRE_Int hypre_BoomerAMGSetDofPoint_long_dbl  ( void *data, HYPRE_Int *dof_point );
HYPRE_Int hypre_BoomerAMGSetDomainType_flt  ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_BoomerAMGSetDomainType_dbl  ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_BoomerAMGSetDomainType_long_dbl  ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_BoomerAMGSetDropTol_flt  ( void *data, hypre_float drop_tol );
HYPRE_Int hypre_BoomerAMGSetDropTol_dbl  ( void *data, hypre_double drop_tol );
HYPRE_Int hypre_BoomerAMGSetDropTol_long_dbl  ( void *data, hypre_long_double drop_tol );
HYPRE_Int hypre_BoomerAMGSetEuBJ_flt  ( void *data, HYPRE_Int eu_bj );
HYPRE_Int hypre_BoomerAMGSetEuBJ_dbl  ( void *data, HYPRE_Int eu_bj );
HYPRE_Int hypre_BoomerAMGSetEuBJ_long_dbl  ( void *data, HYPRE_Int eu_bj );
HYPRE_Int hypre_BoomerAMGSetEuclidFile_flt  ( void *data, char *euclidfile );
HYPRE_Int hypre_BoomerAMGSetEuclidFile_dbl  ( void *data, char *euclidfile );
HYPRE_Int hypre_BoomerAMGSetEuclidFile_long_dbl  ( void *data, char *euclidfile );
HYPRE_Int hypre_BoomerAMGSetEuLevel_flt  ( void *data, HYPRE_Int eu_level );
HYPRE_Int hypre_BoomerAMGSetEuLevel_dbl  ( void *data, HYPRE_Int eu_level );
HYPRE_Int hypre_BoomerAMGSetEuLevel_long_dbl  ( void *data, HYPRE_Int eu_level );
HYPRE_Int hypre_BoomerAMGSetEuSparseA_flt  ( void *data, hypre_float eu_sparse_A );
HYPRE_Int hypre_BoomerAMGSetEuSparseA_dbl  ( void *data, hypre_double eu_sparse_A );
HYPRE_Int hypre_BoomerAMGSetEuSparseA_long_dbl  ( void *data, hypre_long_double eu_sparse_A );
HYPRE_Int hypre_BoomerAMGSetFCycle_flt  ( void *data, HYPRE_Int fcycle );
HYPRE_Int hypre_BoomerAMGSetFCycle_dbl  ( void *data, HYPRE_Int fcycle );
HYPRE_Int hypre_BoomerAMGSetFCycle_long_dbl  ( void *data, HYPRE_Int fcycle );
HYPRE_Int hypre_BoomerAMGSetFilter_flt  ( void *data, hypre_float filter );
HYPRE_Int hypre_BoomerAMGSetFilter_dbl  ( void *data, hypre_double filter );
HYPRE_Int hypre_BoomerAMGSetFilter_long_dbl  ( void *data, hypre_long_double filter );
HYPRE_Int hypre_BoomerAMGSetFilterThresholdR_flt  ( void *data, hypre_float filter_threshold );
HYPRE_Int hypre_BoomerAMGSetFilterThresholdR_dbl  ( void *data, hypre_double filter_threshold );
HYPRE_Int hypre_BoomerAMGSetFilterThresholdR_long_dbl  ( void *data, hypre_long_double filter_threshold );
HYPRE_Int hypre_BoomerAMGSetFPoints_flt ( void *data, HYPRE_Int isolated, HYPRE_Int num_points,
                                     HYPRE_BigInt *indices );
HYPRE_Int hypre_BoomerAMGSetFPoints_dbl ( void *data, HYPRE_Int isolated, HYPRE_Int num_points,
                                     HYPRE_BigInt *indices );
HYPRE_Int hypre_BoomerAMGSetFPoints_long_dbl ( void *data, HYPRE_Int isolated, HYPRE_Int num_points,
                                     HYPRE_BigInt *indices );
HYPRE_Int hypre_BoomerAMGSetFSAIAlgoType_flt  ( void *data, HYPRE_Int fsai_algo_type );
HYPRE_Int hypre_BoomerAMGSetFSAIAlgoType_dbl  ( void *data, HYPRE_Int fsai_algo_type );
HYPRE_Int hypre_BoomerAMGSetFSAIAlgoType_long_dbl  ( void *data, HYPRE_Int fsai_algo_type );
HYPRE_Int hypre_BoomerAMGSetFSAIEigMaxIters_flt  ( void *data, HYPRE_Int fsai_eig_max_iters );
HYPRE_Int hypre_BoomerAMGSetFSAIEigMaxIters_dbl  ( void *data, HYPRE_Int fsai_eig_max_iters );
HYPRE_Int hypre_BoomerAMGSetFSAIEigMaxIters_long_dbl  ( void *data, HYPRE_Int fsai_eig_max_iters );
HYPRE_Int hypre_BoomerAMGSetFSAIKapTolerance_flt  ( void *data, hypre_float fsai_kap_tolerance );
HYPRE_Int hypre_BoomerAMGSetFSAIKapTolerance_dbl  ( void *data, hypre_double fsai_kap_tolerance );
HYPRE_Int hypre_BoomerAMGSetFSAIKapTolerance_long_dbl  ( void *data, hypre_long_double fsai_kap_tolerance );
HYPRE_Int hypre_BoomerAMGSetFSAILocalSolveType_flt  ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_BoomerAMGSetFSAILocalSolveType_dbl  ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_BoomerAMGSetFSAILocalSolveType_long_dbl  ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxNnzRow_flt  ( void *data, HYPRE_Int fsai_max_nnz_row );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxNnzRow_dbl  ( void *data, HYPRE_Int fsai_max_nnz_row );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxNnzRow_long_dbl  ( void *data, HYPRE_Int fsai_max_nnz_row );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxSteps_flt  ( void *data, HYPRE_Int fsai_max_steps );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxSteps_dbl  ( void *data, HYPRE_Int fsai_max_steps );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxSteps_long_dbl  ( void *data, HYPRE_Int fsai_max_steps );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxStepSize_flt  ( void *data, HYPRE_Int fsai_max_step_size );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxStepSize_dbl  ( void *data, HYPRE_Int fsai_max_step_size );
HYPRE_Int hypre_BoomerAMGSetFSAIMaxStepSize_long_dbl  ( void *data, HYPRE_Int fsai_max_step_size );
HYPRE_Int hypre_BoomerAMGSetFSAINumLevels_flt  ( void *data, HYPRE_Int fsai_num_levels );
HYPRE_Int hypre_BoomerAMGSetFSAINumLevels_dbl  ( void *data, HYPRE_Int fsai_num_levels );
HYPRE_Int hypre_BoomerAMGSetFSAINumLevels_long_dbl  ( void *data, HYPRE_Int fsai_num_levels );
HYPRE_Int hypre_BoomerAMGSetFSAIThreshold_flt  ( void *data, hypre_float fsai_threshold );
HYPRE_Int hypre_BoomerAMGSetFSAIThreshold_dbl  ( void *data, hypre_double fsai_threshold );
HYPRE_Int hypre_BoomerAMGSetFSAIThreshold_long_dbl  ( void *data, hypre_long_double fsai_threshold );
HYPRE_Int hypre_BoomerAMGSetGMRESSwitchR_flt  ( void *data, HYPRE_Int gmres_switch );
HYPRE_Int hypre_BoomerAMGSetGMRESSwitchR_dbl  ( void *data, HYPRE_Int gmres_switch );
HYPRE_Int hypre_BoomerAMGSetGMRESSwitchR_long_dbl  ( void *data, HYPRE_Int gmres_switch );
HYPRE_Int hypre_BoomerAMGSetGridRelaxPoints_flt  ( void *data, HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_BoomerAMGSetGridRelaxPoints_dbl  ( void *data, HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_BoomerAMGSetGridRelaxPoints_long_dbl  ( void *data, HYPRE_Int **grid_relax_points );
HYPRE_Int hypre_BoomerAMGSetGridRelaxType_flt  ( void *data, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_BoomerAMGSetGridRelaxType_dbl  ( void *data, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_BoomerAMGSetGridRelaxType_long_dbl  ( void *data, HYPRE_Int *grid_relax_type );
HYPRE_Int hypre_BoomerAMGSetGSMG_flt  ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetGSMG_dbl  ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetGSMG_long_dbl  ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetILUDroptol_flt ( void *data, hypre_float ilu_droptol );
HYPRE_Int hypre_BoomerAMGSetILUDroptol_dbl ( void *data, hypre_double ilu_droptol );
HYPRE_Int hypre_BoomerAMGSetILUDroptol_long_dbl ( void *data, hypre_long_double ilu_droptol );
HYPRE_Int hypre_BoomerAMGSetILULevel_flt ( void *data, HYPRE_Int ilu_lfil );
HYPRE_Int hypre_BoomerAMGSetILULevel_dbl ( void *data, HYPRE_Int ilu_lfil );
HYPRE_Int hypre_BoomerAMGSetILULevel_long_dbl ( void *data, HYPRE_Int ilu_lfil );
HYPRE_Int hypre_BoomerAMGSetILULocalReordering_flt ( void *data, HYPRE_Int ilu_reordering_type );
HYPRE_Int hypre_BoomerAMGSetILULocalReordering_dbl ( void *data, HYPRE_Int ilu_reordering_type );
HYPRE_Int hypre_BoomerAMGSetILULocalReordering_long_dbl ( void *data, HYPRE_Int ilu_reordering_type );
HYPRE_Int hypre_BoomerAMGSetILULowerJacobiIters_flt ( void *data, HYPRE_Int ilu_lower_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetILULowerJacobiIters_dbl ( void *data, HYPRE_Int ilu_lower_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetILULowerJacobiIters_long_dbl ( void *data, HYPRE_Int ilu_lower_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetILUMaxIter_flt ( void *data, HYPRE_Int ilu_max_iter );
HYPRE_Int hypre_BoomerAMGSetILUMaxIter_dbl ( void *data, HYPRE_Int ilu_max_iter );
HYPRE_Int hypre_BoomerAMGSetILUMaxIter_long_dbl ( void *data, HYPRE_Int ilu_max_iter );
HYPRE_Int hypre_BoomerAMGSetILUMaxRowNnz_flt ( void *data, HYPRE_Int ilu_max_row_nnz );
HYPRE_Int hypre_BoomerAMGSetILUMaxRowNnz_dbl ( void *data, HYPRE_Int ilu_max_row_nnz );
HYPRE_Int hypre_BoomerAMGSetILUMaxRowNnz_long_dbl ( void *data, HYPRE_Int ilu_max_row_nnz );
HYPRE_Int hypre_BoomerAMGSetILUTriSolve_flt ( void *data, HYPRE_Int ilu_tri_solve );
HYPRE_Int hypre_BoomerAMGSetILUTriSolve_dbl ( void *data, HYPRE_Int ilu_tri_solve );
HYPRE_Int hypre_BoomerAMGSetILUTriSolve_long_dbl ( void *data, HYPRE_Int ilu_tri_solve );
HYPRE_Int hypre_BoomerAMGSetILUType_flt ( void *data, HYPRE_Int ilu_type );
HYPRE_Int hypre_BoomerAMGSetILUType_dbl ( void *data, HYPRE_Int ilu_type );
HYPRE_Int hypre_BoomerAMGSetILUType_long_dbl ( void *data, HYPRE_Int ilu_type );
HYPRE_Int hypre_BoomerAMGSetILUUpperJacobiIters_flt ( void *data, HYPRE_Int ilu_upper_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetILUUpperJacobiIters_dbl ( void *data, HYPRE_Int ilu_upper_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetILUUpperJacobiIters_long_dbl ( void *data, HYPRE_Int ilu_upper_jacobi_iters );
HYPRE_Int hypre_BoomerAMGSetInterpRefine_flt  ( void *data, HYPRE_Int num_refine );
HYPRE_Int hypre_BoomerAMGSetInterpRefine_dbl  ( void *data, HYPRE_Int num_refine );
HYPRE_Int hypre_BoomerAMGSetInterpRefine_long_dbl  ( void *data, HYPRE_Int num_refine );
HYPRE_Int hypre_BoomerAMGSetInterpType_flt  ( void *data, HYPRE_Int interp_type );
HYPRE_Int hypre_BoomerAMGSetInterpType_dbl  ( void *data, HYPRE_Int interp_type );
HYPRE_Int hypre_BoomerAMGSetInterpType_long_dbl  ( void *data, HYPRE_Int interp_type );
HYPRE_Int hypre_BoomerAMGSetInterpVecAbsQTrunc_flt  ( void *data, hypre_float q_trunc );
HYPRE_Int hypre_BoomerAMGSetInterpVecAbsQTrunc_dbl  ( void *data, hypre_double q_trunc );
HYPRE_Int hypre_BoomerAMGSetInterpVecAbsQTrunc_long_dbl  ( void *data, hypre_long_double q_trunc );
HYPRE_Int hypre_BoomerAMGSetInterpVecFirstLevel_flt  ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetInterpVecFirstLevel_dbl  ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetInterpVecFirstLevel_long_dbl  ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetInterpVecQMax_flt  ( void *data, HYPRE_Int q_max );
HYPRE_Int hypre_BoomerAMGSetInterpVecQMax_dbl  ( void *data, HYPRE_Int q_max );
HYPRE_Int hypre_BoomerAMGSetInterpVecQMax_long_dbl  ( void *data, HYPRE_Int q_max );
HYPRE_Int hypre_BoomerAMGSetInterpVectors_flt  ( void *solver, HYPRE_Int num_vectors,
                                            hypre_ParVector **interp_vectors );
HYPRE_Int hypre_BoomerAMGSetInterpVectors_dbl  ( void *solver, HYPRE_Int num_vectors,
                                            hypre_ParVector **interp_vectors );
HYPRE_Int hypre_BoomerAMGSetInterpVectors_long_dbl  ( void *solver, HYPRE_Int num_vectors,
                                            hypre_ParVector **interp_vectors );
HYPRE_Int hypre_BoomerAMGSetInterpVecVariant_flt  ( void *solver, HYPRE_Int var );
HYPRE_Int hypre_BoomerAMGSetInterpVecVariant_dbl  ( void *solver, HYPRE_Int var );
HYPRE_Int hypre_BoomerAMGSetInterpVecVariant_long_dbl  ( void *solver, HYPRE_Int var );
HYPRE_Int hypre_BoomerAMGSetIsTriangular_flt  ( void *data, HYPRE_Int is_triangular );
HYPRE_Int hypre_BoomerAMGSetIsTriangular_dbl  ( void *data, HYPRE_Int is_triangular );
HYPRE_Int hypre_BoomerAMGSetIsTriangular_long_dbl  ( void *data, HYPRE_Int is_triangular );
HYPRE_Int hypre_BoomerAMGSetISType_flt  ( void *data, HYPRE_Int IS_type );
HYPRE_Int hypre_BoomerAMGSetISType_dbl  ( void *data, HYPRE_Int IS_type );
HYPRE_Int hypre_BoomerAMGSetISType_long_dbl  ( void *data, HYPRE_Int IS_type );
HYPRE_Int hypre_BoomerAMGSetJacobiTruncThreshold_flt  ( void *data, hypre_float jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGSetJacobiTruncThreshold_dbl  ( void *data, hypre_double jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGSetJacobiTruncThreshold_long_dbl  ( void *data, hypre_long_double jacobi_trunc_threshold );
HYPRE_Int hypre_BoomerAMGSetKeepSameSign_flt  ( void *data, HYPRE_Int keep_same_sign );
HYPRE_Int hypre_BoomerAMGSetKeepSameSign_dbl  ( void *data, HYPRE_Int keep_same_sign );
HYPRE_Int hypre_BoomerAMGSetKeepSameSign_long_dbl  ( void *data, HYPRE_Int keep_same_sign );
HYPRE_Int hypre_BoomerAMGSetKeepTranspose_flt  ( void *data, HYPRE_Int keepTranspose );
HYPRE_Int hypre_BoomerAMGSetKeepTranspose_dbl  ( void *data, HYPRE_Int keepTranspose );
HYPRE_Int hypre_BoomerAMGSetKeepTranspose_long_dbl  ( void *data, HYPRE_Int keepTranspose );
HYPRE_Int hypre_BoomerAMGSetLevel_flt  ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevel_dbl  ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevel_long_dbl  ( void *data, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelNonGalerkinTol_flt  ( void *data, hypre_float nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelNonGalerkinTol_dbl  ( void *data, hypre_double nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelNonGalerkinTol_long_dbl  ( void *data, hypre_long_double nongalerkin_tol,
                                                  HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelOuterWt_flt  ( void *data, hypre_float omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelOuterWt_dbl  ( void *data, hypre_double omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelOuterWt_long_dbl  ( void *data, hypre_long_double omega, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelRelaxWt_flt  ( void *data, hypre_float relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelRelaxWt_dbl  ( void *data, hypre_double relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLevelRelaxWt_long_dbl  ( void *data, hypre_long_double relax_weight, HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSetLogging_flt  ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_BoomerAMGSetLogging_dbl  ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_BoomerAMGSetLogging_long_dbl  ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_BoomerAMGSetMaxCoarseSize_flt  ( void *data, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMaxCoarseSize_dbl  ( void *data, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMaxCoarseSize_long_dbl  ( void *data, HYPRE_Int max_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMaxIter_flt  ( void *data, HYPRE_Int max_iter );
HYPRE_Int hypre_BoomerAMGSetMaxIter_dbl  ( void *data, HYPRE_Int max_iter );
HYPRE_Int hypre_BoomerAMGSetMaxIter_long_dbl  ( void *data, HYPRE_Int max_iter );
HYPRE_Int hypre_BoomerAMGSetMaxLevels_flt  ( void *data, HYPRE_Int max_levels );
HYPRE_Int hypre_BoomerAMGSetMaxLevels_dbl  ( void *data, HYPRE_Int max_levels );
HYPRE_Int hypre_BoomerAMGSetMaxLevels_long_dbl  ( void *data, HYPRE_Int max_levels );
HYPRE_Int hypre_BoomerAMGSetMaxNzPerRow_flt  ( void *data, HYPRE_Int max_nz_per_row );
HYPRE_Int hypre_BoomerAMGSetMaxNzPerRow_dbl  ( void *data, HYPRE_Int max_nz_per_row );
HYPRE_Int hypre_BoomerAMGSetMaxNzPerRow_long_dbl  ( void *data, HYPRE_Int max_nz_per_row );
HYPRE_Int hypre_BoomerAMGSetMaxRowSum_flt  ( void *data, hypre_float max_row_sum );
HYPRE_Int hypre_BoomerAMGSetMaxRowSum_dbl  ( void *data, hypre_double max_row_sum );
HYPRE_Int hypre_BoomerAMGSetMaxRowSum_long_dbl  ( void *data, hypre_long_double max_row_sum );
HYPRE_Int hypre_BoomerAMGSetMeasureType_flt  ( void *data, HYPRE_Int measure_type );
HYPRE_Int hypre_BoomerAMGSetMeasureType_dbl  ( void *data, HYPRE_Int measure_type );
HYPRE_Int hypre_BoomerAMGSetMeasureType_long_dbl  ( void *data, HYPRE_Int measure_type );
HYPRE_Int hypre_BoomerAMGSetMinCoarseSize_flt  ( void *data, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMinCoarseSize_dbl  ( void *data, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMinCoarseSize_long_dbl  ( void *data, HYPRE_Int min_coarse_size );
HYPRE_Int hypre_BoomerAMGSetMinIter_flt  ( void *data, HYPRE_Int min_iter );
HYPRE_Int hypre_BoomerAMGSetMinIter_dbl  ( void *data, HYPRE_Int min_iter );
HYPRE_Int hypre_BoomerAMGSetMinIter_long_dbl  ( void *data, HYPRE_Int min_iter );
HYPRE_Int hypre_BoomerAMGSetModuleRAP2_flt  ( void *data, HYPRE_Int mod_rap2 );
HYPRE_Int hypre_BoomerAMGSetModuleRAP2_dbl  ( void *data, HYPRE_Int mod_rap2 );
HYPRE_Int hypre_BoomerAMGSetModuleRAP2_long_dbl  ( void *data, HYPRE_Int mod_rap2 );
HYPRE_Int hypre_BoomerAMGSetMultAdditive_flt  ( void *data, HYPRE_Int mult_additive );
HYPRE_Int hypre_BoomerAMGSetMultAdditive_dbl  ( void *data, HYPRE_Int mult_additive );
HYPRE_Int hypre_BoomerAMGSetMultAdditive_long_dbl  ( void *data, HYPRE_Int mult_additive );
HYPRE_Int hypre_BoomerAMGSetMultAddPMaxElmts_flt  ( void *data, HYPRE_Int add_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetMultAddPMaxElmts_dbl  ( void *data, HYPRE_Int add_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetMultAddPMaxElmts_long_dbl  ( void *data, HYPRE_Int add_P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetMultAddTruncFactor_flt  ( void *data, hypre_float add_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetMultAddTruncFactor_dbl  ( void *data, hypre_double add_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetMultAddTruncFactor_long_dbl  ( void *data, hypre_long_double add_trunc_factor );
HYPRE_Int hypre_BoomerAMGSetNodal_flt  ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodal_dbl  ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodal_long_dbl  ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodalDiag_flt  ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodalDiag_dbl  ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodalDiag_long_dbl  ( void *data, HYPRE_Int nodal );
HYPRE_Int hypre_BoomerAMGSetNodalLevels_flt  ( void *data, HYPRE_Int nodal_levels );
HYPRE_Int hypre_BoomerAMGSetNodalLevels_dbl  ( void *data, HYPRE_Int nodal_levels );
HYPRE_Int hypre_BoomerAMGSetNodalLevels_long_dbl  ( void *data, HYPRE_Int nodal_levels );
HYPRE_Int hypre_BoomerAMGSetNonGalerkinTol_flt  ( void *data, hypre_float nongalerkin_tol );
HYPRE_Int hypre_BoomerAMGSetNonGalerkinTol_dbl  ( void *data, hypre_double nongalerkin_tol );
HYPRE_Int hypre_BoomerAMGSetNonGalerkinTol_long_dbl  ( void *data, hypre_long_double nongalerkin_tol );
HYPRE_Int hypre_BoomerAMGSetNonGalerkTol_flt  ( void *data, HYPRE_Int nongalerk_num_tol,
                                           hypre_float *nongalerk_tol );
HYPRE_Int hypre_BoomerAMGSetNonGalerkTol_dbl  ( void *data, HYPRE_Int nongalerk_num_tol,
                                           hypre_double *nongalerk_tol );
HYPRE_Int hypre_BoomerAMGSetNonGalerkTol_long_dbl  ( void *data, HYPRE_Int nongalerk_num_tol,
                                           hypre_long_double *nongalerk_tol );
HYPRE_Int hypre_BoomerAMGSetNumCRRelaxSteps_flt  ( void *data, HYPRE_Int num_CR_relax_steps );
HYPRE_Int hypre_BoomerAMGSetNumCRRelaxSteps_dbl  ( void *data, HYPRE_Int num_CR_relax_steps );
HYPRE_Int hypre_BoomerAMGSetNumCRRelaxSteps_long_dbl  ( void *data, HYPRE_Int num_CR_relax_steps );
HYPRE_Int hypre_BoomerAMGSetNumFunctions_flt  ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMGSetNumFunctions_dbl  ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMGSetNumFunctions_long_dbl  ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMGSetNumGridSweeps_flt  ( void *data, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGSetNumGridSweeps_dbl  ( void *data, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGSetNumGridSweeps_long_dbl  ( void *data, HYPRE_Int *num_grid_sweeps );
HYPRE_Int hypre_BoomerAMGSetNumIterations_flt  ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_BoomerAMGSetNumIterations_dbl  ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_BoomerAMGSetNumIterations_long_dbl  ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_BoomerAMGSetNumPaths_flt  ( void *data, HYPRE_Int num_paths );
HYPRE_Int hypre_BoomerAMGSetNumPaths_dbl  ( void *data, HYPRE_Int num_paths );
HYPRE_Int hypre_BoomerAMGSetNumPaths_long_dbl  ( void *data, HYPRE_Int num_paths );
HYPRE_Int hypre_BoomerAMGSetNumPoints_flt  ( void *data, HYPRE_Int num_points );
HYPRE_Int hypre_BoomerAMGSetNumPoints_dbl  ( void *data, HYPRE_Int num_points );
HYPRE_Int hypre_BoomerAMGSetNumPoints_long_dbl  ( void *data, HYPRE_Int num_points );
HYPRE_Int hypre_BoomerAMGSetNumSamples_flt  ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetNumSamples_dbl  ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetNumSamples_long_dbl  ( void *data, HYPRE_Int par );
HYPRE_Int hypre_BoomerAMGSetNumSweeps_flt  ( void *data, HYPRE_Int num_sweeps );
HYPRE_Int hypre_BoomerAMGSetNumSweeps_dbl  ( void *data, HYPRE_Int num_sweeps );
HYPRE_Int hypre_BoomerAMGSetNumSweeps_long_dbl  ( void *data, HYPRE_Int num_sweeps );
HYPRE_Int hypre_BoomerAMGSetOmega_flt  ( void *data, hypre_float *omega );
HYPRE_Int hypre_BoomerAMGSetOmega_dbl  ( void *data, hypre_double *omega );
HYPRE_Int hypre_BoomerAMGSetOmega_long_dbl  ( void *data, hypre_long_double *omega );
HYPRE_Int hypre_BoomerAMGSetOuterWt_flt  ( void *data, hypre_float omega );
HYPRE_Int hypre_BoomerAMGSetOuterWt_dbl  ( void *data, hypre_double omega );
HYPRE_Int hypre_BoomerAMGSetOuterWt_long_dbl  ( void *data, hypre_long_double omega );
HYPRE_Int hypre_BoomerAMGSetOverlap_flt  ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_BoomerAMGSetOverlap_dbl  ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_BoomerAMGSetOverlap_long_dbl  ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_BoomerAMGSetPlotFileName_flt  ( void *data, const char *plot_file_name );
HYPRE_Int hypre_BoomerAMGSetPlotFileName_dbl  ( void *data, const char *plot_file_name );
HYPRE_Int hypre_BoomerAMGSetPlotFileName_long_dbl  ( void *data, const char *plot_file_name );
HYPRE_Int hypre_BoomerAMGSetPlotGrids_flt  ( void *data, HYPRE_Int plotgrids );
HYPRE_Int hypre_BoomerAMGSetPlotGrids_dbl  ( void *data, HYPRE_Int plotgrids );
HYPRE_Int hypre_BoomerAMGSetPlotGrids_long_dbl  ( void *data, HYPRE_Int plotgrids );
HYPRE_Int hypre_BoomerAMGSetPMaxElmts_flt  ( void *data, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetPMaxElmts_dbl  ( void *data, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetPMaxElmts_long_dbl  ( void *data, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_BoomerAMGSetPointDofMap_flt  ( void *data, HYPRE_Int *point_dof_map );
HYPRE_Int hypre_BoomerAMGSetPointDofMap_dbl  ( void *data, HYPRE_Int *point_dof_map );
HYPRE_Int hypre_BoomerAMGSetPointDofMap_long_dbl  ( void *data, HYPRE_Int *point_dof_map );
HYPRE_Int hypre_BoomerAMGSetPostInterpType_flt  ( void *data, HYPRE_Int post_interp_type );
HYPRE_Int hypre_BoomerAMGSetPostInterpType_dbl  ( void *data, HYPRE_Int post_interp_type );
HYPRE_Int hypre_BoomerAMGSetPostInterpType_long_dbl  ( void *data, HYPRE_Int post_interp_type );
HYPRE_Int hypre_BoomerAMGSetPrintFileName_flt  ( void *data, const char *print_file_name );
HYPRE_Int hypre_BoomerAMGSetPrintFileName_dbl  ( void *data, const char *print_file_name );
HYPRE_Int hypre_BoomerAMGSetPrintFileName_long_dbl  ( void *data, const char *print_file_name );
HYPRE_Int hypre_BoomerAMGSetPrintLevel_flt  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_BoomerAMGSetPrintLevel_dbl  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_BoomerAMGSetPrintLevel_long_dbl  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_BoomerAMGSetRAP2_flt  ( void *data, HYPRE_Int rap2 );
HYPRE_Int hypre_BoomerAMGSetRAP2_dbl  ( void *data, HYPRE_Int rap2 );
HYPRE_Int hypre_BoomerAMGSetRAP2_long_dbl  ( void *data, HYPRE_Int rap2 );
HYPRE_Int hypre_BoomerAMGSetRedundant_flt  ( void *data, HYPRE_Int redundant );
HYPRE_Int hypre_BoomerAMGSetRedundant_dbl  ( void *data, HYPRE_Int redundant );
HYPRE_Int hypre_BoomerAMGSetRedundant_long_dbl  ( void *data, HYPRE_Int redundant );
HYPRE_Int hypre_BoomerAMGSetRelaxOrder_flt  ( void *data, HYPRE_Int relax_order );
HYPRE_Int hypre_BoomerAMGSetRelaxOrder_dbl  ( void *data, HYPRE_Int relax_order );
HYPRE_Int hypre_BoomerAMGSetRelaxOrder_long_dbl  ( void *data, HYPRE_Int relax_order );
HYPRE_Int hypre_BoomerAMGSetRelaxType_flt  ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BoomerAMGSetRelaxType_dbl  ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BoomerAMGSetRelaxType_long_dbl  ( void *data, HYPRE_Int relax_type );
HYPRE_Int hypre_BoomerAMGSetRelaxWeight_flt  ( void *data, hypre_float *relax_weight );
HYPRE_Int hypre_BoomerAMGSetRelaxWeight_dbl  ( void *data, hypre_double *relax_weight );
HYPRE_Int hypre_BoomerAMGSetRelaxWeight_long_dbl  ( void *data, hypre_long_double *relax_weight );
HYPRE_Int hypre_BoomerAMGSetRelaxWt_flt  ( void *data, hypre_float relax_weight );
HYPRE_Int hypre_BoomerAMGSetRelaxWt_dbl  ( void *data, hypre_double relax_weight );
HYPRE_Int hypre_BoomerAMGSetRelaxWt_long_dbl  ( void *data, hypre_long_double relax_weight );
HYPRE_Int hypre_BoomerAMGSetRestriction_flt  ( void *data, HYPRE_Int restr_par );
HYPRE_Int hypre_BoomerAMGSetRestriction_dbl  ( void *data, HYPRE_Int restr_par );
HYPRE_Int hypre_BoomerAMGSetRestriction_long_dbl  ( void *data, HYPRE_Int restr_par );
HYPRE_Int hypre_BoomerAMGSetSabs_flt  ( void *data, HYPRE_Int Sabs );
HYPRE_Int hypre_BoomerAMGSetSabs_dbl  ( void *data, HYPRE_Int Sabs );
HYPRE_Int hypre_BoomerAMGSetSabs_long_dbl  ( void *data, HYPRE_Int Sabs );
HYPRE_Int hypre_BoomerAMGSetSchwarzRlxWeight_flt  ( void *data, hypre_float schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGSetSchwarzRlxWeight_dbl  ( void *data, hypre_double schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGSetSchwarzRlxWeight_long_dbl  ( void *data, hypre_long_double schwarz_rlx_weight );
HYPRE_Int hypre_BoomerAMGSetSchwarzUseNonSymm_flt  ( void *data, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_BoomerAMGSetSchwarzUseNonSymm_dbl  ( void *data, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_BoomerAMGSetSchwarzUseNonSymm_long_dbl  ( void *data, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_BoomerAMGSetSepWeight_flt  ( void *data, HYPRE_Int sep_weight );
HYPRE_Int hypre_BoomerAMGSetSepWeight_dbl  ( void *data, HYPRE_Int sep_weight );
HYPRE_Int hypre_BoomerAMGSetSepWeight_long_dbl  ( void *data, HYPRE_Int sep_weight );
HYPRE_Int hypre_BoomerAMGSetSeqThreshold_flt  ( void *data, HYPRE_Int seq_threshold );
HYPRE_Int hypre_BoomerAMGSetSeqThreshold_dbl  ( void *data, HYPRE_Int seq_threshold );
HYPRE_Int hypre_BoomerAMGSetSeqThreshold_long_dbl  ( void *data, HYPRE_Int seq_threshold );
HYPRE_Int hypre_BoomerAMGSetSetupType_flt  ( void *data, HYPRE_Int setup_type );
HYPRE_Int hypre_BoomerAMGSetSetupType_dbl  ( void *data, HYPRE_Int setup_type );
HYPRE_Int hypre_BoomerAMGSetSetupType_long_dbl  ( void *data, HYPRE_Int setup_type );
HYPRE_Int hypre_BoomerAMGSetSimple_flt  ( void *data, HYPRE_Int simple );
HYPRE_Int hypre_BoomerAMGSetSimple_dbl  ( void *data, HYPRE_Int simple );
HYPRE_Int hypre_BoomerAMGSetSimple_long_dbl  ( void *data, HYPRE_Int simple );
HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors_flt  ( void *solver, HYPRE_Int smooth_interp_vectors );
HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors_dbl  ( void *solver, HYPRE_Int smooth_interp_vectors );
HYPRE_Int hypre_BoomerAMGSetSmoothInterpVectors_long_dbl  ( void *solver, HYPRE_Int smooth_interp_vectors );
HYPRE_Int hypre_BoomerAMGSetSmoothNumLevels_flt  ( void *data, HYPRE_Int smooth_num_levels );
HYPRE_Int hypre_BoomerAMGSetSmoothNumLevels_dbl  ( void *data, HYPRE_Int smooth_num_levels );
HYPRE_Int hypre_BoomerAMGSetSmoothNumLevels_long_dbl  ( void *data, HYPRE_Int smooth_num_levels );
HYPRE_Int hypre_BoomerAMGSetSmoothNumSweeps_flt  ( void *data, HYPRE_Int smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGSetSmoothNumSweeps_dbl  ( void *data, HYPRE_Int smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGSetSmoothNumSweeps_long_dbl  ( void *data, HYPRE_Int smooth_num_sweeps );
HYPRE_Int hypre_BoomerAMGSetSmoothType_flt  ( void *data, HYPRE_Int smooth_type );
HYPRE_Int hypre_BoomerAMGSetSmoothType_dbl  ( void *data, HYPRE_Int smooth_type );
HYPRE_Int hypre_BoomerAMGSetSmoothType_long_dbl  ( void *data, HYPRE_Int smooth_type );
HYPRE_Int hypre_BoomerAMGSetStrongThreshold_flt  ( void *data, hypre_float strong_threshold );
HYPRE_Int hypre_BoomerAMGSetStrongThreshold_dbl  ( void *data, hypre_double strong_threshold );
HYPRE_Int hypre_BoomerAMGSetStrongThreshold_long_dbl  ( void *data, hypre_long_double strong_threshold );
HYPRE_Int hypre_BoomerAMGSetStrongThresholdR_flt  ( void *data, hypre_float strong_threshold );
HYPRE_Int hypre_BoomerAMGSetStrongThresholdR_dbl  ( void *data, hypre_double strong_threshold );
HYPRE_Int hypre_BoomerAMGSetStrongThresholdR_long_dbl  ( void *data, hypre_long_double strong_threshold );
HYPRE_Int hypre_BoomerAMGSetSym_flt  ( void *data, HYPRE_Int sym );
HYPRE_Int hypre_BoomerAMGSetSym_dbl  ( void *data, HYPRE_Int sym );
HYPRE_Int hypre_BoomerAMGSetSym_long_dbl  ( void *data, HYPRE_Int sym );
HYPRE_Int hypre_BoomerAMGSetThreshold_flt  ( void *data, hypre_float thresh );
HYPRE_Int hypre_BoomerAMGSetThreshold_dbl  ( void *data, hypre_double thresh );
HYPRE_Int hypre_BoomerAMGSetThreshold_long_dbl  ( void *data, hypre_long_double thresh );
HYPRE_Int hypre_BoomerAMGSetTol_flt  ( void *data, hypre_float tol );
HYPRE_Int hypre_BoomerAMGSetTol_dbl  ( void *data, hypre_double tol );
HYPRE_Int hypre_BoomerAMGSetTol_long_dbl  ( void *data, hypre_long_double tol );
HYPRE_Int hypre_BoomerAMGSetTruncFactor_flt  ( void *data, hypre_float trunc_factor );
HYPRE_Int hypre_BoomerAMGSetTruncFactor_dbl  ( void *data, hypre_double trunc_factor );
HYPRE_Int hypre_BoomerAMGSetTruncFactor_long_dbl  ( void *data, hypre_long_double trunc_factor );
HYPRE_Int hypre_BoomerAMGSetVariant_flt  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetVariant_dbl  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetVariant_long_dbl  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_BoomerAMGSetup_flt  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSetup_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSetup_long_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSolve_flt  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSolve_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSolve_long_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGCycleT_flt  ( void *amg_vdata, hypre_ParVector **F_array,
                                  hypre_ParVector **U_array );
HYPRE_Int hypre_BoomerAMGCycleT_dbl  ( void *amg_vdata, hypre_ParVector **F_array,
                                  hypre_ParVector **U_array );
HYPRE_Int hypre_BoomerAMGCycleT_long_dbl  ( void *amg_vdata, hypre_ParVector **F_array,
                                  hypre_ParVector **U_array );
HYPRE_Int hypre_BoomerAMGRelaxT_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                  HYPRE_Int relax_type, HYPRE_Int relax_points, hypre_float relax_weight, hypre_ParVector *u,
                                  hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelaxT_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                  HYPRE_Int relax_type, HYPRE_Int relax_points, hypre_double relax_weight, hypre_ParVector *u,
                                  hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelaxT_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                  HYPRE_Int relax_type, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_ParVector *u,
                                  hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGSolveT_flt  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSolveT_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSolveT_long_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u );
HYPRE_Int hypre_AmgCGCBoundaryFix_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                    HYPRE_Int *CF_marker_offd );
HYPRE_Int hypre_AmgCGCBoundaryFix_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                    HYPRE_Int *CF_marker_offd );
HYPRE_Int hypre_AmgCGCBoundaryFix_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                    HYPRE_Int *CF_marker_offd );
HYPRE_Int hypre_AmgCGCChoose_flt  ( hypre_CSRMatrix *G, HYPRE_Int *vertexrange, HYPRE_Int mpisize,
                               HYPRE_Int **coarse );
HYPRE_Int hypre_AmgCGCChoose_dbl  ( hypre_CSRMatrix *G, HYPRE_Int *vertexrange, HYPRE_Int mpisize,
                               HYPRE_Int **coarse );
HYPRE_Int hypre_AmgCGCChoose_long_dbl  ( hypre_CSRMatrix *G, HYPRE_Int *vertexrange, HYPRE_Int mpisize,
                               HYPRE_Int **coarse );
HYPRE_Int hypre_AmgCGCGraphAssemble_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int *vertexrange,
                                      HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_IJMatrix *ijG );
HYPRE_Int hypre_AmgCGCGraphAssemble_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int *vertexrange,
                                      HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_IJMatrix *ijG );
HYPRE_Int hypre_AmgCGCGraphAssemble_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int *vertexrange,
                                      HYPRE_Int *CF_marker, HYPRE_Int *CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_IJMatrix *ijG );
HYPRE_Int hypre_AmgCGCPrepare_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int nlocal, HYPRE_Int *CF_marker,
                                HYPRE_Int **CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_Int **vrange );
HYPRE_Int hypre_AmgCGCPrepare_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int nlocal, HYPRE_Int *CF_marker,
                                HYPRE_Int **CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_Int **vrange );
HYPRE_Int hypre_AmgCGCPrepare_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int nlocal, HYPRE_Int *CF_marker,
                                HYPRE_Int **CF_marker_offd, HYPRE_Int coarsen_type, HYPRE_Int **vrange );
HYPRE_Int hypre_BoomerAMGCoarsenCGC_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int numberofgrids,
                                      HYPRE_Int coarsen_type, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGCoarsenCGC_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int numberofgrids,
                                      HYPRE_Int coarsen_type, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGCoarsenCGC_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int numberofgrids,
                                      HYPRE_Int coarsen_type, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGCoarsenCGCb_flt  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cgc_its, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenCGCb_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cgc_its, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenCGCb_long_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cgc_its, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_Bisection_flt  ( HYPRE_Int n, hypre_float *diag, hypre_float *offd, hypre_float y,
                            hypre_float z, hypre_float tol, HYPRE_Int k, hypre_float *ev_ptr );
HYPRE_Int hypre_Bisection_dbl  ( HYPRE_Int n, hypre_double *diag, hypre_double *offd, hypre_double y,
                            hypre_double z, hypre_double tol, HYPRE_Int k, hypre_double *ev_ptr );
HYPRE_Int hypre_Bisection_long_dbl  ( HYPRE_Int n, hypre_long_double *diag, hypre_long_double *offd, hypre_long_double y,
                            hypre_long_double z, hypre_long_double tol, HYPRE_Int k, hypre_long_double *ev_ptr );
HYPRE_Int hypre_BoomerAMGCGRelaxWt_flt  ( void *amg_vdata, HYPRE_Int level, HYPRE_Int num_cg_sweeps,
                                     hypre_float *rlx_wt_ptr );
HYPRE_Int hypre_BoomerAMGCGRelaxWt_dbl  ( void *amg_vdata, HYPRE_Int level, HYPRE_Int num_cg_sweeps,
                                     hypre_double *rlx_wt_ptr );
HYPRE_Int hypre_BoomerAMGCGRelaxWt_long_dbl  ( void *amg_vdata, HYPRE_Int level, HYPRE_Int num_cg_sweeps,
                                     hypre_long_double *rlx_wt_ptr );
HYPRE_Int hypre_ParCSRRelax_Cheby_Setup_flt  ( hypre_ParCSRMatrix *A, hypre_float max_eig,
                                          hypre_float min_eig, hypre_float fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          hypre_float **coefs_ptr, hypre_float **ds_ptr );
HYPRE_Int hypre_ParCSRRelax_Cheby_Setup_dbl  ( hypre_ParCSRMatrix *A, hypre_double max_eig,
                                          hypre_double min_eig, hypre_double fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          hypre_double **coefs_ptr, hypre_double **ds_ptr );
HYPRE_Int hypre_ParCSRRelax_Cheby_Setup_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double max_eig,
                                          hypre_long_double min_eig, hypre_long_double fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          hypre_long_double **coefs_ptr, hypre_long_double **ds_ptr );
HYPRE_Int hypre_ParCSRRelax_Cheby_Solve_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                          hypre_float *ds_data, hypre_float *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                          hypre_ParVector *tmp_vec);
HYPRE_Int hypre_ParCSRRelax_Cheby_Solve_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                          hypre_double *ds_data, hypre_double *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                          hypre_ParVector *tmp_vec);
HYPRE_Int hypre_ParCSRRelax_Cheby_Solve_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                          hypre_long_double *ds_data, hypre_long_double *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                          hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                          hypre_ParVector *tmp_vec);
HYPRE_Int hypre_ParCSRRelax_Cheby_SolveHost_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                              hypre_float *ds_data, hypre_float *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                              hypre_ParVector *tmp_vec);
HYPRE_Int hypre_ParCSRRelax_Cheby_SolveHost_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                              hypre_double *ds_data, hypre_double *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                              hypre_ParVector *tmp_vec);
HYPRE_Int hypre_ParCSRRelax_Cheby_SolveHost_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                              hypre_long_double *ds_data, hypre_long_double *coefs, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                              hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r, hypre_ParVector *orig_u_vec,
                                              hypre_ParVector *tmp_vec);
HYPRE_Int hypre_BoomerAMGCoarsen_flt  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int CF_init,
                                   HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsen_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int CF_init,
                                   HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsen_long_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int CF_init,
                                   HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenFalgout_flt  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                          HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                          hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenFalgout_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                          HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                          hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenFalgout_long_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                          HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                          hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenHMIS_flt  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenHMIS_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenHMIS_long_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMIS_flt  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMIS_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMIS_long_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMISHost_flt  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                           HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMISHost_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                           HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenPMISHost_long_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                           HYPRE_Int CF_init, HYPRE_Int debug_flag, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenRuge_flt  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenRuge_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarsenRuge_long_dbl  ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                                       HYPRE_Int measure_type, HYPRE_Int coarsen_type, HYPRE_Int cut_factor, HYPRE_Int debug_flag,
                                       hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCoarseParms_flt  ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                       HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                       hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGCoarseParms_dbl  ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                       HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                       hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGCoarseParms_long_dbl  ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                       HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                       hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGCoarseParmsHost_flt  ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                           HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                           hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGCoarseParmsHost_dbl  ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                           HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                           hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
HYPRE_Int hypre_BoomerAMGCoarseParmsHost_long_dbl  ( MPI_Comm comm, HYPRE_Int local_num_variables,
                                           HYPRE_Int num_functions, hypre_IntArray *dof_func, hypre_IntArray *CF_marker,
                                           hypre_IntArray **coarse_dof_func_ptr, HYPRE_BigInt *coarse_pnts_global );
float *GenerateCoordinates_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                             HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r, HYPRE_Int coorddim );
float *GenerateCoordinates_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                             HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r, HYPRE_Int coorddim );
float *GenerateCoordinates_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                             HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r, HYPRE_Int coorddim );
HYPRE_Int hypre_BoomerAMGCoarsenCR_flt  ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                     HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                     HYPRE_Int num_functions, HYPRE_Int rlx_type, hypre_float relax_weight, hypre_float omega,
                                     hypre_float theta, HYPRE_Solver smoother, hypre_ParCSRMatrix *AN, HYPRE_Int useCG,
                                     hypre_ParCSRMatrix *S );
HYPRE_Int hypre_BoomerAMGCoarsenCR_dbl  ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                     HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                     HYPRE_Int num_functions, HYPRE_Int rlx_type, hypre_double relax_weight, hypre_double omega,
                                     hypre_double theta, HYPRE_Solver smoother, hypre_ParCSRMatrix *AN, HYPRE_Int useCG,
                                     hypre_ParCSRMatrix *S );
HYPRE_Int hypre_BoomerAMGCoarsenCR_long_dbl  ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                     HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                     HYPRE_Int num_functions, HYPRE_Int rlx_type, hypre_long_double relax_weight, hypre_long_double omega,
                                     hypre_long_double theta, HYPRE_Solver smoother, hypre_ParCSRMatrix *AN, HYPRE_Int useCG,
                                     hypre_ParCSRMatrix *S );
HYPRE_Int hypre_BoomerAMGCoarsenCR1_flt  ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                      HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                      HYPRE_Int CRaddCpoints );
HYPRE_Int hypre_BoomerAMGCoarsenCR1_dbl  ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                      HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                      HYPRE_Int CRaddCpoints );
HYPRE_Int hypre_BoomerAMGCoarsenCR1_long_dbl  ( hypre_ParCSRMatrix *A, hypre_IntArray **CF_marker_ptr,
                                      HYPRE_BigInt *coarse_size_ptr, HYPRE_Int num_CR_relax_steps, HYPRE_Int IS_type,
                                      HYPRE_Int CRaddCpoints );
HYPRE_Int hypre_BoomerAMGIndepHMIS_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                     HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMIS_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                     HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMIS_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                     HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMISa_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMISa_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepHMISa_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMIS_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init, HYPRE_Int debug_flag,
                                     HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMIS_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init, HYPRE_Int debug_flag,
                                     HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMIS_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init, HYPRE_Int debug_flag,
                                     HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMISa_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMISa_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepPMISa_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int CF_init,
                                      HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRS_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                   HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRS_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                   HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRS_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                   HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRSa_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                    HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRSa_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                    HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_BoomerAMGIndepRSa_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int measure_type,
                                    HYPRE_Int debug_flag, HYPRE_Int *CF_marker );
HYPRE_Int hypre_cr_flt  ( HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_float *A_data, HYPRE_Int n, HYPRE_Int *cf,
                     HYPRE_Int rlx, hypre_float omega, hypre_float tg, HYPRE_Int mu );
HYPRE_Int hypre_cr_dbl  ( HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_double *A_data, HYPRE_Int n, HYPRE_Int *cf,
                     HYPRE_Int rlx, hypre_double omega, hypre_double tg, HYPRE_Int mu );
HYPRE_Int hypre_cr_long_dbl  ( HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_long_double *A_data, HYPRE_Int n, HYPRE_Int *cf,
                     HYPRE_Int rlx, hypre_long_double omega, hypre_long_double tg, HYPRE_Int mu );
HYPRE_Int hypre_formu_flt  ( HYPRE_Int *cf, HYPRE_Int n, hypre_float *e1, HYPRE_Int *A_i,
                        hypre_float rho );
HYPRE_Int hypre_formu_dbl  ( HYPRE_Int *cf, HYPRE_Int n, hypre_double *e1, HYPRE_Int *A_i,
                        hypre_double rho );
HYPRE_Int hypre_formu_long_dbl  ( HYPRE_Int *cf, HYPRE_Int n, hypre_long_double *e1, HYPRE_Int *A_i,
                        hypre_long_double rho );
HYPRE_Int hypre_fptgscr_flt  ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_float *A_data,
                          HYPRE_Int n, hypre_float *e0, hypre_float *e1 );
HYPRE_Int hypre_fptgscr_dbl  ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_double *A_data,
                          HYPRE_Int n, hypre_double *e0, hypre_double *e1 );
HYPRE_Int hypre_fptgscr_long_dbl  ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_long_double *A_data,
                          HYPRE_Int n, hypre_long_double *e0, hypre_long_double *e1 );
HYPRE_Int hypre_fptjaccr_flt  ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_float *A_data,
                           HYPRE_Int n, hypre_float *e0, hypre_float omega, hypre_float *e1 );
HYPRE_Int hypre_fptjaccr_dbl  ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_double *A_data,
                           HYPRE_Int n, hypre_double *e0, hypre_double omega, hypre_double *e1 );
HYPRE_Int hypre_fptjaccr_long_dbl  ( HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, hypre_long_double *A_data,
                           HYPRE_Int n, hypre_long_double *e0, hypre_long_double omega, hypre_long_double *e1 );
HYPRE_Int hypre_GraphAdd_flt  ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index,
                           HYPRE_Int istack );
HYPRE_Int hypre_GraphAdd_dbl  ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index,
                           HYPRE_Int istack );
HYPRE_Int hypre_GraphAdd_long_dbl  ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index,
                           HYPRE_Int istack );
HYPRE_Int hypre_GraphRemove_flt  ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index );
HYPRE_Int hypre_GraphRemove_dbl  ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index );
HYPRE_Int hypre_GraphRemove_long_dbl  ( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index );
HYPRE_Int hypre_IndepSetGreedy_flt  ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_IndepSetGreedy_dbl  ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_IndepSetGreedy_long_dbl  ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_IndepSetGreedyS_flt  ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_IndepSetGreedyS_dbl  ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_IndepSetGreedyS_long_dbl  ( HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf );
HYPRE_Int hypre_BoomerAMGCycle_flt  ( void *amg_vdata, hypre_ParVector **F_array,
                                 hypre_ParVector **U_array );
HYPRE_Int hypre_BoomerAMGCycle_dbl  ( void *amg_vdata, hypre_ParVector **F_array,
                                 hypre_ParVector **U_array );
HYPRE_Int hypre_BoomerAMGCycle_long_dbl  ( void *amg_vdata, hypre_ParVector **F_array,
                                 hypre_ParVector **U_array );
HYPRE_ParCSRMatrix GenerateDifConv_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                     HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                     hypre_float *value );
HYPRE_ParCSRMatrix GenerateDifConv_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                     HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                     hypre_double *value );
HYPRE_ParCSRMatrix GenerateDifConv_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                     HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                     hypre_long_double *value );
void* hypre_FSAICreate_flt ( void );
void* hypre_FSAICreate_dbl ( void );
void* hypre_FSAICreate_long_dbl ( void );
HYPRE_Int hypre_FSAIDestroy_flt  ( void *data );
HYPRE_Int hypre_FSAIDestroy_dbl  ( void *data );
HYPRE_Int hypre_FSAIDestroy_long_dbl  ( void *data );
HYPRE_Int hypre_FSAIGetAlgoType_flt  ( void *data, HYPRE_Int *algo_type );
HYPRE_Int hypre_FSAIGetAlgoType_dbl  ( void *data, HYPRE_Int *algo_type );
HYPRE_Int hypre_FSAIGetAlgoType_long_dbl  ( void *data, HYPRE_Int *algo_type );
HYPRE_Int hypre_FSAIGetEigMaxIters_flt  ( void *data, HYPRE_Int *eig_max_iters );
HYPRE_Int hypre_FSAIGetEigMaxIters_dbl  ( void *data, HYPRE_Int *eig_max_iters );
HYPRE_Int hypre_FSAIGetEigMaxIters_long_dbl  ( void *data, HYPRE_Int *eig_max_iters );
HYPRE_Int hypre_FSAIGetKapTolerance_flt  ( void *data, hypre_float *kap_tolerance );
HYPRE_Int hypre_FSAIGetKapTolerance_dbl  ( void *data, hypre_double *kap_tolerance );
HYPRE_Int hypre_FSAIGetKapTolerance_long_dbl  ( void *data, hypre_long_double *kap_tolerance );
HYPRE_Int hypre_FSAIGetLocalSolveType_flt  ( void *data, HYPRE_Int *local_solve_type );
HYPRE_Int hypre_FSAIGetLocalSolveType_dbl  ( void *data, HYPRE_Int *local_solve_type );
HYPRE_Int hypre_FSAIGetLocalSolveType_long_dbl  ( void *data, HYPRE_Int *local_solve_type );
HYPRE_Int hypre_FSAIGetLogging_flt  ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_FSAIGetLogging_dbl  ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_FSAIGetLogging_long_dbl  ( void *data, HYPRE_Int *logging );
HYPRE_Int hypre_FSAIGetMaxIterations_flt  ( void *data, HYPRE_Int *max_iterations );
HYPRE_Int hypre_FSAIGetMaxIterations_dbl  ( void *data, HYPRE_Int *max_iterations );
HYPRE_Int hypre_FSAIGetMaxIterations_long_dbl  ( void *data, HYPRE_Int *max_iterations );
HYPRE_Int hypre_FSAIGetMaxNnzRow_flt  ( void *data, HYPRE_Int *max_nnz_row );
HYPRE_Int hypre_FSAIGetMaxNnzRow_dbl  ( void *data, HYPRE_Int *max_nnz_row );
HYPRE_Int hypre_FSAIGetMaxNnzRow_long_dbl  ( void *data, HYPRE_Int *max_nnz_row );
HYPRE_Int hypre_FSAIGetMaxSteps_flt  ( void *data, HYPRE_Int *max_steps );
HYPRE_Int hypre_FSAIGetMaxSteps_dbl  ( void *data, HYPRE_Int *max_steps );
HYPRE_Int hypre_FSAIGetMaxSteps_long_dbl  ( void *data, HYPRE_Int *max_steps );
HYPRE_Int hypre_FSAIGetMaxStepSize_flt  ( void *data, HYPRE_Int *max_step_size );
HYPRE_Int hypre_FSAIGetMaxStepSize_dbl  ( void *data, HYPRE_Int *max_step_size );
HYPRE_Int hypre_FSAIGetMaxStepSize_long_dbl  ( void *data, HYPRE_Int *max_step_size );
HYPRE_Int hypre_FSAIGetNumIterations_flt  ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FSAIGetNumIterations_dbl  ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FSAIGetNumIterations_long_dbl  ( void *data, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FSAIGetNumLevels_flt  ( void *data, HYPRE_Int *num_levels );
HYPRE_Int hypre_FSAIGetNumLevels_dbl  ( void *data, HYPRE_Int *num_levels );
HYPRE_Int hypre_FSAIGetNumLevels_long_dbl  ( void *data, HYPRE_Int *num_levels );
HYPRE_Int hypre_FSAIGetOmega_flt  ( void *data, hypre_float *omega );
HYPRE_Int hypre_FSAIGetOmega_dbl  ( void *data, hypre_double *omega );
HYPRE_Int hypre_FSAIGetOmega_long_dbl  ( void *data, hypre_long_double *omega );
HYPRE_Int hypre_FSAIGetPrintLevel_flt  ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_FSAIGetPrintLevel_dbl  ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_FSAIGetPrintLevel_long_dbl  ( void *data, HYPRE_Int *print_level );
HYPRE_Int hypre_FSAIGetThreshold_flt  ( void *data, hypre_float *threshold );
HYPRE_Int hypre_FSAIGetThreshold_dbl  ( void *data, hypre_double *threshold );
HYPRE_Int hypre_FSAIGetThreshold_long_dbl  ( void *data, hypre_long_double *threshold );
HYPRE_Int hypre_FSAIGetTolerance_flt  ( void *data, hypre_float *tolerance );
HYPRE_Int hypre_FSAIGetTolerance_dbl  ( void *data, hypre_double *tolerance );
HYPRE_Int hypre_FSAIGetTolerance_long_dbl  ( void *data, hypre_long_double *tolerance );
HYPRE_Int hypre_FSAIGetZeroGuess_flt  ( void *data, HYPRE_Int *zero_guess );
HYPRE_Int hypre_FSAIGetZeroGuess_dbl  ( void *data, HYPRE_Int *zero_guess );
HYPRE_Int hypre_FSAIGetZeroGuess_long_dbl  ( void *data, HYPRE_Int *zero_guess );
HYPRE_Int hypre_FSAISetAlgoType_flt  ( void *data, HYPRE_Int algo_type );
HYPRE_Int hypre_FSAISetAlgoType_dbl  ( void *data, HYPRE_Int algo_type );
HYPRE_Int hypre_FSAISetAlgoType_long_dbl  ( void *data, HYPRE_Int algo_type );
HYPRE_Int hypre_FSAISetEigMaxIters_flt  ( void *data, HYPRE_Int eig_max_iters );
HYPRE_Int hypre_FSAISetEigMaxIters_dbl  ( void *data, HYPRE_Int eig_max_iters );
HYPRE_Int hypre_FSAISetEigMaxIters_long_dbl  ( void *data, HYPRE_Int eig_max_iters );
HYPRE_Int hypre_FSAISetKapTolerance_flt  ( void *data, hypre_float kap_tolerance );
HYPRE_Int hypre_FSAISetKapTolerance_dbl  ( void *data, hypre_double kap_tolerance );
HYPRE_Int hypre_FSAISetKapTolerance_long_dbl  ( void *data, hypre_long_double kap_tolerance );
HYPRE_Int hypre_FSAISetLocalSolveType_flt  ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_FSAISetLocalSolveType_dbl  ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_FSAISetLocalSolveType_long_dbl  ( void *data, HYPRE_Int local_solve_type );
HYPRE_Int hypre_FSAISetLogging_flt  ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_FSAISetLogging_dbl  ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_FSAISetLogging_long_dbl  ( void *data, HYPRE_Int logging );
HYPRE_Int hypre_FSAISetMaxIterations_flt  ( void *data, HYPRE_Int max_iterations );
HYPRE_Int hypre_FSAISetMaxIterations_dbl  ( void *data, HYPRE_Int max_iterations );
HYPRE_Int hypre_FSAISetMaxIterations_long_dbl  ( void *data, HYPRE_Int max_iterations );
HYPRE_Int hypre_FSAISetMaxNnzRow_flt  ( void *data, HYPRE_Int max_nnz_row );
HYPRE_Int hypre_FSAISetMaxNnzRow_dbl  ( void *data, HYPRE_Int max_nnz_row );
HYPRE_Int hypre_FSAISetMaxNnzRow_long_dbl  ( void *data, HYPRE_Int max_nnz_row );
HYPRE_Int hypre_FSAISetMaxSteps_flt  ( void *data, HYPRE_Int max_steps );
HYPRE_Int hypre_FSAISetMaxSteps_dbl  ( void *data, HYPRE_Int max_steps );
HYPRE_Int hypre_FSAISetMaxSteps_long_dbl  ( void *data, HYPRE_Int max_steps );
HYPRE_Int hypre_FSAISetMaxStepSize_flt  ( void *data, HYPRE_Int max_step_size );
HYPRE_Int hypre_FSAISetMaxStepSize_dbl  ( void *data, HYPRE_Int max_step_size );
HYPRE_Int hypre_FSAISetMaxStepSize_long_dbl  ( void *data, HYPRE_Int max_step_size );
HYPRE_Int hypre_FSAISetNumIterations_flt  ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_FSAISetNumIterations_dbl  ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_FSAISetNumIterations_long_dbl  ( void *data, HYPRE_Int num_iterations );
HYPRE_Int hypre_FSAISetNumLevels_flt  ( void *data, HYPRE_Int num_levels );
HYPRE_Int hypre_FSAISetNumLevels_dbl  ( void *data, HYPRE_Int num_levels );
HYPRE_Int hypre_FSAISetNumLevels_long_dbl  ( void *data, HYPRE_Int num_levels );
HYPRE_Int hypre_FSAISetOmega_flt  ( void *data, hypre_float omega );
HYPRE_Int hypre_FSAISetOmega_dbl  ( void *data, hypre_double omega );
HYPRE_Int hypre_FSAISetOmega_long_dbl  ( void *data, hypre_long_double omega );
HYPRE_Int hypre_FSAISetPrintLevel_flt  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_FSAISetPrintLevel_dbl  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_FSAISetPrintLevel_long_dbl  ( void *data, HYPRE_Int print_level );
HYPRE_Int hypre_FSAISetThreshold_flt  ( void *data, hypre_float threshold );
HYPRE_Int hypre_FSAISetThreshold_dbl  ( void *data, hypre_double threshold );
HYPRE_Int hypre_FSAISetThreshold_long_dbl  ( void *data, hypre_long_double threshold );
HYPRE_Int hypre_FSAISetTolerance_flt  ( void *data, hypre_float tolerance );
HYPRE_Int hypre_FSAISetTolerance_dbl  ( void *data, hypre_double tolerance );
HYPRE_Int hypre_FSAISetTolerance_long_dbl  ( void *data, hypre_long_double tolerance );
HYPRE_Int hypre_FSAISetZeroGuess_flt  ( void *data, HYPRE_Int zero_guess );
HYPRE_Int hypre_FSAISetZeroGuess_dbl  ( void *data, HYPRE_Int zero_guess );
HYPRE_Int hypre_FSAISetZeroGuess_long_dbl  ( void *data, HYPRE_Int zero_guess );
HYPRE_Int hypre_AddToPattern_flt  ( hypre_Vector *kaporin_gradient, HYPRE_Int *kap_grad_nonzeros,
                               HYPRE_Int *S_Pattern, HYPRE_Int *S_nnz, HYPRE_Int *kg_marker,
                               HYPRE_Int max_step_size );
HYPRE_Int hypre_AddToPattern_dbl  ( hypre_Vector *kaporin_gradient, HYPRE_Int *kap_grad_nonzeros,
                               HYPRE_Int *S_Pattern, HYPRE_Int *S_nnz, HYPRE_Int *kg_marker,
                               HYPRE_Int max_step_size );
HYPRE_Int hypre_AddToPattern_long_dbl  ( hypre_Vector *kaporin_gradient, HYPRE_Int *kap_grad_nonzeros,
                               HYPRE_Int *S_Pattern, HYPRE_Int *S_nnz, HYPRE_Int *kg_marker,
                               HYPRE_Int max_step_size );
HYPRE_Int hypre_CSRMatrixExtractDenseMat_flt  ( hypre_CSRMatrix *A, hypre_Vector *A_sub,
                                           HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                                           HYPRE_Int *marker );
HYPRE_Int hypre_CSRMatrixExtractDenseMat_dbl  ( hypre_CSRMatrix *A, hypre_Vector *A_sub,
                                           HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                                           HYPRE_Int *marker );
HYPRE_Int hypre_CSRMatrixExtractDenseMat_long_dbl  ( hypre_CSRMatrix *A, hypre_Vector *A_sub,
                                           HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                                           HYPRE_Int *marker );
HYPRE_Int hypre_CSRMatrixExtractDenseRow_flt  ( hypre_CSRMatrix *A, hypre_Vector *A_subrow,
                                           HYPRE_Int *marker, HYPRE_Int row_num );
HYPRE_Int hypre_CSRMatrixExtractDenseRow_dbl  ( hypre_CSRMatrix *A, hypre_Vector *A_subrow,
                                           HYPRE_Int *marker, HYPRE_Int row_num );
HYPRE_Int hypre_CSRMatrixExtractDenseRow_long_dbl  ( hypre_CSRMatrix *A, hypre_Vector *A_subrow,
                                           HYPRE_Int *marker, HYPRE_Int row_num );
HYPRE_Int hypre_FindKapGrad_flt  ( hypre_CSRMatrix *A_diag, hypre_Vector *kaporin_gradient,
                              HYPRE_Int *kap_grad_nonzeros, hypre_Vector *G_temp,
                              HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                              HYPRE_Int max_row_size, HYPRE_Int row_num, HYPRE_Int *kg_marker );
HYPRE_Int hypre_FindKapGrad_dbl  ( hypre_CSRMatrix *A_diag, hypre_Vector *kaporin_gradient,
                              HYPRE_Int *kap_grad_nonzeros, hypre_Vector *G_temp,
                              HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                              HYPRE_Int max_row_size, HYPRE_Int row_num, HYPRE_Int *kg_marker );
HYPRE_Int hypre_FindKapGrad_long_dbl  ( hypre_CSRMatrix *A_diag, hypre_Vector *kaporin_gradient,
                              HYPRE_Int *kap_grad_nonzeros, hypre_Vector *G_temp,
                              HYPRE_Int *S_Pattern, HYPRE_Int S_nnz,
                              HYPRE_Int max_row_size, HYPRE_Int row_num, HYPRE_Int *kg_marker );
HYPRE_Int hypre_FSAIComputeOmega_flt  ( void *fsai_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIComputeOmega_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIComputeOmega_long_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIDumpLocalLSDense_flt  ( void *fsai_vdata, const char *filename,
                                       hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIDumpLocalLSDense_dbl  ( void *fsai_vdata, const char *filename,
                                       hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIDumpLocalLSDense_long_dbl  ( void *fsai_vdata, const char *filename,
                                       hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIPrintStats_flt  ( void *fsai_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIPrintStats_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAIPrintStats_long_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_FSAISetup_flt  ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u );
HYPRE_Int hypre_FSAISetup_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u );
HYPRE_Int hypre_FSAISetup_long_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupNative_flt  ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupNative_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupNative_long_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupOMPDyn_flt  ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupOMPDyn_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_FSAISetupOMPDyn_long_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A,
                                  hypre_ParVector *f, hypre_ParVector *u );
void hypre_qsort2_ci_flt  ( hypre_float *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_ci_dbl  ( hypre_double *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
void hypre_qsort2_ci_long_dbl  ( hypre_long_double *v, HYPRE_Int *w, HYPRE_Int left, HYPRE_Int right );
void hypre_swap2_ci_flt  ( hypre_float *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2_ci_dbl  ( hypre_double *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
void hypre_swap2_ci_long_dbl  ( hypre_long_double *v, HYPRE_Int *w, HYPRE_Int i, HYPRE_Int j );
HYPRE_Int hypre_FSAIApply_flt  ( void *fsai_vdata, hypre_float alpha, hypre_ParVector *b,
                            hypre_ParVector *x );
HYPRE_Int hypre_FSAIApply_dbl  ( void *fsai_vdata, hypre_double alpha, hypre_ParVector *b,
                            hypre_ParVector *x );
HYPRE_Int hypre_FSAIApply_long_dbl  ( void *fsai_vdata, hypre_long_double alpha, hypre_ParVector *b,
                            hypre_ParVector *x );
HYPRE_Int hypre_FSAISolve_flt  ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                            hypre_ParVector *x );
HYPRE_Int hypre_FSAISolve_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                            hypre_ParVector *x );
HYPRE_Int hypre_FSAISolve_long_dbl  ( void *fsai_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *b,
                            hypre_ParVector *x );
HYPRE_Int hypre_GaussElimSetup_flt  ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int relax_type );
HYPRE_Int hypre_GaussElimSetup_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int relax_type );
HYPRE_Int hypre_GaussElimSetup_long_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int relax_type );
HYPRE_Int hypre_GaussElimSolve_flt  ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int relax_type );
HYPRE_Int hypre_GaussElimSolve_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int relax_type );
HYPRE_Int hypre_GaussElimSolve_long_dbl  ( hypre_ParAMGData *amg_data, HYPRE_Int level,
                                 HYPRE_Int relax_type );
HYPRE_Int hypre_BoomerAMGBuildInterpGSMG_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, hypre_float trunc_factor, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpGSMG_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, hypre_double trunc_factor, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpGSMG_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, hypre_long_double trunc_factor, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpLS_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int num_smooth, hypre_float *SmoothVecs,
                                         hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpLS_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int num_smooth, hypre_double *SmoothVecs,
                                         hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpLS_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int num_smooth, hypre_long_double *SmoothVecs,
                                         hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGCreateSmoothDirs_flt  ( void *data, hypre_ParCSRMatrix *A,
                                            hypre_float *SmoothVecs, hypre_float thresh, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSmoothDirs_dbl  ( void *data, hypre_ParCSRMatrix *A,
                                            hypre_double *SmoothVecs, hypre_double thresh, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSmoothDirs_long_dbl  ( void *data, hypre_ParCSRMatrix *A,
                                            hypre_long_double *SmoothVecs, hypre_long_double thresh, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSmoothVecs_flt  ( void *data, hypre_ParCSRMatrix *A, HYPRE_Int num_sweeps,
                                            HYPRE_Int level, hypre_float **SmoothVecs_p );
HYPRE_Int hypre_BoomerAMGCreateSmoothVecs_dbl  ( void *data, hypre_ParCSRMatrix *A, HYPRE_Int num_sweeps,
                                            HYPRE_Int level, hypre_double **SmoothVecs_p );
HYPRE_Int hypre_BoomerAMGCreateSmoothVecs_long_dbl  ( void *data, hypre_ParCSRMatrix *A, HYPRE_Int num_sweeps,
                                            HYPRE_Int level, hypre_long_double **SmoothVecs_p );
HYPRE_Int hypre_BoomerAMGFitVectors_flt  ( HYPRE_Int ip, HYPRE_Int n, HYPRE_Int num, const hypre_float *V,
                                      HYPRE_Int nc, const HYPRE_Int *ind, hypre_float *val );
HYPRE_Int hypre_BoomerAMGFitVectors_dbl  ( HYPRE_Int ip, HYPRE_Int n, HYPRE_Int num, const hypre_double *V,
                                      HYPRE_Int nc, const HYPRE_Int *ind, hypre_double *val );
HYPRE_Int hypre_BoomerAMGFitVectors_long_dbl  ( HYPRE_Int ip, HYPRE_Int n, HYPRE_Int num, const hypre_long_double *V,
                                      HYPRE_Int nc, const HYPRE_Int *ind, hypre_long_double *val );
HYPRE_Int hypre_BoomerAMGNormalizeVecs_flt  ( HYPRE_Int n, HYPRE_Int num, hypre_float *V );
HYPRE_Int hypre_BoomerAMGNormalizeVecs_dbl  ( HYPRE_Int n, HYPRE_Int num, hypre_double *V );
HYPRE_Int hypre_BoomerAMGNormalizeVecs_long_dbl  ( HYPRE_Int n, HYPRE_Int num, hypre_long_double *V );
hypre_float hypre_ParCSRMatrixChooseThresh_flt  ( hypre_ParCSRMatrix *S );
hypre_double hypre_ParCSRMatrixChooseThresh_dbl  ( hypre_ParCSRMatrix *S );
hypre_long_double hypre_ParCSRMatrixChooseThresh_long_dbl  ( hypre_ParCSRMatrix *S );
HYPRE_Int hypre_ParCSRMatrixFillSmooth_flt  ( HYPRE_Int nsamples, hypre_float *samples,
                                         hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int num_functions, HYPRE_Int *dof_func );
HYPRE_Int hypre_ParCSRMatrixFillSmooth_dbl  ( HYPRE_Int nsamples, hypre_double *samples,
                                         hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int num_functions, HYPRE_Int *dof_func );
HYPRE_Int hypre_ParCSRMatrixFillSmooth_long_dbl  ( HYPRE_Int nsamples, hypre_long_double *samples,
                                         hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A, HYPRE_Int num_functions, HYPRE_Int *dof_func );
HYPRE_Int hypre_ParCSRMatrixThreshold_flt  ( hypre_ParCSRMatrix *A, hypre_float thresh );
HYPRE_Int hypre_ParCSRMatrixThreshold_dbl  ( hypre_ParCSRMatrix *A, hypre_double thresh );
HYPRE_Int hypre_ParCSRMatrixThreshold_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double thresh );
HYPRE_Int hypre_CSRMatrixDropInplace_flt ( hypre_CSRMatrix *A, hypre_float droptol,
                                      HYPRE_Int max_row_nnz );
HYPRE_Int hypre_CSRMatrixDropInplace_dbl ( hypre_CSRMatrix *A, hypre_double droptol,
                                      HYPRE_Int max_row_nnz );
HYPRE_Int hypre_CSRMatrixDropInplace_long_dbl ( hypre_CSRMatrix *A, hypre_long_double droptol,
                                      HYPRE_Int max_row_nnz );
HYPRE_Int hypre_CSRMatrixNormFro_flt ( hypre_CSRMatrix *A, hypre_float *norm_io);
HYPRE_Int hypre_CSRMatrixNormFro_dbl ( hypre_CSRMatrix *A, hypre_double *norm_io);
HYPRE_Int hypre_CSRMatrixNormFro_long_dbl ( hypre_CSRMatrix *A, hypre_long_double *norm_io);
HYPRE_Int hypre_CSRMatrixResNormFro_flt ( hypre_CSRMatrix *A, hypre_float *norm_io);
HYPRE_Int hypre_CSRMatrixResNormFro_dbl ( hypre_CSRMatrix *A, hypre_double *norm_io);
HYPRE_Int hypre_CSRMatrixResNormFro_long_dbl ( hypre_CSRMatrix *A, hypre_long_double *norm_io);
HYPRE_Int hypre_CSRMatrixTrace_flt ( hypre_CSRMatrix *A, hypre_float *trace_io);
HYPRE_Int hypre_CSRMatrixTrace_dbl ( hypre_CSRMatrix *A, hypre_double *trace_io);
HYPRE_Int hypre_CSRMatrixTrace_long_dbl ( hypre_CSRMatrix *A, hypre_long_double *trace_io);
HYPRE_Int hypre_ILUBuildRASExternalMatrix_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *rperm,
                                           HYPRE_Int **E_i, HYPRE_Int **E_j, hypre_float **E_data );
HYPRE_Int hypre_ILUBuildRASExternalMatrix_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *rperm,
                                           HYPRE_Int **E_i, HYPRE_Int **E_j, hypre_double **E_data );
HYPRE_Int hypre_ILUBuildRASExternalMatrix_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *rperm,
                                           HYPRE_Int **E_i, HYPRE_Int **E_j, hypre_long_double **E_data );
void *hypre_ILUCreate_flt  ( void );
void *hypre_ILUCreate_dbl  ( void );
void *hypre_ILUCreate_long_dbl  ( void );
HYPRE_Int hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal_flt ( hypre_CSRMatrix *matA,
                                                        hypre_CSRMatrix **M,
                                                        hypre_float droptol, hypre_float tol,
                                                        hypre_float eps_tol, HYPRE_Int max_row_nnz,
                                                        HYPRE_Int max_iter,
                                                        HYPRE_Int print_level );
HYPRE_Int hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal_dbl ( hypre_CSRMatrix *matA,
                                                        hypre_CSRMatrix **M,
                                                        hypre_double droptol, hypre_double tol,
                                                        hypre_double eps_tol, HYPRE_Int max_row_nnz,
                                                        HYPRE_Int max_iter,
                                                        HYPRE_Int print_level );
HYPRE_Int hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal_long_dbl ( hypre_CSRMatrix *matA,
                                                        hypre_CSRMatrix **M,
                                                        hypre_long_double droptol, hypre_long_double tol,
                                                        hypre_long_double eps_tol, HYPRE_Int max_row_nnz,
                                                        HYPRE_Int max_iter,
                                                        HYPRE_Int print_level );
HYPRE_Int hypre_ILUDestroy_flt  ( void *ilu_vdata );
HYPRE_Int hypre_ILUDestroy_dbl  ( void *ilu_vdata );
HYPRE_Int hypre_ILUDestroy_long_dbl  ( void *ilu_vdata );
HYPRE_Int hypre_ILUGetFinalRelativeResidualNorm_flt ( void *ilu_vdata, hypre_float *res_norm );
HYPRE_Int hypre_ILUGetFinalRelativeResidualNorm_dbl ( void *ilu_vdata, hypre_double *res_norm );
HYPRE_Int hypre_ILUGetFinalRelativeResidualNorm_long_dbl ( void *ilu_vdata, hypre_long_double *res_norm );
HYPRE_Int hypre_ILUGetInteriorExteriorPerm_flt ( hypre_ParCSRMatrix *A,
                                            HYPRE_MemoryLocation memory_location,
                                            HYPRE_Int **perm, HYPRE_Int *nLU,
                                            HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetInteriorExteriorPerm_dbl ( hypre_ParCSRMatrix *A,
                                            HYPRE_MemoryLocation memory_location,
                                            HYPRE_Int **perm, HYPRE_Int *nLU,
                                            HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetInteriorExteriorPerm_long_dbl ( hypre_ParCSRMatrix *A,
                                            HYPRE_MemoryLocation memory_location,
                                            HYPRE_Int **perm, HYPRE_Int *nLU,
                                            HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetLocalPerm_flt ( hypre_ParCSRMatrix *A, HYPRE_Int **perm_ptr,
                                 HYPRE_Int *nLU, HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetLocalPerm_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int **perm_ptr,
                                 HYPRE_Int *nLU, HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetLocalPerm_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int **perm_ptr,
                                 HYPRE_Int *nLU, HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetNumIterations_flt ( void *ilu_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ILUGetNumIterations_dbl ( void *ilu_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ILUGetNumIterations_long_dbl ( void *ilu_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_ILUGetPermddPQ_flt ( hypre_ParCSRMatrix *A, HYPRE_Int **io_pperm, HYPRE_Int **io_qperm,
                                hypre_float tol, HYPRE_Int *nB, HYPRE_Int *nI,
                                HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetPermddPQ_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int **io_pperm, HYPRE_Int **io_qperm,
                                hypre_double tol, HYPRE_Int *nB, HYPRE_Int *nI,
                                HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetPermddPQ_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int **io_pperm, HYPRE_Int **io_qperm,
                                hypre_long_double tol, HYPRE_Int *nB, HYPRE_Int *nI,
                                HYPRE_Int reordering_type );
HYPRE_Int hypre_ILUGetPermddPQPre_flt ( HYPRE_Int n, HYPRE_Int nLU, HYPRE_Int *A_diag_i,
                                   HYPRE_Int *A_diag_j, hypre_float *A_diag_data,
                                   hypre_float tol, HYPRE_Int *perm, HYPRE_Int *rperm,
                                   HYPRE_Int *pperm_pre, HYPRE_Int *qperm_pre, HYPRE_Int *nB );
HYPRE_Int hypre_ILUGetPermddPQPre_dbl ( HYPRE_Int n, HYPRE_Int nLU, HYPRE_Int *A_diag_i,
                                   HYPRE_Int *A_diag_j, hypre_double *A_diag_data,
                                   hypre_double tol, HYPRE_Int *perm, HYPRE_Int *rperm,
                                   HYPRE_Int *pperm_pre, HYPRE_Int *qperm_pre, HYPRE_Int *nB );
HYPRE_Int hypre_ILUGetPermddPQPre_long_dbl ( HYPRE_Int n, HYPRE_Int nLU, HYPRE_Int *A_diag_i,
                                   HYPRE_Int *A_diag_j, hypre_long_double *A_diag_data,
                                   hypre_long_double tol, HYPRE_Int *perm, HYPRE_Int *rperm,
                                   HYPRE_Int *pperm_pre, HYPRE_Int *qperm_pre, HYPRE_Int *nB );
HYPRE_Int hypre_ILULocalRCM_flt ( hypre_CSRMatrix *A, HYPRE_Int start, HYPRE_Int end,
                             HYPRE_Int **permp, HYPRE_Int **qpermp, HYPRE_Int sym );
HYPRE_Int hypre_ILULocalRCM_dbl ( hypre_CSRMatrix *A, HYPRE_Int start, HYPRE_Int end,
                             HYPRE_Int **permp, HYPRE_Int **qpermp, HYPRE_Int sym );
HYPRE_Int hypre_ILULocalRCM_long_dbl ( hypre_CSRMatrix *A, HYPRE_Int start, HYPRE_Int end,
                             HYPRE_Int **permp, HYPRE_Int **qpermp, HYPRE_Int sym );
HYPRE_Int hypre_ILULocalRCMBuildFinalPerm_flt ( HYPRE_Int start, HYPRE_Int end,
                                           HYPRE_Int *G_perm, HYPRE_Int *perm, HYPRE_Int *qperm,
                                           HYPRE_Int **permp, HYPRE_Int **qpermp );
HYPRE_Int hypre_ILULocalRCMBuildFinalPerm_dbl ( HYPRE_Int start, HYPRE_Int end,
                                           HYPRE_Int *G_perm, HYPRE_Int *perm, HYPRE_Int *qperm,
                                           HYPRE_Int **permp, HYPRE_Int **qpermp );
HYPRE_Int hypre_ILULocalRCMBuildFinalPerm_long_dbl ( HYPRE_Int start, HYPRE_Int end,
                                           HYPRE_Int *G_perm, HYPRE_Int *perm, HYPRE_Int *qperm,
                                           HYPRE_Int **permp, HYPRE_Int **qpermp );
HYPRE_Int hypre_ILULocalRCMBuildLevel_flt ( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                       HYPRE_Int *level_i, HYPRE_Int *level_j, HYPRE_Int *nlevp );
HYPRE_Int hypre_ILULocalRCMBuildLevel_dbl ( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                       HYPRE_Int *level_i, HYPRE_Int *level_j, HYPRE_Int *nlevp );
HYPRE_Int hypre_ILULocalRCMBuildLevel_long_dbl ( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                       HYPRE_Int *level_i, HYPRE_Int *level_j, HYPRE_Int *nlevp );
HYPRE_Int hypre_ILULocalRCMFindPPNode_flt ( hypre_CSRMatrix *A, HYPRE_Int *rootp, HYPRE_Int *marker );
HYPRE_Int hypre_ILULocalRCMFindPPNode_dbl ( hypre_CSRMatrix *A, HYPRE_Int *rootp, HYPRE_Int *marker );
HYPRE_Int hypre_ILULocalRCMFindPPNode_long_dbl ( hypre_CSRMatrix *A, HYPRE_Int *rootp, HYPRE_Int *marker );
HYPRE_Int hypre_ILULocalRCMMindegree_flt ( HYPRE_Int n, HYPRE_Int *degree,
                                      HYPRE_Int *marker, HYPRE_Int *rootp );
HYPRE_Int hypre_ILULocalRCMMindegree_dbl ( HYPRE_Int n, HYPRE_Int *degree,
                                      HYPRE_Int *marker, HYPRE_Int *rootp );
HYPRE_Int hypre_ILULocalRCMMindegree_long_dbl ( HYPRE_Int n, HYPRE_Int *degree,
                                      HYPRE_Int *marker, HYPRE_Int *rootp );
HYPRE_Int hypre_ILULocalRCMNumbering_flt ( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                      HYPRE_Int *perm, HYPRE_Int *current_nump );
HYPRE_Int hypre_ILULocalRCMNumbering_dbl ( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                      HYPRE_Int *perm, HYPRE_Int *current_nump );
HYPRE_Int hypre_ILULocalRCMNumbering_long_dbl ( hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                                      HYPRE_Int *perm, HYPRE_Int *current_nump );
HYPRE_Int hypre_ILULocalRCMOrder_flt ( hypre_CSRMatrix *A, HYPRE_Int *perm );
HYPRE_Int hypre_ILULocalRCMOrder_dbl ( hypre_CSRMatrix *A, HYPRE_Int *perm );
HYPRE_Int hypre_ILULocalRCMOrder_long_dbl ( hypre_CSRMatrix *A, HYPRE_Int *perm );
HYPRE_Int hypre_ILULocalRCMQsort_flt ( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end,
                                  HYPRE_Int *degree );
HYPRE_Int hypre_ILULocalRCMQsort_dbl ( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end,
                                  HYPRE_Int *degree );
HYPRE_Int hypre_ILULocalRCMQsort_long_dbl ( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end,
                                  HYPRE_Int *degree );
HYPRE_Int hypre_ILULocalRCMReverse_flt ( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end );
HYPRE_Int hypre_ILULocalRCMReverse_dbl ( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end );
HYPRE_Int hypre_ILULocalRCMReverse_long_dbl ( HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end );
HYPRE_Int hypre_ILUMaxQSplitRabsI_flt ( hypre_float *arrayR, HYPRE_Int *arrayI, HYPRE_Int left,
                                   HYPRE_Int bound, HYPRE_Int right );
HYPRE_Int hypre_ILUMaxQSplitRabsI_dbl ( hypre_double *arrayR, HYPRE_Int *arrayI, HYPRE_Int left,
                                   HYPRE_Int bound, HYPRE_Int right );
HYPRE_Int hypre_ILUMaxQSplitRabsI_long_dbl ( hypre_long_double *arrayR, HYPRE_Int *arrayI, HYPRE_Int left,
                                   HYPRE_Int bound, HYPRE_Int right );
HYPRE_Int hypre_ILUMaxRabs_flt ( hypre_float *array_data, HYPRE_Int *array_j, HYPRE_Int start,
                            HYPRE_Int end, HYPRE_Int nLU, HYPRE_Int *rperm, hypre_float *value,
                            HYPRE_Int *index, hypre_float *l1_norm, HYPRE_Int *nnz );
HYPRE_Int hypre_ILUMaxRabs_dbl ( hypre_double *array_data, HYPRE_Int *array_j, HYPRE_Int start,
                            HYPRE_Int end, HYPRE_Int nLU, HYPRE_Int *rperm, hypre_double *value,
                            HYPRE_Int *index, hypre_double *l1_norm, HYPRE_Int *nnz );
HYPRE_Int hypre_ILUMaxRabs_long_dbl ( hypre_long_double *array_data, HYPRE_Int *array_j, HYPRE_Int start,
                            HYPRE_Int end, HYPRE_Int nLU, HYPRE_Int *rperm, hypre_long_double *value,
                            HYPRE_Int *index, hypre_long_double *l1_norm, HYPRE_Int *nnz );
HYPRE_Int hypre_ILUMaxrHeapAddRabsI_flt ( hypre_float *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxrHeapAddRabsI_dbl ( hypre_double *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxrHeapAddRabsI_long_dbl ( hypre_long_double *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxrHeapRemoveRabsI_flt ( hypre_float *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxrHeapRemoveRabsI_dbl ( hypre_double *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMaxrHeapRemoveRabsI_long_dbl ( hypre_long_double *heap, HYPRE_Int *I1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddI_flt ( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddI_dbl ( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddI_long_dbl ( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIIIi_flt ( HYPRE_Int *heap, HYPRE_Int *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIIIi_dbl ( HYPRE_Int *heap, HYPRE_Int *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIIIi_long_dbl ( HYPRE_Int *heap, HYPRE_Int *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIRIi_flt ( HYPRE_Int *heap, hypre_float *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIRIi_dbl ( HYPRE_Int *heap, hypre_double *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapAddIRIi_long_dbl ( HYPRE_Int *heap, hypre_long_double *I1,
                                   HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveI_flt ( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveI_dbl ( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveI_long_dbl ( HYPRE_Int *heap, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIIIi_flt ( HYPRE_Int *heap, HYPRE_Int *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIIIi_dbl ( HYPRE_Int *heap, HYPRE_Int *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIIIi_long_dbl ( HYPRE_Int *heap, HYPRE_Int *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIRIi_flt ( HYPRE_Int *heap, hypre_float *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIRIi_dbl ( HYPRE_Int *heap, hypre_double *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUMinHeapRemoveIRIi_long_dbl ( HYPRE_Int *heap, hypre_long_double *I1,
                                      HYPRE_Int *Ii1, HYPRE_Int len );
HYPRE_Int hypre_ILUParCSRInverseNSH_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **M,
                                     hypre_float *droptol, hypre_float mr_tol,
                                     hypre_float nsh_tol, hypre_float eps_tol,
                                     HYPRE_Int mr_max_row_nnz, HYPRE_Int nsh_max_row_nnz,
                                     HYPRE_Int mr_max_iter, HYPRE_Int nsh_max_iter,
                                     HYPRE_Int mr_col_version, HYPRE_Int print_level );
HYPRE_Int hypre_ILUParCSRInverseNSH_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **M,
                                     hypre_double *droptol, hypre_double mr_tol,
                                     hypre_double nsh_tol, hypre_double eps_tol,
                                     HYPRE_Int mr_max_row_nnz, HYPRE_Int nsh_max_row_nnz,
                                     HYPRE_Int mr_max_iter, HYPRE_Int nsh_max_iter,
                                     HYPRE_Int mr_col_version, HYPRE_Int print_level );
HYPRE_Int hypre_ILUParCSRInverseNSH_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **M,
                                     hypre_long_double *droptol, hypre_long_double mr_tol,
                                     hypre_long_double nsh_tol, hypre_long_double eps_tol,
                                     HYPRE_Int mr_max_row_nnz, HYPRE_Int nsh_max_row_nnz,
                                     HYPRE_Int mr_max_iter, HYPRE_Int nsh_max_iter,
                                     HYPRE_Int mr_col_version, HYPRE_Int print_level );
HYPRE_Int hypre_ILUSetDropThreshold_flt ( void *ilu_vdata, hypre_float threshold );
HYPRE_Int hypre_ILUSetDropThreshold_dbl ( void *ilu_vdata, hypre_double threshold );
HYPRE_Int hypre_ILUSetDropThreshold_long_dbl ( void *ilu_vdata, hypre_long_double threshold );
HYPRE_Int hypre_ILUSetDropThresholdArray_flt ( void *ilu_vdata, hypre_float *threshold );
HYPRE_Int hypre_ILUSetDropThresholdArray_dbl ( void *ilu_vdata, hypre_double *threshold );
HYPRE_Int hypre_ILUSetDropThresholdArray_long_dbl ( void *ilu_vdata, hypre_long_double *threshold );
HYPRE_Int hypre_ILUSetLevelOfFill_flt ( void *ilu_vdata, HYPRE_Int lfil );
HYPRE_Int hypre_ILUSetLevelOfFill_dbl ( void *ilu_vdata, HYPRE_Int lfil );
HYPRE_Int hypre_ILUSetLevelOfFill_long_dbl ( void *ilu_vdata, HYPRE_Int lfil );
HYPRE_Int hypre_ILUSetLocalReordering_flt ( void *ilu_vdata, HYPRE_Int ordering_type );
HYPRE_Int hypre_ILUSetLocalReordering_dbl ( void *ilu_vdata, HYPRE_Int ordering_type );
HYPRE_Int hypre_ILUSetLocalReordering_long_dbl ( void *ilu_vdata, HYPRE_Int ordering_type );
HYPRE_Int hypre_ILUSetLogging_flt ( void *ilu_vdata, HYPRE_Int logging );
HYPRE_Int hypre_ILUSetLogging_dbl ( void *ilu_vdata, HYPRE_Int logging );
HYPRE_Int hypre_ILUSetLogging_long_dbl ( void *ilu_vdata, HYPRE_Int logging );
HYPRE_Int hypre_ILUSetLowerJacobiIters_flt ( void *ilu_vdata, HYPRE_Int lower_jacobi_iters );
HYPRE_Int hypre_ILUSetLowerJacobiIters_dbl ( void *ilu_vdata, HYPRE_Int lower_jacobi_iters );
HYPRE_Int hypre_ILUSetLowerJacobiIters_long_dbl ( void *ilu_vdata, HYPRE_Int lower_jacobi_iters );
HYPRE_Int hypre_ILUSetMaxIter_flt ( void *ilu_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_ILUSetMaxIter_dbl ( void *ilu_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_ILUSetMaxIter_long_dbl ( void *ilu_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_ILUSetMaxNnzPerRow_flt ( void *ilu_vdata, HYPRE_Int nzmax );
HYPRE_Int hypre_ILUSetMaxNnzPerRow_dbl ( void *ilu_vdata, HYPRE_Int nzmax );
HYPRE_Int hypre_ILUSetMaxNnzPerRow_long_dbl ( void *ilu_vdata, HYPRE_Int nzmax );
HYPRE_Int hypre_ILUSetPrintLevel_flt ( void *ilu_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_ILUSetPrintLevel_dbl ( void *ilu_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_ILUSetPrintLevel_long_dbl ( void *ilu_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_ILUSetSchurNSHDropThreshold_flt ( void *ilu_vdata, hypre_float threshold );
HYPRE_Int hypre_ILUSetSchurNSHDropThreshold_dbl ( void *ilu_vdata, hypre_double threshold );
HYPRE_Int hypre_ILUSetSchurNSHDropThreshold_long_dbl ( void *ilu_vdata, hypre_long_double threshold );
HYPRE_Int hypre_ILUSetSchurNSHDropThresholdArray_flt ( void *ilu_vdata, hypre_float *threshold );
HYPRE_Int hypre_ILUSetSchurNSHDropThresholdArray_dbl ( void *ilu_vdata, hypre_double *threshold );
HYPRE_Int hypre_ILUSetSchurNSHDropThresholdArray_long_dbl ( void *ilu_vdata, hypre_long_double *threshold );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThreshold_flt ( void *ilu_vdata, hypre_float sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThreshold_dbl ( void *ilu_vdata, hypre_double sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThreshold_long_dbl ( void *ilu_vdata, hypre_long_double sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThresholdArray_flt ( void *ilu_vdata,
                                                         hypre_float *sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThresholdArray_dbl ( void *ilu_vdata,
                                                         hypre_double *sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondILUDropThresholdArray_long_dbl ( void *ilu_vdata,
                                                         hypre_long_double *sp_ilu_droptol );
HYPRE_Int hypre_ILUSetSchurPrecondILULevelOfFill_flt ( void *ilu_vdata, HYPRE_Int sp_ilu_lfil );
HYPRE_Int hypre_ILUSetSchurPrecondILULevelOfFill_dbl ( void *ilu_vdata, HYPRE_Int sp_ilu_lfil );
HYPRE_Int hypre_ILUSetSchurPrecondILULevelOfFill_long_dbl ( void *ilu_vdata, HYPRE_Int sp_ilu_lfil );
HYPRE_Int hypre_ILUSetSchurPrecondILUMaxNnzPerRow_flt ( void *ilu_vdata,
                                                   HYPRE_Int sp_ilu_max_row_nnz );
HYPRE_Int hypre_ILUSetSchurPrecondILUMaxNnzPerRow_dbl ( void *ilu_vdata,
                                                   HYPRE_Int sp_ilu_max_row_nnz );
HYPRE_Int hypre_ILUSetSchurPrecondILUMaxNnzPerRow_long_dbl ( void *ilu_vdata,
                                                   HYPRE_Int sp_ilu_max_row_nnz );
HYPRE_Int hypre_ILUSetSchurPrecondILUType_flt ( void *ilu_vdata, HYPRE_Int sp_ilu_type );
HYPRE_Int hypre_ILUSetSchurPrecondILUType_dbl ( void *ilu_vdata, HYPRE_Int sp_ilu_type );
HYPRE_Int hypre_ILUSetSchurPrecondILUType_long_dbl ( void *ilu_vdata, HYPRE_Int sp_ilu_type );
HYPRE_Int hypre_ILUSetSchurPrecondLowerJacobiIters_flt ( void *ilu_vdata,
                                                    HYPRE_Int sp_lower_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurPrecondLowerJacobiIters_dbl ( void *ilu_vdata,
                                                    HYPRE_Int sp_lower_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurPrecondLowerJacobiIters_long_dbl ( void *ilu_vdata,
                                                    HYPRE_Int sp_lower_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurPrecondMaxIter_flt ( void *ilu_vdata, HYPRE_Int sp_max_iter );
HYPRE_Int hypre_ILUSetSchurPrecondMaxIter_dbl ( void *ilu_vdata, HYPRE_Int sp_max_iter );
HYPRE_Int hypre_ILUSetSchurPrecondMaxIter_long_dbl ( void *ilu_vdata, HYPRE_Int sp_max_iter );
HYPRE_Int hypre_ILUSetSchurPrecondPrintLevel_flt ( void *ilu_vdata, HYPRE_Int sp_print_level );
HYPRE_Int hypre_ILUSetSchurPrecondPrintLevel_dbl ( void *ilu_vdata, HYPRE_Int sp_print_level );
HYPRE_Int hypre_ILUSetSchurPrecondPrintLevel_long_dbl ( void *ilu_vdata, HYPRE_Int sp_print_level );
HYPRE_Int hypre_ILUSetSchurPrecondTol_flt ( void *ilu_vdata, HYPRE_Int sp_tol );
HYPRE_Int hypre_ILUSetSchurPrecondTol_dbl ( void *ilu_vdata, HYPRE_Int sp_tol );
HYPRE_Int hypre_ILUSetSchurPrecondTol_long_dbl ( void *ilu_vdata, HYPRE_Int sp_tol );
HYPRE_Int hypre_ILUSetSchurPrecondTriSolve_flt ( void *ilu_vdata, HYPRE_Int sp_tri_solve );
HYPRE_Int hypre_ILUSetSchurPrecondTriSolve_dbl ( void *ilu_vdata, HYPRE_Int sp_tri_solve );
HYPRE_Int hypre_ILUSetSchurPrecondTriSolve_long_dbl ( void *ilu_vdata, HYPRE_Int sp_tri_solve );
HYPRE_Int hypre_ILUSetSchurPrecondUpperJacobiIters_flt ( void *ilu_vdata,
                                                    HYPRE_Int sp_upper_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurPrecondUpperJacobiIters_dbl ( void *ilu_vdata,
                                                    HYPRE_Int sp_upper_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurPrecondUpperJacobiIters_long_dbl ( void *ilu_vdata,
                                                    HYPRE_Int sp_upper_jacobi_iters );
HYPRE_Int hypre_ILUSetSchurSolverAbsoluteTol_flt ( void *ilu_vdata, hypre_float ss_absolute_tol );
HYPRE_Int hypre_ILUSetSchurSolverAbsoluteTol_dbl ( void *ilu_vdata, hypre_double ss_absolute_tol );
HYPRE_Int hypre_ILUSetSchurSolverAbsoluteTol_long_dbl ( void *ilu_vdata, hypre_long_double ss_absolute_tol );
HYPRE_Int hypre_ILUSetSchurSolverLogging_flt ( void *ilu_vdata, HYPRE_Int ss_logging );
HYPRE_Int hypre_ILUSetSchurSolverLogging_dbl ( void *ilu_vdata, HYPRE_Int ss_logging );
HYPRE_Int hypre_ILUSetSchurSolverLogging_long_dbl ( void *ilu_vdata, HYPRE_Int ss_logging );
HYPRE_Int hypre_ILUSetSchurSolverMaxIter_flt ( void *ilu_vdata, HYPRE_Int ss_max_iter );
HYPRE_Int hypre_ILUSetSchurSolverMaxIter_dbl ( void *ilu_vdata, HYPRE_Int ss_max_iter );
HYPRE_Int hypre_ILUSetSchurSolverMaxIter_long_dbl ( void *ilu_vdata, HYPRE_Int ss_max_iter );
HYPRE_Int hypre_ILUSetSchurSolverPrintLevel_flt ( void *ilu_vdata, HYPRE_Int ss_print_level );
HYPRE_Int hypre_ILUSetSchurSolverPrintLevel_dbl ( void *ilu_vdata, HYPRE_Int ss_print_level );
HYPRE_Int hypre_ILUSetSchurSolverPrintLevel_long_dbl ( void *ilu_vdata, HYPRE_Int ss_print_level );
HYPRE_Int hypre_ILUSetSchurSolverRelChange_flt ( void *ilu_vdata, HYPRE_Int ss_rel_change );
HYPRE_Int hypre_ILUSetSchurSolverRelChange_dbl ( void *ilu_vdata, HYPRE_Int ss_rel_change );
HYPRE_Int hypre_ILUSetSchurSolverRelChange_long_dbl ( void *ilu_vdata, HYPRE_Int ss_rel_change );
HYPRE_Int hypre_ILUSetSchurSolverTol_flt ( void *ilu_vdata, hypre_float ss_tol );
HYPRE_Int hypre_ILUSetSchurSolverTol_dbl ( void *ilu_vdata, hypre_double ss_tol );
HYPRE_Int hypre_ILUSetSchurSolverTol_long_dbl ( void *ilu_vdata, hypre_long_double ss_tol );
HYPRE_Int hypre_ILUSetTol_flt ( void *ilu_vdata, hypre_float tol );
HYPRE_Int hypre_ILUSetTol_dbl ( void *ilu_vdata, hypre_double tol );
HYPRE_Int hypre_ILUSetTol_long_dbl ( void *ilu_vdata, hypre_long_double tol );
HYPRE_Int hypre_ILUSetTriSolve_flt ( void *ilu_vdata, HYPRE_Int tri_solve );
HYPRE_Int hypre_ILUSetTriSolve_dbl ( void *ilu_vdata, HYPRE_Int tri_solve );
HYPRE_Int hypre_ILUSetTriSolve_long_dbl ( void *ilu_vdata, HYPRE_Int tri_solve );
HYPRE_Int hypre_ILUSetType_flt ( void *ilu_vdata, HYPRE_Int ilu_type );
HYPRE_Int hypre_ILUSetType_dbl ( void *ilu_vdata, HYPRE_Int ilu_type );
HYPRE_Int hypre_ILUSetType_long_dbl ( void *ilu_vdata, HYPRE_Int ilu_type );
HYPRE_Int hypre_ILUSetUpperJacobiIters_flt ( void *ilu_vdata, HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSetUpperJacobiIters_dbl ( void *ilu_vdata, HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSetUpperJacobiIters_long_dbl ( void *ilu_vdata, HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSortOffdColmap_flt ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ILUSortOffdColmap_dbl ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ILUSortOffdColmap_long_dbl ( hypre_ParCSRMatrix *A );
HYPRE_Int hypre_ILUWriteSolverParams_flt ( void *ilu_vdata );
HYPRE_Int hypre_ILUWriteSolverParams_dbl ( void *ilu_vdata );
HYPRE_Int hypre_ILUWriteSolverParams_long_dbl ( void *ilu_vdata );
void *hypre_NSHCreate_flt ( void );
void *hypre_NSHCreate_dbl ( void );
void *hypre_NSHCreate_long_dbl ( void );
HYPRE_Int hypre_NSHDestroy_flt ( void *data );
HYPRE_Int hypre_NSHDestroy_dbl ( void *data );
HYPRE_Int hypre_NSHDestroy_long_dbl ( void *data );
HYPRE_Int hypre_NSHSetColVersion_flt ( void *nsh_vdata, HYPRE_Int mr_col_version );
HYPRE_Int hypre_NSHSetColVersion_dbl ( void *nsh_vdata, HYPRE_Int mr_col_version );
HYPRE_Int hypre_NSHSetColVersion_long_dbl ( void *nsh_vdata, HYPRE_Int mr_col_version );
HYPRE_Int hypre_NSHSetDropThreshold_flt ( void *nsh_vdata, hypre_float droptol );
HYPRE_Int hypre_NSHSetDropThreshold_dbl ( void *nsh_vdata, hypre_double droptol );
HYPRE_Int hypre_NSHSetDropThreshold_long_dbl ( void *nsh_vdata, hypre_long_double droptol );
HYPRE_Int hypre_NSHSetDropThresholdArray_flt ( void *nsh_vdata, hypre_float *droptol );
HYPRE_Int hypre_NSHSetDropThresholdArray_dbl ( void *nsh_vdata, hypre_double *droptol );
HYPRE_Int hypre_NSHSetDropThresholdArray_long_dbl ( void *nsh_vdata, hypre_long_double *droptol );
HYPRE_Int hypre_NSHSetGlobalSolver_flt ( void *nsh_vdata, HYPRE_Int global_solver );
HYPRE_Int hypre_NSHSetGlobalSolver_dbl ( void *nsh_vdata, HYPRE_Int global_solver );
HYPRE_Int hypre_NSHSetGlobalSolver_long_dbl ( void *nsh_vdata, HYPRE_Int global_solver );
HYPRE_Int hypre_NSHSetLogging_flt ( void *nsh_vdata, HYPRE_Int logging );
HYPRE_Int hypre_NSHSetLogging_dbl ( void *nsh_vdata, HYPRE_Int logging );
HYPRE_Int hypre_NSHSetLogging_long_dbl ( void *nsh_vdata, HYPRE_Int logging );
HYPRE_Int hypre_NSHSetMaxIter_flt ( void *nsh_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_NSHSetMaxIter_dbl ( void *nsh_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_NSHSetMaxIter_long_dbl ( void *nsh_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_NSHSetMRMaxIter_flt ( void *nsh_vdata, HYPRE_Int mr_max_iter );
HYPRE_Int hypre_NSHSetMRMaxIter_dbl ( void *nsh_vdata, HYPRE_Int mr_max_iter );
HYPRE_Int hypre_NSHSetMRMaxIter_long_dbl ( void *nsh_vdata, HYPRE_Int mr_max_iter );
HYPRE_Int hypre_NSHSetMRMaxRowNnz_flt ( void *nsh_vdata, HYPRE_Int mr_max_row_nnz );
HYPRE_Int hypre_NSHSetMRMaxRowNnz_dbl ( void *nsh_vdata, HYPRE_Int mr_max_row_nnz );
HYPRE_Int hypre_NSHSetMRMaxRowNnz_long_dbl ( void *nsh_vdata, HYPRE_Int mr_max_row_nnz );
HYPRE_Int hypre_NSHSetMRTol_flt ( void *nsh_vdata, hypre_float mr_tol );
HYPRE_Int hypre_NSHSetMRTol_dbl ( void *nsh_vdata, hypre_double mr_tol );
HYPRE_Int hypre_NSHSetMRTol_long_dbl ( void *nsh_vdata, hypre_long_double mr_tol );
HYPRE_Int hypre_NSHSetNSHMaxIter_flt ( void *nsh_vdata, HYPRE_Int nsh_max_iter );
HYPRE_Int hypre_NSHSetNSHMaxIter_dbl ( void *nsh_vdata, HYPRE_Int nsh_max_iter );
HYPRE_Int hypre_NSHSetNSHMaxIter_long_dbl ( void *nsh_vdata, HYPRE_Int nsh_max_iter );
HYPRE_Int hypre_NSHSetNSHMaxRowNnz_flt ( void *nsh_vdata, HYPRE_Int nsh_max_row_nnz );
HYPRE_Int hypre_NSHSetNSHMaxRowNnz_dbl ( void *nsh_vdata, HYPRE_Int nsh_max_row_nnz );
HYPRE_Int hypre_NSHSetNSHMaxRowNnz_long_dbl ( void *nsh_vdata, HYPRE_Int nsh_max_row_nnz );
HYPRE_Int hypre_NSHSetNSHTol_flt ( void *nsh_vdata, hypre_float nsh_tol );
HYPRE_Int hypre_NSHSetNSHTol_dbl ( void *nsh_vdata, hypre_double nsh_tol );
HYPRE_Int hypre_NSHSetNSHTol_long_dbl ( void *nsh_vdata, hypre_long_double nsh_tol );
HYPRE_Int hypre_NSHSetPrintLevel_flt ( void *nsh_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_NSHSetPrintLevel_dbl ( void *nsh_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_NSHSetPrintLevel_long_dbl ( void *nsh_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_NSHSetTol_flt ( void *nsh_vdata, hypre_float tol );
HYPRE_Int hypre_NSHSetTol_dbl ( void *nsh_vdata, hypre_double tol );
HYPRE_Int hypre_NSHSetTol_long_dbl ( void *nsh_vdata, hypre_long_double tol );
HYPRE_Int hypre_NSHWriteSolverParams_flt ( void *nsh_vdata );
HYPRE_Int hypre_NSHWriteSolverParams_dbl ( void *nsh_vdata );
HYPRE_Int hypre_NSHWriteSolverParams_long_dbl ( void *nsh_vdata );
HYPRE_Int hypre_ParCSRMatrixNormFro_flt ( hypre_ParCSRMatrix *A, hypre_float *norm_io);
HYPRE_Int hypre_ParCSRMatrixNormFro_dbl ( hypre_ParCSRMatrix *A, hypre_double *norm_io);
HYPRE_Int hypre_ParCSRMatrixNormFro_long_dbl ( hypre_ParCSRMatrix *A, hypre_long_double *norm_io);
HYPRE_Int hypre_ParCSRMatrixResNormFro_flt ( hypre_ParCSRMatrix *A, hypre_float *norm_io);
HYPRE_Int hypre_ParCSRMatrixResNormFro_dbl ( hypre_ParCSRMatrix *A, hypre_double *norm_io);
HYPRE_Int hypre_ParCSRMatrixResNormFro_long_dbl ( hypre_ParCSRMatrix *A, hypre_long_double *norm_io);
HYPRE_Int hypre_ParILURAPSchurGMRESCommInfoHost_flt ( void *ilu_vdata, HYPRE_Int *my_id,
                                                 HYPRE_Int *num_procs );
HYPRE_Int hypre_ParILURAPSchurGMRESCommInfoHost_dbl ( void *ilu_vdata, HYPRE_Int *my_id,
                                                 HYPRE_Int *num_procs );
HYPRE_Int hypre_ParILURAPSchurGMRESCommInfoHost_long_dbl ( void *ilu_vdata, HYPRE_Int *my_id,
                                                 HYPRE_Int *num_procs );
HYPRE_Int hypre_ParILURAPSchurGMRESMatvecHost_flt ( void *matvec_data, hypre_float alpha,
                                               void *ilu_vdata, void *x,
                                               hypre_float beta, void *y );
HYPRE_Int hypre_ParILURAPSchurGMRESMatvecHost_dbl ( void *matvec_data, hypre_double alpha,
                                               void *ilu_vdata, void *x,
                                               hypre_double beta, void *y );
HYPRE_Int hypre_ParILURAPSchurGMRESMatvecHost_long_dbl ( void *matvec_data, hypre_long_double alpha,
                                               void *ilu_vdata, void *x,
                                               hypre_long_double beta, void *y );
HYPRE_Int hypre_ParILURAPSchurGMRESSolveHost_flt ( void *ilu_vdata, void *ilu_vdata2,
                                              hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParILURAPSchurGMRESSolveHost_dbl ( void *ilu_vdata, void *ilu_vdata2,
                                              hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParILURAPSchurGMRESSolveHost_long_dbl ( void *ilu_vdata, void *ilu_vdata2,
                                              hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSetup_flt ( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSetup_dbl ( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSetup_long_dbl ( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSetupILU0_flt ( hypre_ParCSRMatrix  *A, HYPRE_Int *perm, HYPRE_Int *qperm,
                              HYPRE_Int nLU, HYPRE_Int nI, hypre_ParCSRMatrix **Lptr,
                              hypre_float **Dptr, hypre_ParCSRMatrix **Uptr,
                              hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILU0_dbl ( hypre_ParCSRMatrix  *A, HYPRE_Int *perm, HYPRE_Int *qperm,
                              HYPRE_Int nLU, HYPRE_Int nI, hypre_ParCSRMatrix **Lptr,
                              hypre_double **Dptr, hypre_ParCSRMatrix **Uptr,
                              hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILU0_long_dbl ( hypre_ParCSRMatrix  *A, HYPRE_Int *perm, HYPRE_Int *qperm,
                              HYPRE_Int nLU, HYPRE_Int nI, hypre_ParCSRMatrix **Lptr,
                              hypre_long_double **Dptr, hypre_ParCSRMatrix **Uptr,
                              hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILU0RAS_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 hypre_float **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILU0RAS_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 hypre_double **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILU0RAS_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 hypre_long_double **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUK_flt ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *permp,
                              HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                              hypre_ParCSRMatrix **Lptr, hypre_float **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUK_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *permp,
                              HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                              hypre_ParCSRMatrix **Lptr, hypre_double **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUK_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *permp,
                              HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                              hypre_ParCSRMatrix **Lptr, hypre_long_double **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUKRAS_flt ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 hypre_float **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUKRAS_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 hypre_double **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUKRAS_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *perm,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
                                 hypre_long_double **Dptr, hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUKRASSymbolic_flt ( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                         HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                                         HYPRE_Int *E_i, HYPRE_Int *E_j, HYPRE_Int ext,
                                         HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                         HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                         HYPRE_Int *U_diag_i, HYPRE_Int **L_diag_j,
                                         HYPRE_Int **U_diag_j );
HYPRE_Int hypre_ILUSetupILUKRASSymbolic_dbl ( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                         HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                                         HYPRE_Int *E_i, HYPRE_Int *E_j, HYPRE_Int ext,
                                         HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                         HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                         HYPRE_Int *U_diag_i, HYPRE_Int **L_diag_j,
                                         HYPRE_Int **U_diag_j );
HYPRE_Int hypre_ILUSetupILUKRASSymbolic_long_dbl ( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                         HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                                         HYPRE_Int *E_i, HYPRE_Int *E_j, HYPRE_Int ext,
                                         HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                         HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                         HYPRE_Int *U_diag_i, HYPRE_Int **L_diag_j,
                                         HYPRE_Int **U_diag_j );
HYPRE_Int hypre_ILUSetupILUKSymbolic_flt ( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                      HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                      HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                      HYPRE_Int *U_diag_i, HYPRE_Int *S_diag_i,
                                      HYPRE_Int **L_diag_j, HYPRE_Int **U_diag_j,
                                      HYPRE_Int **S_diag_j, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUKSymbolic_dbl ( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                      HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                      HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                      HYPRE_Int *U_diag_i, HYPRE_Int *S_diag_i,
                                      HYPRE_Int **L_diag_j, HYPRE_Int **U_diag_j,
                                      HYPRE_Int **S_diag_j, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUKSymbolic_long_dbl ( HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j,
                                      HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *rperm,
                                      HYPRE_Int *iw, HYPRE_Int nLU, HYPRE_Int *L_diag_i,
                                      HYPRE_Int *U_diag_i, HYPRE_Int *S_diag_i,
                                      HYPRE_Int **L_diag_j, HYPRE_Int **U_diag_j,
                                      HYPRE_Int **S_diag_j, HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUT_flt ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, hypre_float *tol,
                              HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU,
                              HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, hypre_float **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUT_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, hypre_double *tol,
                              HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU,
                              HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, hypre_double **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUT_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil, hypre_long_double *tol,
                              HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU,
                              HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, hypre_long_double **Dptr,
                              hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                              HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupILUTRAS_flt ( hypre_ParCSRMatrix *A, HYPRE_Int lfil,
                                 hypre_float *tol, HYPRE_Int *perm, HYPRE_Int nLU,
                                 hypre_ParCSRMatrix **Lptr, hypre_float **Dptr,
                                 hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUTRAS_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil,
                                 hypre_double *tol, HYPRE_Int *perm, HYPRE_Int nLU,
                                 hypre_ParCSRMatrix **Lptr, hypre_double **Dptr,
                                 hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupILUTRAS_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int lfil,
                                 hypre_long_double *tol, HYPRE_Int *perm, HYPRE_Int nLU,
                                 hypre_ParCSRMatrix **Lptr, hypre_long_double **Dptr,
                                 hypre_ParCSRMatrix **Uptr );
HYPRE_Int hypre_ILUSetupLDUtoCusparse_flt ( hypre_ParCSRMatrix *L, hypre_float *D,
                                       hypre_ParCSRMatrix  *U, hypre_ParCSRMatrix **LDUp );
HYPRE_Int hypre_ILUSetupLDUtoCusparse_dbl ( hypre_ParCSRMatrix *L, hypre_double *D,
                                       hypre_ParCSRMatrix  *U, hypre_ParCSRMatrix **LDUp );
HYPRE_Int hypre_ILUSetupLDUtoCusparse_long_dbl ( hypre_ParCSRMatrix *L, hypre_long_double *D,
                                       hypre_ParCSRMatrix  *U, hypre_ParCSRMatrix **LDUp );
HYPRE_Int hypre_ILUSetupMILU0_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *permp,
                               HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                               hypre_ParCSRMatrix **Lptr, hypre_float **Dptr,
                               hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                               HYPRE_Int **u_end, HYPRE_Int modified );
HYPRE_Int hypre_ILUSetupMILU0_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *permp,
                               HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                               hypre_ParCSRMatrix **Lptr, hypre_double **Dptr,
                               hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                               HYPRE_Int **u_end, HYPRE_Int modified );
HYPRE_Int hypre_ILUSetupMILU0_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *permp,
                               HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
                               hypre_ParCSRMatrix **Lptr, hypre_long_double **Dptr,
                               hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr,
                               HYPRE_Int **u_end, HYPRE_Int modified );
HYPRE_Int hypre_ILUSetupRAPILU0_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr, hypre_float **Dptr,
                                 hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **mLptr,
                                 hypre_float **mDptr, hypre_ParCSRMatrix **mUptr,
                                 HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupRAPILU0_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr, hypre_double **Dptr,
                                 hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **mLptr,
                                 hypre_double **mDptr, hypre_ParCSRMatrix **mUptr,
                                 HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupRAPILU0_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                 HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr, hypre_long_double **Dptr,
                                 hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **mLptr,
                                 hypre_long_double **mDptr, hypre_ParCSRMatrix **mUptr,
                                 HYPRE_Int **u_end );
HYPRE_Int hypre_ILUSetupRAPILU0Device_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                       HYPRE_Int nLU, hypre_ParCSRMatrix **Apermptr,
                                       hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **ALUptr,
                                       hypre_CSRMatrix **BLUptr, hypre_CSRMatrix **CLUptr,
                                       hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr,
                                       HYPRE_Int test_opt );
HYPRE_Int hypre_ILUSetupRAPILU0Device_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                       HYPRE_Int nLU, hypre_ParCSRMatrix **Apermptr,
                                       hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **ALUptr,
                                       hypre_CSRMatrix **BLUptr, hypre_CSRMatrix **CLUptr,
                                       hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr,
                                       HYPRE_Int test_opt );
HYPRE_Int hypre_ILUSetupRAPILU0Device_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n,
                                       HYPRE_Int nLU, hypre_ParCSRMatrix **Apermptr,
                                       hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **ALUptr,
                                       hypre_CSRMatrix **BLUptr, hypre_CSRMatrix **CLUptr,
                                       hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr,
                                       HYPRE_Int test_opt );
HYPRE_Int hypre_ILUSetupRAPMILU0_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **ALUp,
                                  HYPRE_Int modified );
HYPRE_Int hypre_ILUSetupRAPMILU0_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **ALUp,
                                  HYPRE_Int modified );
HYPRE_Int hypre_ILUSetupRAPMILU0_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **ALUp,
                                  HYPRE_Int modified );
HYPRE_Int hypre_NSHSetup_flt ( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_NSHSetup_dbl ( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_NSHSetup_long_dbl ( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ParILUExtractEBFC_flt ( hypre_CSRMatrix *A_diag, HYPRE_Int nLU,
                                   hypre_CSRMatrix **Bp, hypre_CSRMatrix **Cp,
                                   hypre_CSRMatrix **Ep, hypre_CSRMatrix **Fp );
HYPRE_Int hypre_ParILUExtractEBFC_dbl ( hypre_CSRMatrix *A_diag, HYPRE_Int nLU,
                                   hypre_CSRMatrix **Bp, hypre_CSRMatrix **Cp,
                                   hypre_CSRMatrix **Ep, hypre_CSRMatrix **Fp );
HYPRE_Int hypre_ParILUExtractEBFC_long_dbl ( hypre_CSRMatrix *A_diag, HYPRE_Int nLU,
                                   hypre_CSRMatrix **Bp, hypre_CSRMatrix **Cp,
                                   hypre_CSRMatrix **Ep, hypre_CSRMatrix **Fp );
HYPRE_Int hypre_ParILURAPReorder_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                  HYPRE_Int *rqperm, hypre_ParCSRMatrix **A_pq );
HYPRE_Int hypre_ParILURAPReorder_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                  HYPRE_Int *rqperm, hypre_ParCSRMatrix **A_pq );
HYPRE_Int hypre_ParILURAPReorder_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *perm,
                                  HYPRE_Int *rqperm, hypre_ParCSRMatrix **A_pq );
HYPRE_Int hypre_ILUSolve_flt ( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSolve_dbl ( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSolve_long_dbl ( void *ilu_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_ILUSolveLU_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                            hypre_ParCSRMatrix *L, hypre_float *D, hypre_ParCSRMatrix *U,
                            hypre_ParVector *ftemp, hypre_ParVector *utemp );
HYPRE_Int hypre_ILUSolveLU_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                            hypre_ParCSRMatrix *L, hypre_double *D, hypre_ParCSRMatrix *U,
                            hypre_ParVector *ftemp, hypre_ParVector *utemp );
HYPRE_Int hypre_ILUSolveLU_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                            hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                            hypre_ParCSRMatrix *L, hypre_long_double *D, hypre_ParCSRMatrix *U,
                            hypre_ParVector *ftemp, hypre_ParVector *utemp );
HYPRE_Int hypre_ILUSolveLUIter_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                hypre_ParCSRMatrix *L, hypre_float *D, hypre_ParCSRMatrix *U,
                                hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                hypre_ParVector *xtemp, HYPRE_Int lower_jacobi_iters,
                                HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSolveLUIter_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                hypre_ParCSRMatrix *L, hypre_double *D, hypre_ParCSRMatrix *U,
                                hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                hypre_ParVector *xtemp, HYPRE_Int lower_jacobi_iters,
                                HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSolveLUIter_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                hypre_ParCSRMatrix *L, hypre_long_double *D, hypre_ParCSRMatrix *U,
                                hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                hypre_ParVector *xtemp, HYPRE_Int lower_jacobi_iters,
                                HYPRE_Int upper_jacobi_iters );
HYPRE_Int hypre_ILUSolveLURAS_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                               HYPRE_Int *perm, hypre_ParCSRMatrix *L, hypre_float *D,
                               hypre_ParCSRMatrix *U, hypre_ParVector *ftemp,
                               hypre_ParVector *utemp, hypre_float *fext, hypre_float *uext );
HYPRE_Int hypre_ILUSolveLURAS_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                               HYPRE_Int *perm, hypre_ParCSRMatrix *L, hypre_double *D,
                               hypre_ParCSRMatrix *U, hypre_ParVector *ftemp,
                               hypre_ParVector *utemp, hypre_double *fext, hypre_double *uext );
HYPRE_Int hypre_ILUSolveLURAS_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_ParVector *u,
                               HYPRE_Int *perm, hypre_ParCSRMatrix *L, hypre_long_double *D,
                               hypre_ParCSRMatrix *U, hypre_ParVector *ftemp,
                               hypre_ParVector *utemp, hypre_long_double *fext, hypre_long_double *uext );
HYPRE_Int hypre_ILUSolveRAPGMRESHost_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                      hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                      hypre_ParCSRMatrix *L, hypre_float *D, hypre_ParCSRMatrix *U,
                                      hypre_ParCSRMatrix *mL, hypre_float *mD,
                                      hypre_ParCSRMatrix *mU, hypre_ParVector *ftemp,
                                      hypre_ParVector *utemp, hypre_ParVector *xtemp,
                                      hypre_ParVector *ytemp, HYPRE_Solver schur_solver,
                                      HYPRE_Solver schur_precond, hypre_ParVector *rhs,
                                      hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveRAPGMRESHost_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                      hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                      hypre_ParCSRMatrix *L, hypre_double *D, hypre_ParCSRMatrix *U,
                                      hypre_ParCSRMatrix *mL, hypre_double *mD,
                                      hypre_ParCSRMatrix *mU, hypre_ParVector *ftemp,
                                      hypre_ParVector *utemp, hypre_ParVector *xtemp,
                                      hypre_ParVector *ytemp, HYPRE_Solver schur_solver,
                                      HYPRE_Solver schur_precond, hypre_ParVector *rhs,
                                      hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveRAPGMRESHost_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                      hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                      hypre_ParCSRMatrix *L, hypre_long_double *D, hypre_ParCSRMatrix *U,
                                      hypre_ParCSRMatrix *mL, hypre_long_double *mD,
                                      hypre_ParCSRMatrix *mU, hypre_ParVector *ftemp,
                                      hypre_ParVector *utemp, hypre_ParVector *xtemp,
                                      hypre_ParVector *ytemp, HYPRE_Solver schur_solver,
                                      HYPRE_Solver schur_precond, hypre_ParVector *rhs,
                                      hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveSchurGMRES_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int *perm,
                                    HYPRE_Int *qperm, HYPRE_Int nLU,
                                    hypre_ParCSRMatrix *L, hypre_float *D,
                                    hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                    hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                    HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                                    hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveSchurGMRES_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int *perm,
                                    HYPRE_Int *qperm, HYPRE_Int nLU,
                                    hypre_ParCSRMatrix *L, hypre_double *D,
                                    hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                    hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                    HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                                    hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveSchurGMRES_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int *perm,
                                    HYPRE_Int *qperm, HYPRE_Int nLU,
                                    hypre_ParCSRMatrix *L, hypre_long_double *D,
                                    hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                    hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                    HYPRE_Solver schur_solver, HYPRE_Solver schur_precond,
                                    hypre_ParVector *rhs, hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveSchurNSH_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                  hypre_ParCSRMatrix *L, hypre_float *D,
                                  hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                  hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                  HYPRE_Solver schur_solver, hypre_ParVector *rhs,
                                  hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveSchurNSH_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                  hypre_ParCSRMatrix *L, hypre_double *D,
                                  hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                  hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                  HYPRE_Solver schur_solver, hypre_ParVector *rhs,
                                  hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_ILUSolveSchurNSH_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                  hypre_ParVector *u, HYPRE_Int *perm, HYPRE_Int nLU,
                                  hypre_ParCSRMatrix *L, hypre_long_double *D,
                                  hypre_ParCSRMatrix *U, hypre_ParCSRMatrix *S,
                                  hypre_ParVector *ftemp, hypre_ParVector *utemp,
                                  HYPRE_Solver schur_solver, hypre_ParVector *rhs,
                                  hypre_ParVector *x, HYPRE_Int *u_end );
HYPRE_Int hypre_NSHSolve_flt ( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_NSHSolve_dbl ( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_NSHSolve_long_dbl ( void *nsh_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_NSHSolveInverse_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, hypre_ParCSRMatrix *M,
                                 hypre_ParVector *ftemp, hypre_ParVector *utemp );
HYPRE_Int hypre_NSHSolveInverse_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, hypre_ParCSRMatrix *M,
                                 hypre_ParVector *ftemp, hypre_ParVector *utemp );
HYPRE_Int hypre_NSHSolveInverse_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, hypre_ParCSRMatrix *M,
                                 hypre_ParVector *ftemp, hypre_ParVector *utemp );
HYPRE_Int hypre_BoomerAMGIndepSet_flt  ( hypre_ParCSRMatrix *S, hypre_float *measure_array,
                                    HYPRE_Int *graph_array, HYPRE_Int graph_array_size, HYPRE_Int *graph_array_offd,
                                    HYPRE_Int graph_array_offd_size, HYPRE_Int *IS_marker, HYPRE_Int *IS_marker_offd );
HYPRE_Int hypre_BoomerAMGIndepSet_dbl  ( hypre_ParCSRMatrix *S, hypre_double *measure_array,
                                    HYPRE_Int *graph_array, HYPRE_Int graph_array_size, HYPRE_Int *graph_array_offd,
                                    HYPRE_Int graph_array_offd_size, HYPRE_Int *IS_marker, HYPRE_Int *IS_marker_offd );
HYPRE_Int hypre_BoomerAMGIndepSet_long_dbl  ( hypre_ParCSRMatrix *S, hypre_long_double *measure_array,
                                    HYPRE_Int *graph_array, HYPRE_Int graph_array_size, HYPRE_Int *graph_array_offd,
                                    HYPRE_Int graph_array_offd_size, HYPRE_Int *IS_marker, HYPRE_Int *IS_marker_offd );
HYPRE_Int hypre_BoomerAMGIndepSetInit_flt  ( hypre_ParCSRMatrix *S, hypre_float *measure_array,
                                        HYPRE_Int seq_rand );
HYPRE_Int hypre_BoomerAMGIndepSetInit_dbl  ( hypre_ParCSRMatrix *S, hypre_double *measure_array,
                                        HYPRE_Int seq_rand );
HYPRE_Int hypre_BoomerAMGIndepSetInit_long_dbl  ( hypre_ParCSRMatrix *S, hypre_long_double *measure_array,
                                        HYPRE_Int seq_rand );
HYPRE_Int hypre_BoomerAMGBuildDirInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, HYPRE_Int interp_type,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildDirInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, HYPRE_Int interp_type,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildDirInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, HYPRE_Int interp_type,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                       hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                       HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                       hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                       HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                       hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                       HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpHE_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpHE_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpHE_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpModUnk_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpModUnk_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpModUnk_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildInterpOnePnt_flt ( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildInterpOnePnt_dbl ( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildInterpOnePnt_long_dbl ( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildInterpOnePntHost_flt ( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildInterpOnePntHost_dbl ( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildInterpOnePntHost_long_dbl ( hypre_ParCSRMatrix  *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGInterpTruncation_flt  ( hypre_ParCSRMatrix *P, hypre_float trunc_factor,
                                            HYPRE_Int max_elmts );
HYPRE_Int hypre_BoomerAMGInterpTruncation_dbl  ( hypre_ParCSRMatrix *P, hypre_double trunc_factor,
                                            HYPRE_Int max_elmts );
HYPRE_Int hypre_BoomerAMGInterpTruncation_long_dbl  ( hypre_ParCSRMatrix *P, hypre_long_double trunc_factor,
                                            HYPRE_Int max_elmts );
HYPRE_Int hypre_BoomerAMGTruncandBuild_flt  ( hypre_ParCSRMatrix *P, hypre_float trunc_factor,
                                         HYPRE_Int max_elmts );
HYPRE_Int hypre_BoomerAMGTruncandBuild_dbl  ( hypre_ParCSRMatrix *P, hypre_double trunc_factor,
                                         HYPRE_Int max_elmts );
HYPRE_Int hypre_BoomerAMGTruncandBuild_long_dbl  ( hypre_ParCSRMatrix *P, hypre_long_double trunc_factor,
                                         HYPRE_Int max_elmts );
hypre_ParCSRMatrix *hypre_CreateC_flt  ( hypre_ParCSRMatrix *A, hypre_float w );
hypre_ParCSRMatrix *hypre_CreateC_dbl  ( hypre_ParCSRMatrix *A, hypre_double w );
hypre_ParCSRMatrix *hypre_CreateC_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double w );
void hypre_BoomerAMGJacobiInterp_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                   hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                   HYPRE_Int level, hypre_float truncation_threshold, hypre_float truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                   hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                   HYPRE_Int level, hypre_double truncation_threshold, hypre_double truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                   hypre_ParCSRMatrix *S, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                   HYPRE_Int level, hypre_long_double truncation_threshold, hypre_long_double truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_1_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                     hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker, HYPRE_Int level, hypre_float truncation_threshold,
                                     hypre_float truncation_threshold_minus, HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd,
                                     hypre_float weight_AF );
void hypre_BoomerAMGJacobiInterp_1_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                     hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker, HYPRE_Int level, hypre_double truncation_threshold,
                                     hypre_double truncation_threshold_minus, HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd,
                                     hypre_double weight_AF );
void hypre_BoomerAMGJacobiInterp_1_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                     hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker, HYPRE_Int level, hypre_long_double truncation_threshold,
                                     hypre_long_double truncation_threshold_minus, HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd,
                                     hypre_long_double weight_AF );
void hypre_BoomerAMGTruncateInterp_flt  ( hypre_ParCSRMatrix *P, hypre_float eps, hypre_float dlt,
                                     HYPRE_Int *CF_marker );
void hypre_BoomerAMGTruncateInterp_dbl  ( hypre_ParCSRMatrix *P, hypre_double eps, hypre_double dlt,
                                     HYPRE_Int *CF_marker );
void hypre_BoomerAMGTruncateInterp_long_dbl  ( hypre_ParCSRMatrix *P, hypre_long_double eps, hypre_long_double dlt,
                                     HYPRE_Int *CF_marker );
HYPRE_Int hypre_ParCSRMatrix_dof_func_offd_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                             HYPRE_Int *dof_func, HYPRE_Int **dof_func_offd );
HYPRE_Int hypre_ParCSRMatrix_dof_func_offd_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                             HYPRE_Int *dof_func, HYPRE_Int **dof_func_offd );
HYPRE_Int hypre_ParCSRMatrix_dof_func_offd_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                             HYPRE_Int *dof_func, HYPRE_Int **dof_func_offd );
HYPRE_Int hypre_ParKrylovAxpy_flt  ( hypre_float alpha, void *x, void *y );
HYPRE_Int hypre_ParKrylovAxpy_dbl  ( hypre_double alpha, void *x, void *y );
HYPRE_Int hypre_ParKrylovAxpy_long_dbl  ( hypre_long_double alpha, void *x, void *y );
void *hypre_ParKrylovCAlloc_flt  ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
void *hypre_ParKrylovCAlloc_dbl  ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
void *hypre_ParKrylovCAlloc_long_dbl  ( size_t count, size_t elt_size, HYPRE_MemoryLocation location );
HYPRE_Int hypre_ParKrylovClearVector_flt  ( void *x );
HYPRE_Int hypre_ParKrylovClearVector_dbl  ( void *x );
HYPRE_Int hypre_ParKrylovClearVector_long_dbl  ( void *x );
HYPRE_Int hypre_ParKrylovCommInfo_flt  ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
HYPRE_Int hypre_ParKrylovCommInfo_dbl  ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
HYPRE_Int hypre_ParKrylovCommInfo_long_dbl  ( void *A, HYPRE_Int *my_id, HYPRE_Int *num_procs );
HYPRE_Int hypre_ParKrylovCopyVector_flt  ( void *x, void *y );
HYPRE_Int hypre_ParKrylovCopyVector_dbl  ( void *x, void *y );
HYPRE_Int hypre_ParKrylovCopyVector_long_dbl  ( void *x, void *y );
void *hypre_ParKrylovCreateVector_flt  ( void *vvector );
void *hypre_ParKrylovCreateVector_dbl  ( void *vvector );
void *hypre_ParKrylovCreateVector_long_dbl  ( void *vvector );
void *hypre_ParKrylovCreateVectorArray_flt  ( HYPRE_Int n, void *vvector );
void *hypre_ParKrylovCreateVectorArray_dbl  ( HYPRE_Int n, void *vvector );
void *hypre_ParKrylovCreateVectorArray_long_dbl  ( HYPRE_Int n, void *vvector );
HYPRE_Int hypre_ParKrylovDestroyVector_flt  ( void *vvector );
HYPRE_Int hypre_ParKrylovDestroyVector_dbl  ( void *vvector );
HYPRE_Int hypre_ParKrylovDestroyVector_long_dbl  ( void *vvector );
HYPRE_Int hypre_ParKrylovFree_flt  ( void *ptr );
HYPRE_Int hypre_ParKrylovFree_dbl  ( void *ptr );
HYPRE_Int hypre_ParKrylovFree_long_dbl  ( void *ptr );
HYPRE_Int hypre_ParKrylovIdentity_flt  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_ParKrylovIdentity_dbl  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_ParKrylovIdentity_long_dbl  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_ParKrylovIdentitySetup_flt  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_ParKrylovIdentitySetup_dbl  ( void *vdata, void *A, void *b, void *x );
HYPRE_Int hypre_ParKrylovIdentitySetup_long_dbl  ( void *vdata, void *A, void *b, void *x );
hypre_float hypre_ParKrylovInnerProd_flt  ( void *x, void *y );
hypre_double hypre_ParKrylovInnerProd_dbl  ( void *x, void *y );
hypre_long_double hypre_ParKrylovInnerProd_long_dbl  ( void *x, void *y );
HYPRE_Int hypre_ParKrylovMassAxpy_flt ( hypre_float *alpha, void **x, void *y, HYPRE_Int k,
                                   HYPRE_Int unroll);
HYPRE_Int hypre_ParKrylovMassAxpy_dbl ( hypre_double *alpha, void **x, void *y, HYPRE_Int k,
                                   HYPRE_Int unroll);
HYPRE_Int hypre_ParKrylovMassAxpy_long_dbl ( hypre_long_double *alpha, void **x, void *y, HYPRE_Int k,
                                   HYPRE_Int unroll);
HYPRE_Int hypre_ParKrylovMassDotpTwo_flt  ( void *x, void *y, void **z, HYPRE_Int k, HYPRE_Int unroll,
                                       void *result_x, void *result_y );
HYPRE_Int hypre_ParKrylovMassDotpTwo_dbl  ( void *x, void *y, void **z, HYPRE_Int k, HYPRE_Int unroll,
                                       void *result_x, void *result_y );
HYPRE_Int hypre_ParKrylovMassDotpTwo_long_dbl  ( void *x, void *y, void **z, HYPRE_Int k, HYPRE_Int unroll,
                                       void *result_x, void *result_y );
HYPRE_Int hypre_ParKrylovMassInnerProd_flt  ( void *x, void **y, HYPRE_Int k, HYPRE_Int unroll,
                                         void *result );
HYPRE_Int hypre_ParKrylovMassInnerProd_dbl  ( void *x, void **y, HYPRE_Int k, HYPRE_Int unroll,
                                         void *result );
HYPRE_Int hypre_ParKrylovMassInnerProd_long_dbl  ( void *x, void **y, HYPRE_Int k, HYPRE_Int unroll,
                                         void *result );
HYPRE_Int hypre_ParKrylovMatvec_flt  ( void *matvec_data, hypre_float alpha, void *A, void *x,
                                  hypre_float beta, void *y );
HYPRE_Int hypre_ParKrylovMatvec_dbl  ( void *matvec_data, hypre_double alpha, void *A, void *x,
                                  hypre_double beta, void *y );
HYPRE_Int hypre_ParKrylovMatvec_long_dbl  ( void *matvec_data, hypre_long_double alpha, void *A, void *x,
                                  hypre_long_double beta, void *y );
void *hypre_ParKrylovMatvecCreate_flt  ( void *A, void *x );
void *hypre_ParKrylovMatvecCreate_dbl  ( void *A, void *x );
void *hypre_ParKrylovMatvecCreate_long_dbl  ( void *A, void *x );
HYPRE_Int hypre_ParKrylovMatvecDestroy_flt  ( void *matvec_data );
HYPRE_Int hypre_ParKrylovMatvecDestroy_dbl  ( void *matvec_data );
HYPRE_Int hypre_ParKrylovMatvecDestroy_long_dbl  ( void *matvec_data );
HYPRE_Int hypre_ParKrylovMatvecT_flt  ( void *matvec_data, hypre_float alpha, void *A, void *x,
                                   hypre_float beta, void *y );
HYPRE_Int hypre_ParKrylovMatvecT_dbl  ( void *matvec_data, hypre_double alpha, void *A, void *x,
                                   hypre_double beta, void *y );
HYPRE_Int hypre_ParKrylovMatvecT_long_dbl  ( void *matvec_data, hypre_long_double alpha, void *A, void *x,
                                   hypre_long_double beta, void *y );
HYPRE_Int hypre_ParKrylovScaleVector_flt  ( hypre_float alpha, void *x );
HYPRE_Int hypre_ParKrylovScaleVector_dbl  ( hypre_double alpha, void *x );
HYPRE_Int hypre_ParKrylovScaleVector_long_dbl  ( hypre_long_double alpha, void *x );
HYPRE_ParCSRMatrix GenerateLaplacian27pt_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                           HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                           hypre_float *value );
HYPRE_ParCSRMatrix GenerateLaplacian27pt_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                           HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                           hypre_double *value );
HYPRE_ParCSRMatrix GenerateLaplacian27pt_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                           HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                           hypre_long_double *value );
HYPRE_ParCSRMatrix GenerateLaplacian9pt_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_Int P, HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, hypre_float *value );
HYPRE_ParCSRMatrix GenerateLaplacian9pt_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_Int P, HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, hypre_double *value );
HYPRE_ParCSRMatrix GenerateLaplacian9pt_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_Int P, HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, hypre_long_double *value );
HYPRE_BigInt hypre_map2_flt  ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_Int p, HYPRE_Int q,
                          HYPRE_BigInt nx, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part );
HYPRE_BigInt hypre_map2_dbl  ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_Int p, HYPRE_Int q,
                          HYPRE_BigInt nx, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part );
HYPRE_BigInt hypre_map2_long_dbl  ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_Int p, HYPRE_Int q,
                          HYPRE_BigInt nx, HYPRE_BigInt *nx_part, HYPRE_BigInt *ny_part );
HYPRE_ParCSRMatrix GenerateLaplacian_flt  ( MPI_Comm comm, HYPRE_BigInt ix, HYPRE_BigInt ny,
                                       HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                       hypre_float *value );
HYPRE_ParCSRMatrix GenerateLaplacian_dbl  ( MPI_Comm comm, HYPRE_BigInt ix, HYPRE_BigInt ny,
                                       HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                       hypre_double *value );
HYPRE_ParCSRMatrix GenerateLaplacian_long_dbl  ( MPI_Comm comm, HYPRE_BigInt ix, HYPRE_BigInt ny,
                                       HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                       hypre_long_double *value );
HYPRE_ParCSRMatrix GenerateSysLaplacian_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          HYPRE_Int num_fun, hypre_float *mtrx, hypre_float *value );
HYPRE_ParCSRMatrix GenerateSysLaplacian_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          HYPRE_Int num_fun, hypre_double *mtrx, hypre_double *value );
HYPRE_ParCSRMatrix GenerateSysLaplacian_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          HYPRE_Int num_fun, hypre_long_double *mtrx, hypre_long_double *value );
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                               HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                               HYPRE_Int num_fun, hypre_float *mtrx, hypre_float *value );
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                               HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                               HYPRE_Int num_fun, hypre_double *mtrx, hypre_double *value );
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                               HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                               HYPRE_Int num_fun, hypre_long_double *mtrx, hypre_long_double *value );
HYPRE_BigInt hypre_map_flt  ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_BigInt iz, HYPRE_Int p,
                         HYPRE_Int q, HYPRE_Int r, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt *nx_part,
                         HYPRE_BigInt *ny_part, HYPRE_BigInt *nz_part );
HYPRE_BigInt hypre_map_dbl  ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_BigInt iz, HYPRE_Int p,
                         HYPRE_Int q, HYPRE_Int r, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt *nx_part,
                         HYPRE_BigInt *ny_part, HYPRE_BigInt *nz_part );
HYPRE_BigInt hypre_map_long_dbl  ( HYPRE_BigInt ix, HYPRE_BigInt iy, HYPRE_BigInt iz, HYPRE_Int p,
                         HYPRE_Int q, HYPRE_Int r, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt *nx_part,
                         HYPRE_BigInt *ny_part, HYPRE_BigInt *nz_part );
HYPRE_Int hypre_BoomerAMGBuildExtInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_float trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_double trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPICCInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                              HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, hypre_float trunc_factor,
                                              HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPICCInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                              HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, hypre_double trunc_factor,
                                              HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPICCInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                              HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                              HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                            HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, hypre_float trunc_factor,
                                            HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                            HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, hypre_double trunc_factor,
                                            HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                            hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                            HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                            HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                            HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterpHost_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, hypre_float trunc_factor,
                                                HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterpHost_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, hypre_double trunc_factor,
                                                HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildExtPIInterpHost_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                                HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                                HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFF1Interp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_float trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFF1Interp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_double trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFF1Interp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                          HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFFInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                         HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_float trunc_factor,
                                         HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFFInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                         HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_double trunc_factor,
                                         HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildFFInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                         hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                         HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                         HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                         HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildStdInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_float trunc_factor,
                                          HYPRE_Int max_elmts, HYPRE_Int sep_weight,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildStdInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_double trunc_factor,
                                          HYPRE_Int max_elmts, HYPRE_Int sep_weight,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildStdInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                          HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                          HYPRE_Int max_elmts, HYPRE_Int sep_weight,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildRestrDist2AIR_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             hypre_float filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr, HYPRE_Int AIR1_5,
                                             HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_Int hypre_BoomerAMGBuildRestrDist2AIR_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             hypre_double filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr, HYPRE_Int AIR1_5,
                                             HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_Int hypre_BoomerAMGBuildRestrDist2AIR_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             hypre_long_double filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr, HYPRE_Int AIR1_5,
                                             HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIR_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                               HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int NeumannDeg,
                                               hypre_float strong_thresholdR, hypre_float filter_thresholdR, HYPRE_Int debug_flag,
                                               hypre_ParCSRMatrix **R_ptr);
HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIR_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                               HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int NeumannDeg,
                                               hypre_double strong_thresholdR, hypre_double filter_thresholdR, HYPRE_Int debug_flag,
                                               hypre_ParCSRMatrix **R_ptr);
HYPRE_Int hypre_BoomerAMGBuildRestrNeumannAIR_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                               HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int NeumannDeg,
                                               hypre_long_double strong_thresholdR, hypre_long_double filter_thresholdR, HYPRE_Int debug_flag,
                                               hypre_ParCSRMatrix **R_ptr);
HYPRE_Int hypre_MGRCoarseParms_flt ( MPI_Comm comm, HYPRE_Int num_rows, hypre_IntArray *CF_marker,
                                HYPRE_BigInt *row_starts_cpts, HYPRE_BigInt *row_starts_fpts );
HYPRE_Int hypre_MGRCoarseParms_dbl ( MPI_Comm comm, HYPRE_Int num_rows, hypre_IntArray *CF_marker,
                                HYPRE_BigInt *row_starts_cpts, HYPRE_BigInt *row_starts_fpts );
HYPRE_Int hypre_MGRCoarseParms_long_dbl ( MPI_Comm comm, HYPRE_Int num_rows, hypre_IntArray *CF_marker,
                                HYPRE_BigInt *row_starts_cpts, HYPRE_BigInt *row_starts_fpts );
HYPRE_Int hypre_block_jacobi_solve_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int method, hypre_float *diaginv,
                                    hypre_ParVector *Vtemp );
HYPRE_Int hypre_block_jacobi_solve_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int method, hypre_double *diaginv,
                                    hypre_ParVector *Vtemp );
HYPRE_Int hypre_block_jacobi_solve_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int method, hypre_long_double *diaginv,
                                    hypre_ParVector *Vtemp );
HYPRE_Int hypre_MGRAddVectorP_flt ( hypre_IntArray *CF_marker, HYPRE_Int point_type, hypre_float a,
                               hypre_ParVector *fromVector, hypre_float b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRAddVectorP_dbl ( hypre_IntArray *CF_marker, HYPRE_Int point_type, hypre_double a,
                               hypre_ParVector *fromVector, hypre_double b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRAddVectorP_long_dbl ( hypre_IntArray *CF_marker, HYPRE_Int point_type, hypre_long_double a,
                               hypre_ParVector *fromVector, hypre_long_double b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRAddVectorR_flt ( hypre_IntArray *CF_marker, HYPRE_Int point_type, hypre_float a,
                               hypre_ParVector *fromVector, hypre_float b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRAddVectorR_dbl ( hypre_IntArray *CF_marker, HYPRE_Int point_type, hypre_double a,
                               hypre_ParVector *fromVector, hypre_double b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRAddVectorR_long_dbl ( hypre_IntArray *CF_marker, HYPRE_Int point_type, hypre_long_double a,
                               hypre_ParVector *fromVector, hypre_long_double b,
                               hypre_ParVector **toVector );
HYPRE_Int hypre_MGRApproximateInverse_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **A_inv );
HYPRE_Int hypre_MGRApproximateInverse_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **A_inv );
HYPRE_Int hypre_MGRApproximateInverse_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **A_inv );
HYPRE_Int hypre_MGRBlockRelaxSetup_flt ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                    hypre_float **diaginvptr );
HYPRE_Int hypre_MGRBlockRelaxSetup_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                    hypre_double **diaginvptr );
HYPRE_Int hypre_MGRBlockRelaxSetup_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                    hypre_long_double **diaginvptr );
HYPRE_Int hypre_MGRBlockRelaxSolve_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int n_block, HYPRE_Int left_size,
                                    HYPRE_Int method, hypre_float *diaginv,
                                    hypre_ParVector *Vtemp );
HYPRE_Int hypre_MGRBlockRelaxSolve_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int n_block, HYPRE_Int left_size,
                                    HYPRE_Int method, hypre_double *diaginv,
                                    hypre_ParVector *Vtemp );
HYPRE_Int hypre_MGRBlockRelaxSolve_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                    hypre_ParVector *u, HYPRE_Int blk_size,
                                    HYPRE_Int n_block, HYPRE_Int left_size,
                                    HYPRE_Int method, hypre_long_double *diaginv,
                                    hypre_ParVector *Vtemp );
HYPRE_Int hypre_MGRBlockRelaxSolveDevice_flt ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          hypre_ParVector *Vtemp, hypre_float relax_weight );
HYPRE_Int hypre_MGRBlockRelaxSolveDevice_dbl ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          hypre_ParVector *Vtemp, hypre_double relax_weight );
HYPRE_Int hypre_MGRBlockRelaxSolveDevice_long_dbl ( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          hypre_ParVector *Vtemp, hypre_long_double relax_weight );
HYPRE_Int hypre_MGRBuildAff_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                             hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRBuildAff_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                             hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRBuildAff_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                             hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRBuildBlockJacobiWp_flt ( hypre_ParCSRMatrix *A_FF, hypre_ParCSRMatrix *A_FC,
                                       HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                       HYPRE_BigInt *cpts_starts_in,
                                       hypre_ParCSRMatrix **Wp_ptr );
HYPRE_Int hypre_MGRBuildBlockJacobiWp_dbl ( hypre_ParCSRMatrix *A_FF, hypre_ParCSRMatrix *A_FC,
                                       HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                       HYPRE_BigInt *cpts_starts_in,
                                       hypre_ParCSRMatrix **Wp_ptr );
HYPRE_Int hypre_MGRBuildBlockJacobiWp_long_dbl ( hypre_ParCSRMatrix *A_FF, hypre_ParCSRMatrix *A_FC,
                                       HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                       HYPRE_BigInt *cpts_starts_in,
                                       hypre_ParCSRMatrix **Wp_ptr );
HYPRE_Int hypre_MGRBuildInterp_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                HYPRE_Int debug_flag, hypre_float trunc_factor,
                                HYPRE_Int max_elmts, HYPRE_Int block_jacobi_bsize,
                                hypre_ParCSRMatrix  **P, HYPRE_Int method,
                                HYPRE_Int numsweeps );
HYPRE_Int hypre_MGRBuildInterp_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                HYPRE_Int debug_flag, hypre_double trunc_factor,
                                HYPRE_Int max_elmts, HYPRE_Int block_jacobi_bsize,
                                hypre_ParCSRMatrix  **P, HYPRE_Int method,
                                HYPRE_Int numsweeps );
HYPRE_Int hypre_MGRBuildInterp_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                                hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global,
                                HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                HYPRE_Int max_elmts, HYPRE_Int block_jacobi_bsize,
                                hypre_ParCSRMatrix  **P, HYPRE_Int method,
                                HYPRE_Int numsweeps );
HYPRE_Int hypre_MGRBuildP_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                           HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                           HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildP_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                           HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                           HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildP_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                           HYPRE_BigInt *num_cpts_global, HYPRE_Int method,
                           HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPBlockJacobi_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                      hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *Wp,
                                      HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                      HYPRE_BigInt *cpts_starts, HYPRE_Int debug_flag,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPBlockJacobi_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                      hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *Wp,
                                      HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                      HYPRE_BigInt *cpts_starts, HYPRE_Int debug_flag,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPBlockJacobi_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                      hypre_ParCSRMatrix *A_FC, hypre_ParCSRMatrix *Wp,
                                      HYPRE_Int blk_size, HYPRE_Int *CF_marker,
                                      HYPRE_BigInt *cpts_starts, HYPRE_Int debug_flag,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPFromWp_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                 HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPFromWp_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                 HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPFromWp_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                 HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPFromWpHost_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                     HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                                     hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPFromWpHost_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                     HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                                     hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildPFromWpHost_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *Wp,
                                     HYPRE_Int *CF_marker, HYPRE_Int debug_flag,
                                     hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_MGRBuildRestrict_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                  hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                                  HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions,
                                  HYPRE_Int *dof_func, HYPRE_Int debug_flag,
                                  hypre_float trunc_factor, HYPRE_Int max_elmts,
                                  hypre_float strong_threshold, hypre_float max_row_sum,
                                  HYPRE_Int blk_size, hypre_ParCSRMatrix **RT,
                                  HYPRE_Int method, HYPRE_Int numsweeps );
HYPRE_Int hypre_MGRBuildRestrict_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                  hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                                  HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions,
                                  HYPRE_Int *dof_func, HYPRE_Int debug_flag,
                                  hypre_double trunc_factor, HYPRE_Int max_elmts,
                                  hypre_double strong_threshold, hypre_double max_row_sum,
                                  HYPRE_Int blk_size, hypre_ParCSRMatrix **RT,
                                  HYPRE_Int method, HYPRE_Int numsweeps );
HYPRE_Int hypre_MGRBuildRestrict_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *A_FF,
                                  hypre_ParCSRMatrix *A_FC, HYPRE_Int *CF_marker,
                                  HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions,
                                  HYPRE_Int *dof_func, HYPRE_Int debug_flag,
                                  hypre_long_double trunc_factor, HYPRE_Int max_elmts,
                                  hypre_long_double strong_threshold, hypre_long_double max_row_sum,
                                  HYPRE_Int blk_size, hypre_ParCSRMatrix **RT,
                                  HYPRE_Int method, HYPRE_Int numsweeps );
HYPRE_Int hypre_MGRCoarsen_flt ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                            HYPRE_Int final_coarse_size, HYPRE_Int *final_coarse_indexes,
                            HYPRE_Int debug_flag, hypre_IntArray **CF_marker,
                            HYPRE_Int last_level );
HYPRE_Int hypre_MGRCoarsen_dbl ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                            HYPRE_Int final_coarse_size, HYPRE_Int *final_coarse_indexes,
                            HYPRE_Int debug_flag, hypre_IntArray **CF_marker,
                            HYPRE_Int last_level );
HYPRE_Int hypre_MGRCoarsen_long_dbl ( hypre_ParCSRMatrix *S, hypre_ParCSRMatrix *A,
                            HYPRE_Int final_coarse_size, HYPRE_Int *final_coarse_indexes,
                            HYPRE_Int debug_flag, hypre_IntArray **CF_marker,
                            HYPRE_Int last_level );
HYPRE_Int hypre_MGRComputeNonGalerkinCoarseGrid_flt ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                                 hypre_ParCSRMatrix *RT, HYPRE_Int bsize,
                                                 HYPRE_Int ordering, HYPRE_Int method,
                                                 HYPRE_Int Pmax, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix **A_H_ptr );
HYPRE_Int hypre_MGRComputeNonGalerkinCoarseGrid_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                                 hypre_ParCSRMatrix *RT, HYPRE_Int bsize,
                                                 HYPRE_Int ordering, HYPRE_Int method,
                                                 HYPRE_Int Pmax, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix **A_H_ptr );
HYPRE_Int hypre_MGRComputeNonGalerkinCoarseGrid_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                                 hypre_ParCSRMatrix *RT, HYPRE_Int bsize,
                                                 HYPRE_Int ordering, HYPRE_Int method,
                                                 HYPRE_Int Pmax, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix **A_H_ptr );
void *hypre_MGRCreate_flt  ( void );
void *hypre_MGRCreate_dbl  ( void );
void *hypre_MGRCreate_long_dbl  ( void );
void *hypre_MGRCreateFrelaxVcycleData_flt ( void );
void *hypre_MGRCreateFrelaxVcycleData_dbl ( void );
void *hypre_MGRCreateFrelaxVcycleData_long_dbl ( void );
void *hypre_MGRCreateGSElimData_flt ( void );
void *hypre_MGRCreateGSElimData_dbl ( void );
void *hypre_MGRCreateGSElimData_long_dbl ( void );
HYPRE_Int hypre_MGRDestroy_flt  ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroy_dbl  ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroy_long_dbl  ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroyFrelaxVcycleData_flt ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroyFrelaxVcycleData_dbl ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroyFrelaxVcycleData_long_dbl ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroyGSElimData_flt ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroyGSElimData_dbl ( void *mgr_vdata );
HYPRE_Int hypre_MGRDestroyGSElimData_long_dbl ( void *mgr_vdata );
HYPRE_Int hypre_MGRGetCoarseGridConvergenceFactor_flt ( void *mgr_data, hypre_float *conv_factor );
HYPRE_Int hypre_MGRGetCoarseGridConvergenceFactor_dbl ( void *mgr_data, hypre_double *conv_factor );
HYPRE_Int hypre_MGRGetCoarseGridConvergenceFactor_long_dbl ( void *mgr_data, hypre_long_double *conv_factor );
HYPRE_Int hypre_MGRGetFinalRelativeResidualNorm_flt ( void *mgr_vdata, hypre_float *res_norm );
HYPRE_Int hypre_MGRGetFinalRelativeResidualNorm_dbl ( void *mgr_vdata, hypre_double *res_norm );
HYPRE_Int hypre_MGRGetFinalRelativeResidualNorm_long_dbl ( void *mgr_vdata, hypre_long_double *res_norm );
HYPRE_Int hypre_MGRGetNumIterations_flt ( void *mgr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_MGRGetNumIterations_dbl ( void *mgr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_MGRGetNumIterations_long_dbl ( void *mgr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_MGRGetSubBlock_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *row_cf_marker,
                                HYPRE_Int *col_cf_marker, HYPRE_Int debug_flag,
                                hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRGetSubBlock_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *row_cf_marker,
                                HYPRE_Int *col_cf_marker, HYPRE_Int debug_flag,
                                hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRGetSubBlock_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *row_cf_marker,
                                HYPRE_Int *col_cf_marker, HYPRE_Int debug_flag,
                                hypre_ParCSRMatrix **A_ff_ptr );
HYPRE_Int hypre_MGRSetBlockJacobiBlockSize_flt ( void *mgr_vdata, HYPRE_Int blk_size );
HYPRE_Int hypre_MGRSetBlockJacobiBlockSize_dbl ( void *mgr_vdata, HYPRE_Int blk_size );
HYPRE_Int hypre_MGRSetBlockJacobiBlockSize_long_dbl ( void *mgr_vdata, HYPRE_Int blk_size );
HYPRE_Int hypre_MGRSetBlockSize_flt ( void *mgr_vdata, HYPRE_Int bsize );
HYPRE_Int hypre_MGRSetBlockSize_dbl ( void *mgr_vdata, HYPRE_Int bsize );
HYPRE_Int hypre_MGRSetBlockSize_long_dbl ( void *mgr_vdata, HYPRE_Int bsize );
HYPRE_Int hypre_MGRSetCoarseGridMethod_flt ( void *mgr_vdata, HYPRE_Int *cg_method );
HYPRE_Int hypre_MGRSetCoarseGridMethod_dbl ( void *mgr_vdata, HYPRE_Int *cg_method );
HYPRE_Int hypre_MGRSetCoarseGridMethod_long_dbl ( void *mgr_vdata, HYPRE_Int *cg_method );
HYPRE_Int hypre_MGRSetCoarseGridPrintLevel_flt ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetCoarseGridPrintLevel_dbl ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetCoarseGridPrintLevel_long_dbl ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetCoarseSolver_flt ( void *mgr_vdata,
                                    HYPRE_Int (*cgrid_solver_solve)(void*, void*, void*, void*),
                                    HYPRE_Int (*cgrid_solver_setup)(void*, void*, void*, void*),
                                    void *coarse_grid_solver );
HYPRE_Int hypre_MGRSetCoarseSolver_dbl ( void *mgr_vdata,
                                    HYPRE_Int (*cgrid_solver_solve)(void*, void*, void*, void*),
                                    HYPRE_Int (*cgrid_solver_setup)(void*, void*, void*, void*),
                                    void *coarse_grid_solver );
HYPRE_Int hypre_MGRSetCoarseSolver_long_dbl ( void *mgr_vdata,
                                    HYPRE_Int (*cgrid_solver_solve)(void*, void*, void*, void*),
                                    HYPRE_Int (*cgrid_solver_setup)(void*, void*, void*, void*),
                                    void *coarse_grid_solver );
HYPRE_Int hypre_MGRSetCpointsByBlock_flt ( void *mgr_vdata, HYPRE_Int  block_size,
                                      HYPRE_Int  max_num_levels,
                                      HYPRE_Int *block_num_coarse_points,
                                      HYPRE_Int  **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByBlock_dbl ( void *mgr_vdata, HYPRE_Int  block_size,
                                      HYPRE_Int  max_num_levels,
                                      HYPRE_Int *block_num_coarse_points,
                                      HYPRE_Int  **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByBlock_long_dbl ( void *mgr_vdata, HYPRE_Int  block_size,
                                      HYPRE_Int  max_num_levels,
                                      HYPRE_Int *block_num_coarse_points,
                                      HYPRE_Int  **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByContiguousBlock_flt ( void *mgr_vdata, HYPRE_Int block_size,
                                                HYPRE_Int  max_num_levels,
                                                HYPRE_BigInt *begin_idx_array,
                                                HYPRE_Int *block_num_coarse_points,
                                                HYPRE_Int **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByContiguousBlock_dbl ( void *mgr_vdata, HYPRE_Int block_size,
                                                HYPRE_Int  max_num_levels,
                                                HYPRE_BigInt *begin_idx_array,
                                                HYPRE_Int *block_num_coarse_points,
                                                HYPRE_Int **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByContiguousBlock_long_dbl ( void *mgr_vdata, HYPRE_Int block_size,
                                                HYPRE_Int  max_num_levels,
                                                HYPRE_BigInt *begin_idx_array,
                                                HYPRE_Int *block_num_coarse_points,
                                                HYPRE_Int **block_coarse_indexes );
HYPRE_Int hypre_MGRSetCpointsByPointMarkerArray_flt ( void *mgr_vdata, HYPRE_Int block_size,
                                                 HYPRE_Int  max_num_levels,
                                                 HYPRE_Int *block_num_coarse_points,
                                                 HYPRE_Int **block_coarse_indexes,
                                                 HYPRE_Int *point_marker_array );
HYPRE_Int hypre_MGRSetCpointsByPointMarkerArray_dbl ( void *mgr_vdata, HYPRE_Int block_size,
                                                 HYPRE_Int  max_num_levels,
                                                 HYPRE_Int *block_num_coarse_points,
                                                 HYPRE_Int **block_coarse_indexes,
                                                 HYPRE_Int *point_marker_array );
HYPRE_Int hypre_MGRSetCpointsByPointMarkerArray_long_dbl ( void *mgr_vdata, HYPRE_Int block_size,
                                                 HYPRE_Int  max_num_levels,
                                                 HYPRE_Int *block_num_coarse_points,
                                                 HYPRE_Int **block_coarse_indexes,
                                                 HYPRE_Int *point_marker_array );
HYPRE_Int hypre_MGRSetFRelaxMethod_flt ( void *mgr_vdata, HYPRE_Int relax_method );
HYPRE_Int hypre_MGRSetFRelaxMethod_dbl ( void *mgr_vdata, HYPRE_Int relax_method );
HYPRE_Int hypre_MGRSetFRelaxMethod_long_dbl ( void *mgr_vdata, HYPRE_Int relax_method );
HYPRE_Int hypre_MGRSetFrelaxPrintLevel_flt ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetFrelaxPrintLevel_dbl ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetFrelaxPrintLevel_long_dbl ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetFSolver_flt ( void *mgr_vdata,
                               HYPRE_Int (*fine_grid_solver_solve)(void*, void*, void*, void*),
                               HYPRE_Int (*fine_grid_solver_setup)(void*, void*, void*, void*),
                               void *fsolver );
HYPRE_Int hypre_MGRSetFSolver_dbl ( void *mgr_vdata,
                               HYPRE_Int (*fine_grid_solver_solve)(void*, void*, void*, void*),
                               HYPRE_Int (*fine_grid_solver_setup)(void*, void*, void*, void*),
                               void *fsolver );
HYPRE_Int hypre_MGRSetFSolver_long_dbl ( void *mgr_vdata,
                               HYPRE_Int (*fine_grid_solver_solve)(void*, void*, void*, void*),
                               HYPRE_Int (*fine_grid_solver_setup)(void*, void*, void*, void*),
                               void *fsolver );
HYPRE_Int hypre_MGRSetGlobalSmoothCycle_flt ( void *mgr_vdata, HYPRE_Int global_smooth_cycle );
HYPRE_Int hypre_MGRSetGlobalSmoothCycle_dbl ( void *mgr_vdata, HYPRE_Int global_smooth_cycle );
HYPRE_Int hypre_MGRSetGlobalSmoothCycle_long_dbl ( void *mgr_vdata, HYPRE_Int global_smooth_cycle );
HYPRE_Int hypre_MGRSetGlobalSmoothType_flt ( void *mgr_vdata, HYPRE_Int iter_type );
HYPRE_Int hypre_MGRSetGlobalSmoothType_dbl ( void *mgr_vdata, HYPRE_Int iter_type );
HYPRE_Int hypre_MGRSetGlobalSmoothType_long_dbl ( void *mgr_vdata, HYPRE_Int iter_type );
HYPRE_Int hypre_MGRSetInterpType_flt ( void *mgr_vdata, HYPRE_Int interpType );
HYPRE_Int hypre_MGRSetInterpType_dbl ( void *mgr_vdata, HYPRE_Int interpType );
HYPRE_Int hypre_MGRSetInterpType_long_dbl ( void *mgr_vdata, HYPRE_Int interpType );
HYPRE_Int hypre_MGRSetLevelFRelaxMethod_flt ( void *mgr_vdata, HYPRE_Int *relax_method );
HYPRE_Int hypre_MGRSetLevelFRelaxMethod_dbl ( void *mgr_vdata, HYPRE_Int *relax_method );
HYPRE_Int hypre_MGRSetLevelFRelaxMethod_long_dbl ( void *mgr_vdata, HYPRE_Int *relax_method );
HYPRE_Int hypre_MGRSetLevelFRelaxNumFunctions_flt ( void *mgr_vdata, HYPRE_Int *num_functions );
HYPRE_Int hypre_MGRSetLevelFRelaxNumFunctions_dbl ( void *mgr_vdata, HYPRE_Int *num_functions );
HYPRE_Int hypre_MGRSetLevelFRelaxNumFunctions_long_dbl ( void *mgr_vdata, HYPRE_Int *num_functions );
HYPRE_Int hypre_MGRSetLevelFRelaxType_flt ( void *mgr_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_MGRSetLevelFRelaxType_dbl ( void *mgr_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_MGRSetLevelFRelaxType_long_dbl ( void *mgr_vdata, HYPRE_Int *relax_type );
HYPRE_Int hypre_MGRSetLevelInterpType_flt ( void *mgr_vdata, HYPRE_Int *interpType );
HYPRE_Int hypre_MGRSetLevelInterpType_dbl ( void *mgr_vdata, HYPRE_Int *interpType );
HYPRE_Int hypre_MGRSetLevelInterpType_long_dbl ( void *mgr_vdata, HYPRE_Int *interpType );
HYPRE_Int hypre_MGRSetLevelNumRelaxSweeps_flt ( void *mgr_vdata, HYPRE_Int *nsweeps );
HYPRE_Int hypre_MGRSetLevelNumRelaxSweeps_dbl ( void *mgr_vdata, HYPRE_Int *nsweeps );
HYPRE_Int hypre_MGRSetLevelNumRelaxSweeps_long_dbl ( void *mgr_vdata, HYPRE_Int *nsweeps );
HYPRE_Int hypre_MGRSetLevelRestrictType_flt ( void *mgr_vdata, HYPRE_Int *restrictType );
HYPRE_Int hypre_MGRSetLevelRestrictType_dbl ( void *mgr_vdata, HYPRE_Int *restrictType );
HYPRE_Int hypre_MGRSetLevelRestrictType_long_dbl ( void *mgr_vdata, HYPRE_Int *restrictType );
HYPRE_Int hypre_MGRSetLevelSmoothIters_flt ( void *mgr_vdata, HYPRE_Int *level_smooth_iters );
HYPRE_Int hypre_MGRSetLevelSmoothIters_dbl ( void *mgr_vdata, HYPRE_Int *level_smooth_iters );
HYPRE_Int hypre_MGRSetLevelSmoothIters_long_dbl ( void *mgr_vdata, HYPRE_Int *level_smooth_iters );
HYPRE_Int hypre_MGRSetLevelSmoothType_flt ( void *mgr_vdata, HYPRE_Int *level_smooth_type );
HYPRE_Int hypre_MGRSetLevelSmoothType_dbl ( void *mgr_vdata, HYPRE_Int *level_smooth_type );
HYPRE_Int hypre_MGRSetLevelSmoothType_long_dbl ( void *mgr_vdata, HYPRE_Int *level_smooth_type );
HYPRE_Int hypre_MGRSetLogging_flt ( void *mgr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_MGRSetLogging_dbl ( void *mgr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_MGRSetLogging_long_dbl ( void *mgr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_MGRSetMaxCoarseLevels_flt ( void *mgr_vdata, HYPRE_Int maxlev );
HYPRE_Int hypre_MGRSetMaxCoarseLevels_dbl ( void *mgr_vdata, HYPRE_Int maxlev );
HYPRE_Int hypre_MGRSetMaxCoarseLevels_long_dbl ( void *mgr_vdata, HYPRE_Int maxlev );
HYPRE_Int hypre_MGRSetMaxGlobalSmoothIters_flt ( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetMaxGlobalSmoothIters_dbl ( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetMaxGlobalSmoothIters_long_dbl ( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetMaxIter_flt ( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetMaxIter_dbl ( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetMaxIter_long_dbl ( void *mgr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_MGRSetNonCpointsToFpoints_flt ( void *mgr_vdata, HYPRE_Int nonCptToFptFlag );
HYPRE_Int hypre_MGRSetNonCpointsToFpoints_dbl ( void *mgr_vdata, HYPRE_Int nonCptToFptFlag );
HYPRE_Int hypre_MGRSetNonCpointsToFpoints_long_dbl ( void *mgr_vdata, HYPRE_Int nonCptToFptFlag );
HYPRE_Int hypre_MGRSetNumInterpSweeps_flt ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumInterpSweeps_dbl ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumInterpSweeps_long_dbl ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRelaxSweeps_flt ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRelaxSweeps_dbl ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRelaxSweeps_long_dbl ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRestrictSweeps_flt ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRestrictSweeps_dbl ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetNumRestrictSweeps_long_dbl ( void *mgr_vdata, HYPRE_Int nsweeps );
HYPRE_Int hypre_MGRSetPMaxElmts_flt ( void *mgr_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_MGRSetPMaxElmts_dbl ( void *mgr_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_MGRSetPMaxElmts_long_dbl ( void *mgr_vdata, HYPRE_Int P_max_elmts );
HYPRE_Int hypre_MGRSetPrintLevel_flt ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetPrintLevel_dbl ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetPrintLevel_long_dbl ( void *mgr_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_MGRSetRelaxType_flt ( void *mgr_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_MGRSetRelaxType_dbl ( void *mgr_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_MGRSetRelaxType_long_dbl ( void *mgr_vdata, HYPRE_Int relax_type );
HYPRE_Int hypre_MGRSetReservedCoarseNodes_flt ( void *mgr_vdata, HYPRE_Int reserved_coarse_size,
                                           HYPRE_BigInt *reserved_coarse_nodes );
HYPRE_Int hypre_MGRSetReservedCoarseNodes_dbl ( void *mgr_vdata, HYPRE_Int reserved_coarse_size,
                                           HYPRE_BigInt *reserved_coarse_nodes );
HYPRE_Int hypre_MGRSetReservedCoarseNodes_long_dbl ( void *mgr_vdata, HYPRE_Int reserved_coarse_size,
                                           HYPRE_BigInt *reserved_coarse_nodes );
HYPRE_Int hypre_MGRSetReservedCpointsLevelToKeep_flt ( void *mgr_vdata, HYPRE_Int level );
HYPRE_Int hypre_MGRSetReservedCpointsLevelToKeep_dbl ( void *mgr_vdata, HYPRE_Int level );
HYPRE_Int hypre_MGRSetReservedCpointsLevelToKeep_long_dbl ( void *mgr_vdata, HYPRE_Int level );
HYPRE_Int hypre_MGRSetRestrictType_flt ( void *mgr_vdata, HYPRE_Int restrictType );
HYPRE_Int hypre_MGRSetRestrictType_dbl ( void *mgr_vdata, HYPRE_Int restrictType );
HYPRE_Int hypre_MGRSetRestrictType_long_dbl ( void *mgr_vdata, HYPRE_Int restrictType );
HYPRE_Int hypre_MGRSetTol_flt ( void *mgr_vdata, hypre_float tol );
HYPRE_Int hypre_MGRSetTol_dbl ( void *mgr_vdata, hypre_double tol );
HYPRE_Int hypre_MGRSetTol_long_dbl ( void *mgr_vdata, hypre_long_double tol );
HYPRE_Int hypre_MGRSetTruncateCoarseGridThreshold_flt ( void *mgr_vdata, hypre_float threshold );
HYPRE_Int hypre_MGRSetTruncateCoarseGridThreshold_dbl ( void *mgr_vdata, hypre_double threshold );
HYPRE_Int hypre_MGRSetTruncateCoarseGridThreshold_long_dbl ( void *mgr_vdata, hypre_long_double threshold );
HYPRE_Int hypre_MGRTruncateAcfCPR_flt ( hypre_ParCSRMatrix *A_CF, hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRTruncateAcfCPR_dbl ( hypre_ParCSRMatrix *A_CF, hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRTruncateAcfCPR_long_dbl ( hypre_ParCSRMatrix *A_CF, hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRTruncateAcfCPRDevice_flt ( hypre_ParCSRMatrix  *A_CF,
                                         hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRTruncateAcfCPRDevice_dbl ( hypre_ParCSRMatrix  *A_CF,
                                         hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRTruncateAcfCPRDevice_long_dbl ( hypre_ParCSRMatrix  *A_CF,
                                         hypre_ParCSRMatrix **A_CF_new_ptr );
HYPRE_Int hypre_MGRWriteSolverParams_flt ( void *mgr_vdata );
HYPRE_Int hypre_MGRWriteSolverParams_dbl ( void *mgr_vdata );
HYPRE_Int hypre_MGRWriteSolverParams_long_dbl ( void *mgr_vdata );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrix_flt ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                             HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                             HYPRE_Int diag_type, hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrix_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                             HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                             HYPRE_Int diag_type, hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrix_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                             HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                             HYPRE_Int diag_type, hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrixHost_flt ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                                 HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                                 HYPRE_Int diag_type,
                                                 hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrixHost_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                                 HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                                 HYPRE_Int diag_type,
                                                 hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixBlockDiagMatrixHost_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int blk_size,
                                                 HYPRE_Int point_type, HYPRE_Int *CF_marker,
                                                 HYPRE_Int diag_type,
                                                 hypre_ParCSRMatrix **B_ptr );
HYPRE_Int hypre_ParCSRMatrixExtractBlockDiagHost_flt ( hypre_ParCSRMatrix *par_A, HYPRE_Int blk_size,
                                                  HYPRE_Int num_points, HYPRE_Int point_type,
                                                  HYPRE_Int *CF_marker, HYPRE_Int diag_size,
                                                  HYPRE_Int diag_type, hypre_float *diag_data );
HYPRE_Int hypre_ParCSRMatrixExtractBlockDiagHost_dbl ( hypre_ParCSRMatrix *par_A, HYPRE_Int blk_size,
                                                  HYPRE_Int num_points, HYPRE_Int point_type,
                                                  HYPRE_Int *CF_marker, HYPRE_Int diag_size,
                                                  HYPRE_Int diag_type, hypre_double *diag_data );
HYPRE_Int hypre_ParCSRMatrixExtractBlockDiagHost_long_dbl ( hypre_ParCSRMatrix *par_A, HYPRE_Int blk_size,
                                                  HYPRE_Int num_points, HYPRE_Int point_type,
                                                  HYPRE_Int *CF_marker, HYPRE_Int diag_size,
                                                  HYPRE_Int diag_type, hypre_long_double *diag_data );
HYPRE_Int hypre_MGRSetup_flt ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRSetup_dbl ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRSetup_long_dbl ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRSetupFrelaxVcycleData_flt ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          HYPRE_Int level );
HYPRE_Int hypre_MGRSetupFrelaxVcycleData_dbl ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          HYPRE_Int level );
HYPRE_Int hypre_MGRSetupFrelaxVcycleData_long_dbl ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                                          hypre_ParVector *f, hypre_ParVector *u,
                                          HYPRE_Int level );
HYPRE_Int hypre_MGRCycle_flt ( void *mgr_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );
HYPRE_Int hypre_MGRCycle_dbl ( void *mgr_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );
HYPRE_Int hypre_MGRCycle_long_dbl ( void *mgr_vdata, hypre_ParVector **F_array, hypre_ParVector **U_array );
HYPRE_Int hypre_MGRFrelaxVcycle_flt  ( void *mgr_vdata, hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRFrelaxVcycle_dbl  ( void *mgr_vdata, hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRFrelaxVcycle_long_dbl  ( void *mgr_vdata, hypre_ParVector *f, hypre_ParVector *u );
HYPRE_Int hypre_MGRSolve_flt ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector  *u );
HYPRE_Int hypre_MGRSolve_dbl ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector  *u );
HYPRE_Int hypre_MGRSolve_long_dbl ( void *mgr_vdata, hypre_ParCSRMatrix *A,
                          hypre_ParVector *f, hypre_ParVector  *u );
HYPRE_Int hypre_BoomerAMGBuildModExtInterp_flt (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtInterp_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtInterp_long_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                           hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                           HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPEInterp_flt (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPEInterp_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPEInterp_long_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPIInterp_flt (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPIInterp_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModExtPIInterp_long_dbl (hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int max_elmts, hypre_ParCSRMatrix  **P_ptr);
HYPRE_Int hypre_BoomerAMGBuildModMultipass_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, hypre_float trunc_factor,
                                             HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModMultipass_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, hypre_double trunc_factor,
                                             HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModMultipass_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                             hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, hypre_long_double trunc_factor,
                                             HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                             hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModMultipassHost_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, hypre_float trunc_factor,
                                                 HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModMultipassHost_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, hypre_double trunc_factor,
                                                 HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildModMultipassHost_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, hypre_long_double trunc_factor,
                                                 HYPRE_Int P_max_elmts, HYPRE_Int interp_type, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                                 hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultipassPi_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                      HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                      HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, hypre_float *row_sums,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultipassPi_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                      HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                      HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, hypre_double *row_sums,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultipassPi_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                      HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                      HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, hypre_long_double *row_sums,
                                      hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_GenerateMultiPi_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                  hypre_ParCSRMatrix *P, HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                  HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, HYPRE_Int num_functions,
                                  HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_GenerateMultiPi_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                  hypre_ParCSRMatrix *P, HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                  HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, HYPRE_Int num_functions,
                                  HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_GenerateMultiPi_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                  hypre_ParCSRMatrix *P, HYPRE_BigInt *c_pts_starts, HYPRE_Int *pass_order, HYPRE_Int *pass_marker,
                                  HYPRE_Int *pass_marker_offd, HYPRE_Int num_points, HYPRE_Int color, HYPRE_Int num_functions,
                                  HYPRE_Int *dof_func, HYPRE_Int *dof_func_offd, hypre_ParCSRMatrix **Pi_ptr );
HYPRE_Int hypre_BoomerAMGBuildMultipass_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildMultipass_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildMultipass_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                          hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                          HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                          hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildMultipassHost_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, hypre_float trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                              hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildMultipassHost_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, hypre_double trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                              hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildMultipassHost_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                              hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                              HYPRE_Int debug_flag, hypre_long_double trunc_factor, HYPRE_Int P_max_elmts, HYPRE_Int weight_option,
                                              hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGCreateNodalA_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                        HYPRE_Int *dof_func, HYPRE_Int option, HYPRE_Int diag_option, hypre_ParCSRMatrix **AN_ptr );
HYPRE_Int hypre_BoomerAMGCreateNodalA_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                        HYPRE_Int *dof_func, HYPRE_Int option, HYPRE_Int diag_option, hypre_ParCSRMatrix **AN_ptr );
HYPRE_Int hypre_BoomerAMGCreateNodalA_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int num_functions,
                                        HYPRE_Int *dof_func, HYPRE_Int option, HYPRE_Int diag_option, hypre_ParCSRMatrix **AN_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCF_flt  ( HYPRE_Int *CFN_marker, HYPRE_Int num_functions,
                                          HYPRE_Int num_nodes, hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCF_dbl  ( HYPRE_Int *CFN_marker, HYPRE_Int num_functions,
                                          HYPRE_Int num_nodes, hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCF_long_dbl  ( HYPRE_Int *CFN_marker, HYPRE_Int num_functions,
                                          HYPRE_Int num_nodes, hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCFS_flt  ( hypre_ParCSRMatrix *SN, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *CFN_marker, HYPRE_Int num_functions, HYPRE_Int nodal, HYPRE_Int keep_same_sign,
                                           hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCFS_dbl  ( hypre_ParCSRMatrix *SN, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *CFN_marker, HYPRE_Int num_functions, HYPRE_Int nodal, HYPRE_Int keep_same_sign,
                                           hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateScalarCFS_long_dbl  ( hypre_ParCSRMatrix *SN, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *CFN_marker, HYPRE_Int num_functions, HYPRE_Int nodal, HYPRE_Int keep_same_sign,
                                           hypre_IntArray **dof_func_ptr, hypre_IntArray **CF_marker_ptr, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGBuildNonGalerkinCoarseOperator_flt ( hypre_ParCSRMatrix **RAP_ptr,
                                                         hypre_ParCSRMatrix *AP, hypre_float strong_threshold, hypre_float max_row_sum,
                                                         HYPRE_Int num_functions, HYPRE_Int *dof_func_value, HYPRE_Int * CF_marker, hypre_float droptol,
                                                         HYPRE_Int sym_collapse, hypre_float lump_percent, HYPRE_Int collapse_beta );
HYPRE_Int hypre_BoomerAMGBuildNonGalerkinCoarseOperator_dbl ( hypre_ParCSRMatrix **RAP_ptr,
                                                         hypre_ParCSRMatrix *AP, hypre_double strong_threshold, hypre_double max_row_sum,
                                                         HYPRE_Int num_functions, HYPRE_Int *dof_func_value, HYPRE_Int * CF_marker, hypre_double droptol,
                                                         HYPRE_Int sym_collapse, hypre_double lump_percent, HYPRE_Int collapse_beta );
HYPRE_Int hypre_BoomerAMGBuildNonGalerkinCoarseOperator_long_dbl ( hypre_ParCSRMatrix **RAP_ptr,
                                                         hypre_ParCSRMatrix *AP, hypre_long_double strong_threshold, hypre_long_double max_row_sum,
                                                         HYPRE_Int num_functions, HYPRE_Int *dof_func_value, HYPRE_Int * CF_marker, hypre_long_double droptol,
                                                         HYPRE_Int sym_collapse, hypre_long_double lump_percent, HYPRE_Int collapse_beta );
HYPRE_Int hypre_BoomerAMG_MyCreateS_flt  ( hypre_ParCSRMatrix *A, hypre_float strength_threshold,
                                      hypre_float max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMG_MyCreateS_dbl  ( hypre_ParCSRMatrix *A, hypre_double strength_threshold,
                                      hypre_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMG_MyCreateS_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double strength_threshold,
                                      hypre_long_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_GrabSubArray_flt  ( HYPRE_Int *indices, HYPRE_Int start, HYPRE_Int end,
                               HYPRE_BigInt *array, HYPRE_BigInt *output );
HYPRE_Int hypre_GrabSubArray_dbl  ( HYPRE_Int *indices, HYPRE_Int start, HYPRE_Int end,
                               HYPRE_BigInt *array, HYPRE_BigInt *output );
HYPRE_Int hypre_GrabSubArray_long_dbl  ( HYPRE_Int *indices, HYPRE_Int start, HYPRE_Int end,
                               HYPRE_BigInt *array, HYPRE_BigInt *output );
HYPRE_Int hypre_IntersectTwoArrays_flt  ( HYPRE_Int *x, hypre_float *x_data, HYPRE_Int x_length,
                                     HYPRE_Int *y, HYPRE_Int y_length, HYPRE_Int *z, hypre_float *output_x_data,
                                     HYPRE_Int *intersect_length );
HYPRE_Int hypre_IntersectTwoArrays_dbl  ( HYPRE_Int *x, hypre_double *x_data, HYPRE_Int x_length,
                                     HYPRE_Int *y, HYPRE_Int y_length, HYPRE_Int *z, hypre_double *output_x_data,
                                     HYPRE_Int *intersect_length );
HYPRE_Int hypre_IntersectTwoArrays_long_dbl  ( HYPRE_Int *x, hypre_long_double *x_data, HYPRE_Int x_length,
                                     HYPRE_Int *y, HYPRE_Int y_length, HYPRE_Int *z, hypre_long_double *output_x_data,
                                     HYPRE_Int *intersect_length );
HYPRE_Int hypre_IntersectTwoBigArrays_flt  ( HYPRE_BigInt *x, hypre_float *x_data, HYPRE_Int x_length,
                                        HYPRE_BigInt *y, HYPRE_Int y_length, HYPRE_BigInt *z, hypre_float *output_x_data,
                                        HYPRE_Int *intersect_length );
HYPRE_Int hypre_IntersectTwoBigArrays_dbl  ( HYPRE_BigInt *x, hypre_double *x_data, HYPRE_Int x_length,
                                        HYPRE_BigInt *y, HYPRE_Int y_length, HYPRE_BigInt *z, hypre_double *output_x_data,
                                        HYPRE_Int *intersect_length );
HYPRE_Int hypre_IntersectTwoBigArrays_long_dbl  ( HYPRE_BigInt *x, hypre_long_double *x_data, HYPRE_Int x_length,
                                        HYPRE_BigInt *y, HYPRE_Int y_length, HYPRE_BigInt *z, hypre_long_double *output_x_data,
                                        HYPRE_Int *intersect_length );
HYPRE_Int hypre_NonGalerkinIJBigBufferInit_flt  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                             HYPRE_BigInt *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBigBufferInit_dbl  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                             HYPRE_BigInt *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBigBufferInit_long_dbl  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                             HYPRE_BigInt *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompress_flt  ( HYPRE_MemoryLocation memory_location,
                                              HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_cnt,
                                              HYPRE_Int *ijbuf_rowcounter, hypre_float **ijbuf_data, HYPRE_BigInt **ijbuf_cols,
                                              HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompress_dbl  ( HYPRE_MemoryLocation memory_location,
                                              HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_cnt,
                                              HYPRE_Int *ijbuf_rowcounter, hypre_double **ijbuf_data, HYPRE_BigInt **ijbuf_cols,
                                              HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompress_long_dbl  ( HYPRE_MemoryLocation memory_location,
                                              HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_cnt,
                                              HYPRE_Int *ijbuf_rowcounter, hypre_long_double **ijbuf_data, HYPRE_BigInt **ijbuf_cols,
                                              HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompressRow_flt  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter,
                                                 hypre_float *ijbuf_data, HYPRE_BigInt *ijbuf_cols, HYPRE_BigInt *ijbuf_rownums,
                                                 HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompressRow_dbl  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter,
                                                 hypre_double *ijbuf_data, HYPRE_BigInt *ijbuf_cols, HYPRE_BigInt *ijbuf_rownums,
                                                 HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferCompressRow_long_dbl  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter,
                                                 hypre_long_double *ijbuf_data, HYPRE_BigInt *ijbuf_cols, HYPRE_BigInt *ijbuf_rownums,
                                                 HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferEmpty_flt  ( HYPRE_IJMatrix B, HYPRE_Int ijbuf_size,
                                           HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter, hypre_float **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferEmpty_dbl  ( HYPRE_IJMatrix B, HYPRE_Int ijbuf_size,
                                           HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter, hypre_double **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferEmpty_long_dbl  ( HYPRE_IJMatrix B, HYPRE_Int ijbuf_size,
                                           HYPRE_Int *ijbuf_cnt, HYPRE_Int ijbuf_rowcounter, hypre_long_double **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferInit_flt  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                          HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferInit_dbl  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                          HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferInit_long_dbl  ( HYPRE_Int *ijbuf_cnt, HYPRE_Int *ijbuf_rowcounter,
                                          HYPRE_Int *ijbuf_numcols );
HYPRE_Int hypre_NonGalerkinIJBufferNewRow_flt  ( HYPRE_BigInt *ijbuf_rownums, HYPRE_Int *ijbuf_numcols,
                                            HYPRE_Int *ijbuf_rowcounter, HYPRE_BigInt new_row );
HYPRE_Int hypre_NonGalerkinIJBufferNewRow_dbl  ( HYPRE_BigInt *ijbuf_rownums, HYPRE_Int *ijbuf_numcols,
                                            HYPRE_Int *ijbuf_rowcounter, HYPRE_BigInt new_row );
HYPRE_Int hypre_NonGalerkinIJBufferNewRow_long_dbl  ( HYPRE_BigInt *ijbuf_rownums, HYPRE_Int *ijbuf_numcols,
                                            HYPRE_Int *ijbuf_rowcounter, HYPRE_BigInt new_row );
HYPRE_Int hypre_NonGalerkinIJBufferWrite_flt  ( HYPRE_IJMatrix B, HYPRE_Int *ijbuf_cnt,
                                           HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_rowcounter, hypre_float **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols,
                                           HYPRE_BigInt row_to_write, HYPRE_BigInt col_to_write, hypre_float val_to_write );
HYPRE_Int hypre_NonGalerkinIJBufferWrite_dbl  ( HYPRE_IJMatrix B, HYPRE_Int *ijbuf_cnt,
                                           HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_rowcounter, hypre_double **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols,
                                           HYPRE_BigInt row_to_write, HYPRE_BigInt col_to_write, hypre_double val_to_write );
HYPRE_Int hypre_NonGalerkinIJBufferWrite_long_dbl  ( HYPRE_IJMatrix B, HYPRE_Int *ijbuf_cnt,
                                           HYPRE_Int ijbuf_size, HYPRE_Int *ijbuf_rowcounter, hypre_long_double **ijbuf_data,
                                           HYPRE_BigInt **ijbuf_cols, HYPRE_BigInt **ijbuf_rownums, HYPRE_Int **ijbuf_numcols,
                                           HYPRE_BigInt row_to_write, HYPRE_BigInt col_to_write, hypre_long_double val_to_write );
hypre_ParCSRMatrix *hypre_NonGalerkinSparsityPattern_flt (hypre_ParCSRMatrix *R_IAP,
                                                      hypre_ParCSRMatrix *RAP, HYPRE_Int * CF_marker, hypre_float droptol, HYPRE_Int sym_collapse,
                                                      HYPRE_Int collapse_beta );
hypre_ParCSRMatrix *hypre_NonGalerkinSparsityPattern_dbl (hypre_ParCSRMatrix *R_IAP,
                                                      hypre_ParCSRMatrix *RAP, HYPRE_Int * CF_marker, hypre_double droptol, HYPRE_Int sym_collapse,
                                                      HYPRE_Int collapse_beta );
hypre_ParCSRMatrix *hypre_NonGalerkinSparsityPattern_long_dbl (hypre_ParCSRMatrix *R_IAP,
                                                      hypre_ParCSRMatrix *RAP, HYPRE_Int * CF_marker, hypre_long_double droptol, HYPRE_Int sym_collapse,
                                                      HYPRE_Int collapse_beta );
HYPRE_Int hypre_SortedCopyParCSRData_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
HYPRE_Int hypre_SortedCopyParCSRData_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
HYPRE_Int hypre_SortedCopyParCSRData_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *B );
HYPRE_Int hypre_GenerateSendMapAndCommPkg_flt  ( MPI_Comm comm, HYPRE_Int num_sends, HYPRE_Int num_recvs,
                                            HYPRE_Int *recv_procs, HYPRE_Int *send_procs, HYPRE_Int *recv_vec_starts, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_GenerateSendMapAndCommPkg_dbl  ( MPI_Comm comm, HYPRE_Int num_sends, HYPRE_Int num_recvs,
                                            HYPRE_Int *recv_procs, HYPRE_Int *send_procs, HYPRE_Int *recv_vec_starts, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_GenerateSendMapAndCommPkg_long_dbl  ( MPI_Comm comm, HYPRE_Int num_sends, HYPRE_Int num_recvs,
                                            HYPRE_Int *recv_procs, HYPRE_Int *send_procs, HYPRE_Int *recv_vec_starts, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_GetCommPkgRTFromCommPkgA_flt  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *fine_to_coarse, HYPRE_Int *tmp_map_offd );
HYPRE_Int hypre_GetCommPkgRTFromCommPkgA_dbl  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *fine_to_coarse, HYPRE_Int *tmp_map_offd );
HYPRE_Int hypre_GetCommPkgRTFromCommPkgA_long_dbl  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                           HYPRE_Int *fine_to_coarse, HYPRE_Int *tmp_map_offd );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperator_flt  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                               hypre_ParCSRMatrix *P, hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperator_dbl  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                               hypre_ParCSRMatrix *P, hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperator_long_dbl  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                               hypre_ParCSRMatrix *P, hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperatorKT_flt  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keepTranspose, hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperatorKT_dbl  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keepTranspose, hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGBuildCoarseOperatorKT_long_dbl  ( hypre_ParCSRMatrix *RT, hypre_ParCSRMatrix *A,
                                                 hypre_ParCSRMatrix *P, HYPRE_Int keepTranspose, hypre_ParCSRMatrix **RAP_ptr );
HYPRE_Int hypre_BoomerAMGRelax_FCFJacobi_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, hypre_float relax_weight,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax_FCFJacobi_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, hypre_double relax_weight,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax_FCFJacobi_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, hypre_long_double relax_weight,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelaxIF_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                   HYPRE_Int relax_type, HYPRE_Int relax_order, HYPRE_Int cycle_type, hypre_float relax_weight,
                                   hypre_float omega, hypre_float *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp,
                                   hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelaxIF_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                   HYPRE_Int relax_type, HYPRE_Int relax_order, HYPRE_Int cycle_type, hypre_double relax_weight,
                                   hypre_double omega, hypre_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp,
                                   hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelaxIF_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                   HYPRE_Int relax_type, HYPRE_Int relax_order, HYPRE_Int cycle_type, hypre_long_double relax_weight,
                                   hypre_long_double omega, hypre_long_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp,
                                   hypre_ParVector *Ztemp );
HYPRE_Int hypre_ParCSRRelax_L1_Jacobi_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float *l1_norms,
                                        hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_ParCSRRelax_L1_Jacobi_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double *l1_norms,
                                        hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_ParCSRRelax_L1_Jacobi_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double *l1_norms,
                                        hypre_ParVector *u, hypre_ParVector *Vtemp );
hypre_float hypre_LINPACKcgpthy_flt  ( hypre_float *a, hypre_float *b );
hypre_double hypre_LINPACKcgpthy_dbl  ( hypre_double *a, hypre_double *b );
hypre_long_double hypre_LINPACKcgpthy_long_dbl  ( hypre_long_double *a, hypre_long_double *b );
HYPRE_Int hypre_LINPACKcgtql1_flt  ( HYPRE_Int *n, hypre_float *d, hypre_float *e, HYPRE_Int *ierr );
HYPRE_Int hypre_LINPACKcgtql1_dbl  ( HYPRE_Int *n, hypre_double *d, hypre_double *e, HYPRE_Int *ierr );
HYPRE_Int hypre_LINPACKcgtql1_long_dbl  ( HYPRE_Int *n, hypre_long_double *d, hypre_long_double *e, HYPRE_Int *ierr );
HYPRE_Int hypre_ParCSRMaxEigEstimate_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int scale, hypre_float *max_eig,
                                       hypre_float *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimate_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale, hypre_double *max_eig,
                                       hypre_double *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimate_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale, hypre_long_double *max_eig,
                                       hypre_long_double *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCG_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int scale, HYPRE_Int max_iter,
                                         hypre_float *max_eig, hypre_float *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCG_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale, HYPRE_Int max_iter,
                                         hypre_double *max_eig, hypre_double *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCG_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale, HYPRE_Int max_iter,
                                         hypre_long_double *max_eig, hypre_long_double *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCGHost_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                             HYPRE_Int max_iter, hypre_float *max_eig, hypre_float *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCGHost_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                             HYPRE_Int max_iter, hypre_double *max_eig, hypre_double *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateCGHost_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                             HYPRE_Int max_iter, hypre_long_double *max_eig, hypre_long_double *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateHost_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                           hypre_float *max_eig, hypre_float *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateHost_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                           hypre_double *max_eig, hypre_double *min_eig );
HYPRE_Int hypre_ParCSRMaxEigEstimateHost_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int scale,
                                           hypre_long_double *max_eig, hypre_long_double *min_eig );
HYPRE_Int hypre_ParCSRRelax_CG_flt  ( HYPRE_Solver solver, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int num_its );
HYPRE_Int hypre_ParCSRRelax_CG_dbl  ( HYPRE_Solver solver, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int num_its );
HYPRE_Int hypre_ParCSRRelax_CG_long_dbl  ( HYPRE_Solver solver, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int num_its );
HYPRE_Int hypre_ParCSRRelax_Cheby_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_float max_eig,
                                    hypre_float min_eig, hypre_float fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                    hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r );
HYPRE_Int hypre_ParCSRRelax_Cheby_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_double max_eig,
                                    hypre_double min_eig, hypre_double fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                    hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r );
HYPRE_Int hypre_ParCSRRelax_Cheby_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_long_double max_eig,
                                    hypre_long_double min_eig, hypre_long_double fraction, HYPRE_Int order, HYPRE_Int scale, HYPRE_Int variant,
                                    hypre_ParVector *u, hypre_ParVector *v, hypre_ParVector *r );
HYPRE_Int hypre_BoomerAMGRelax_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                 HYPRE_Int relax_type, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                 hypre_float *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                 HYPRE_Int relax_type, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                 hypre_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *f, HYPRE_Int *cf_marker,
                                 HYPRE_Int relax_type, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                 hypre_long_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax0WeightedJacobi_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                               HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_ParVector *u,
                                               hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax0WeightedJacobi_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                               HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_ParVector *u,
                                               hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax0WeightedJacobi_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                               HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_ParVector *u,
                                               hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax10TopoOrderedGaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                                        hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax10TopoOrderedGaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                                        hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax10TopoOrderedGaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                        HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                                        hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax11TwoStageGaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     hypre_float relax_weight, hypre_float omega,
                                                     hypre_float *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax11TwoStageGaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     hypre_double relax_weight, hypre_double omega,
                                                     hypre_double *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax11TwoStageGaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     hypre_long_double relax_weight, hypre_long_double omega,
                                                     hypre_long_double *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax12TwoStageGaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     hypre_float relax_weight, hypre_float omega,
                                                     hypre_float *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax12TwoStageGaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     hypre_double relax_weight, hypre_double omega,
                                                     hypre_double *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax12TwoStageGaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points,
                                                     hypre_long_double relax_weight, hypre_long_double omega,
                                                     hypre_long_double *A_diag_diag, hypre_ParVector *u,
                                                     hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax13HybridL1GaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                                     hypre_float *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax13HybridL1GaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                                     hypre_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax13HybridL1GaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                                     hypre_long_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax14HybridL1GaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                                     hypre_float *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax14HybridL1GaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                                     hypre_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax14HybridL1GaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                     HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                                     hypre_long_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax18WeightedL1Jacobi_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float *l1_norms,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax18WeightedL1Jacobi_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double *l1_norms,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax18WeightedL1Jacobi_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double *l1_norms,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax19GaussElim_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax19GaussElim_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax19GaussElim_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax1GaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax1GaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax1GaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax2GaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax2GaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax2GaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                            HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax3HybridGaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax3HybridGaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax3HybridGaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax4HybridGaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax4HybridGaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax4HybridGaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                  HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                                  hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax5ChaoticHybridGaussSeidel_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax5ChaoticHybridGaussSeidel_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax5ChaoticHybridGaussSeidel_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax6HybridSSOR_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax6HybridSSOR_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax6HybridSSOR_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                           HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                           hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax7Jacobi_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                       HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float *l1_norms,
                                       hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax7Jacobi_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                       HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double *l1_norms,
                                       hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax7Jacobi_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                       HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double *l1_norms,
                                       hypre_ParVector *u, hypre_ParVector *Vtemp );
HYPRE_Int hypre_BoomerAMGRelax8HybridL1SSOR_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                             HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                             hypre_float *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax8HybridL1SSOR_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                             HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                             hypre_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax8HybridL1SSOR_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                             HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                             hypre_long_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );
HYPRE_Int hypre_BoomerAMGRelax98GaussElimPivot_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax98GaussElimPivot_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelax98GaussElimPivot_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelaxComputeL1Norms_flt ( hypre_ParCSRMatrix *A, HYPRE_Int relax_type,
                                              HYPRE_Int relax_order, HYPRE_Int coarsest_lvl,
                                              hypre_IntArray *CF_marker,
                                              hypre_float **l1_norms_data_ptr );
HYPRE_Int hypre_BoomerAMGRelaxComputeL1Norms_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int relax_type,
                                              HYPRE_Int relax_order, HYPRE_Int coarsest_lvl,
                                              hypre_IntArray *CF_marker,
                                              hypre_double **l1_norms_data_ptr );
HYPRE_Int hypre_BoomerAMGRelaxComputeL1Norms_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int relax_type,
                                              HYPRE_Int relax_order, HYPRE_Int coarsest_lvl,
                                              hypre_IntArray *CF_marker,
                                              hypre_long_double **l1_norms_data_ptr );
HYPRE_Int hypre_BoomerAMGRelaxHybridGaussSeidel_core_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                      HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                                      hypre_float *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                                      HYPRE_Int GS_order, HYPRE_Int Symm, HYPRE_Int Skip_diag, HYPRE_Int forced_seq,
                                                      HYPRE_Int Topo_order );
HYPRE_Int hypre_BoomerAMGRelaxHybridGaussSeidel_core_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                      HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                                      hypre_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                                      HYPRE_Int GS_order, HYPRE_Int Symm, HYPRE_Int Skip_diag, HYPRE_Int forced_seq,
                                                      HYPRE_Int Topo_order );
HYPRE_Int hypre_BoomerAMGRelaxHybridGaussSeidel_core_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                                      HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                                      hypre_long_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                                      HYPRE_Int GS_order, HYPRE_Int Symm, HYPRE_Int Skip_diag, HYPRE_Int forced_seq,
                                                      HYPRE_Int Topo_order );
HYPRE_Int hypre_BoomerAMGRelaxHybridSOR_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_float relax_weight, hypre_float omega,
                                         hypre_float *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                         HYPRE_Int direction, HYPRE_Int symm, HYPRE_Int skip_diag, HYPRE_Int force_seq );
HYPRE_Int hypre_BoomerAMGRelaxHybridSOR_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_double relax_weight, hypre_double omega,
                                         hypre_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                         HYPRE_Int direction, HYPRE_Int symm, HYPRE_Int skip_diag, HYPRE_Int force_seq );
HYPRE_Int hypre_BoomerAMGRelaxHybridSOR_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                         HYPRE_Int *cf_marker, HYPRE_Int relax_points, hypre_long_double relax_weight, hypre_long_double omega,
                                         hypre_long_double *l1_norms, hypre_ParVector *u, hypre_ParVector *Vtemp, hypre_ParVector *Ztemp,
                                         HYPRE_Int direction, HYPRE_Int symm, HYPRE_Int skip_diag, HYPRE_Int force_seq );
HYPRE_Int hypre_BoomerAMGRelaxKaczmarz_flt ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_float omega,
                                        hypre_float *l1_norms, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelaxKaczmarz_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_double omega,
                                        hypre_double *l1_norms, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGRelaxKaczmarz_long_dbl ( hypre_ParCSRMatrix *A, hypre_ParVector *f, hypre_long_double omega,
                                        hypre_long_double *l1_norms, hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGBuildRestrAIR_flt ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        hypre_float filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr,
                                        HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_Int hypre_BoomerAMGBuildRestrAIR_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        hypre_double filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr,
                                        HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_Int hypre_BoomerAMGBuildRestrAIR_long_dbl ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                        hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        hypre_long_double filter_thresholdR, HYPRE_Int debug_flag, hypre_ParCSRMatrix **R_ptr,
                                        HYPRE_Int is_triangular, HYPRE_Int gmres_switch);
HYPRE_ParCSRMatrix GenerateRotate7pt_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_Int P,
                                       HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, hypre_float alpha, hypre_float eps );
HYPRE_ParCSRMatrix GenerateRotate7pt_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_Int P,
                                       HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, hypre_double alpha, hypre_double eps );
HYPRE_ParCSRMatrix GenerateRotate7pt_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_Int P,
                                       HYPRE_Int Q, HYPRE_Int p, HYPRE_Int q, hypre_long_double alpha, hypre_long_double eps );
HYPRE_Int hypre_ParCSRMatrixScaledNorm_flt  ( hypre_ParCSRMatrix *A, hypre_float *scnorm );
HYPRE_Int hypre_ParCSRMatrixScaledNorm_dbl  ( hypre_ParCSRMatrix *A, hypre_double *scnorm );
HYPRE_Int hypre_ParCSRMatrixScaledNorm_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double *scnorm );
HYPRE_Int hypre_SchwarzCFSolve_flt  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt );
HYPRE_Int hypre_SchwarzCFSolve_dbl  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt );
HYPRE_Int hypre_SchwarzCFSolve_long_dbl  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                                 hypre_ParVector *u, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt );
void *hypre_SchwarzCreate_flt  ( void );
void *hypre_SchwarzCreate_dbl  ( void );
void *hypre_SchwarzCreate_long_dbl  ( void );
HYPRE_Int hypre_SchwarzDestroy_flt  ( void *data );
HYPRE_Int hypre_SchwarzDestroy_dbl  ( void *data );
HYPRE_Int hypre_SchwarzDestroy_long_dbl  ( void *data );
HYPRE_Int hypre_SchwarzReScale_flt  ( void *data, HYPRE_Int size, hypre_float value );
HYPRE_Int hypre_SchwarzReScale_dbl  ( void *data, HYPRE_Int size, hypre_double value );
HYPRE_Int hypre_SchwarzReScale_long_dbl  ( void *data, HYPRE_Int size, hypre_long_double value );
HYPRE_Int hypre_SchwarzSetDofFunc_flt  ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_SchwarzSetDofFunc_dbl  ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_SchwarzSetDofFunc_long_dbl  ( void *data, HYPRE_Int *dof_func );
HYPRE_Int hypre_SchwarzSetDomainStructure_flt  ( void *data, hypre_CSRMatrix *domain_structure );
HYPRE_Int hypre_SchwarzSetDomainStructure_dbl  ( void *data, hypre_CSRMatrix *domain_structure );
HYPRE_Int hypre_SchwarzSetDomainStructure_long_dbl  ( void *data, hypre_CSRMatrix *domain_structure );
HYPRE_Int hypre_SchwarzSetDomainType_flt  ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_SchwarzSetDomainType_dbl  ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_SchwarzSetDomainType_long_dbl  ( void *data, HYPRE_Int domain_type );
HYPRE_Int hypre_SchwarzSetNonSymm_flt  ( void *data, HYPRE_Int value );
HYPRE_Int hypre_SchwarzSetNonSymm_dbl  ( void *data, HYPRE_Int value );
HYPRE_Int hypre_SchwarzSetNonSymm_long_dbl  ( void *data, HYPRE_Int value );
HYPRE_Int hypre_SchwarzSetNumFunctions_flt  ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_SchwarzSetNumFunctions_dbl  ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_SchwarzSetNumFunctions_long_dbl  ( void *data, HYPRE_Int num_functions );
HYPRE_Int hypre_SchwarzSetOverlap_flt  ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_SchwarzSetOverlap_dbl  ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_SchwarzSetOverlap_long_dbl  ( void *data, HYPRE_Int overlap );
HYPRE_Int hypre_SchwarzSetRelaxWeight_flt  ( void *data, hypre_float relax_weight );
HYPRE_Int hypre_SchwarzSetRelaxWeight_dbl  ( void *data, hypre_double relax_weight );
HYPRE_Int hypre_SchwarzSetRelaxWeight_long_dbl  ( void *data, hypre_long_double relax_weight );
HYPRE_Int hypre_SchwarzSetScale_flt  ( void *data, hypre_float *scale );
HYPRE_Int hypre_SchwarzSetScale_dbl  ( void *data, hypre_double *scale );
HYPRE_Int hypre_SchwarzSetScale_long_dbl  ( void *data, hypre_long_double *scale );
HYPRE_Int hypre_SchwarzSetup_flt  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_SchwarzSetup_dbl  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_SchwarzSetup_long_dbl  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_SchwarzSetVariant_flt  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_SchwarzSetVariant_dbl  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_SchwarzSetVariant_long_dbl  ( void *data, HYPRE_Int variant );
HYPRE_Int hypre_SchwarzSolve_flt  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_SchwarzSolve_dbl  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_SchwarzSolve_long_dbl  ( void *schwarz_vdata, hypre_ParCSRMatrix *A, hypre_ParVector *f,
                               hypre_ParVector *u );
HYPRE_Int hypre_BoomerAMGSetupStats_flt  ( void *amg_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_BoomerAMGSetupStats_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_BoomerAMGSetupStats_long_dbl  ( void *amg_vdata, hypre_ParCSRMatrix *A );
HYPRE_Int hypre_BoomerAMGWriteSolverParams_flt  ( void *data );
HYPRE_Int hypre_BoomerAMGWriteSolverParams_dbl  ( void *data );
HYPRE_Int hypre_BoomerAMGWriteSolverParams_long_dbl  ( void *data );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker_flt  ( hypre_IntArray *CF_marker,
                                           hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker_dbl  ( hypre_IntArray *CF_marker,
                                           hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker_long_dbl  ( hypre_IntArray *CF_marker,
                                           hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2_flt  ( hypre_IntArray *CF_marker,
                                            hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2_dbl  ( hypre_IntArray *CF_marker,
                                            hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2_long_dbl  ( hypre_IntArray *CF_marker,
                                            hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2Host_flt  ( hypre_IntArray *CF_marker,
                                                hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2Host_dbl  ( hypre_IntArray *CF_marker,
                                                hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarker2Host_long_dbl  ( hypre_IntArray *CF_marker,
                                                hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarkerHost_flt  ( hypre_IntArray *CF_marker,
                                               hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarkerHost_dbl  ( hypre_IntArray *CF_marker,
                                               hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCorrectCFMarkerHost_long_dbl  ( hypre_IntArray *CF_marker,
                                               hypre_IntArray *new_CF_marker );
HYPRE_Int hypre_BoomerAMGCreate2ndS_flt  ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                      HYPRE_Int num_paths, HYPRE_BigInt *coarse_row_starts, hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_BoomerAMGCreate2ndS_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                      HYPRE_Int num_paths, HYPRE_BigInt *coarse_row_starts, hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_BoomerAMGCreate2ndS_long_dbl  ( hypre_ParCSRMatrix *S, HYPRE_Int *CF_marker,
                                      HYPRE_Int num_paths, HYPRE_BigInt *coarse_row_starts, hypre_ParCSRMatrix **C_ptr );
HYPRE_Int hypre_BoomerAMGCreateS_flt  ( hypre_ParCSRMatrix *A, hypre_float strength_threshold,
                                   hypre_float max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateS_dbl  ( hypre_ParCSRMatrix *A, hypre_double strength_threshold,
                                   hypre_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateS_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double strength_threshold,
                                   hypre_long_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabs_flt  ( hypre_ParCSRMatrix *A, hypre_float strength_threshold,
                                      hypre_float max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabs_dbl  ( hypre_ParCSRMatrix *A, hypre_double strength_threshold,
                                      hypre_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabs_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double strength_threshold,
                                      hypre_long_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabsHost_flt  ( hypre_ParCSRMatrix *A, hypre_float strength_threshold,
                                          hypre_float max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabsHost_dbl  ( hypre_ParCSRMatrix *A, hypre_double strength_threshold,
                                          hypre_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSabsHost_long_dbl  ( hypre_ParCSRMatrix *A, hypre_long_double strength_threshold,
                                          hypre_long_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr );
HYPRE_Int hypre_BoomerAMGCreateSCommPkg_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                          HYPRE_Int **col_offd_S_to_A_ptr );
HYPRE_Int hypre_BoomerAMGCreateSCommPkg_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                          HYPRE_Int **col_offd_S_to_A_ptr );
HYPRE_Int hypre_BoomerAMGCreateSCommPkg_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *S,
                                          HYPRE_Int **col_offd_S_to_A_ptr );
HYPRE_Int hypre_BoomerAMGCreateSFromCFMarker_flt (hypre_ParCSRMatrix    *A,
                                             hypre_float strength_threshold, hypre_float max_row_sum, HYPRE_Int *CF_marker,
                                             HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int SMRK, hypre_ParCSRMatrix    **S_ptr);
HYPRE_Int hypre_BoomerAMGCreateSFromCFMarker_dbl (hypre_ParCSRMatrix    *A,
                                             hypre_double strength_threshold, hypre_double max_row_sum, HYPRE_Int *CF_marker,
                                             HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int SMRK, hypre_ParCSRMatrix    **S_ptr);
HYPRE_Int hypre_BoomerAMGCreateSFromCFMarker_long_dbl (hypre_ParCSRMatrix    *A,
                                             hypre_long_double strength_threshold, hypre_long_double max_row_sum, HYPRE_Int *CF_marker,
                                             HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int SMRK, hypre_ParCSRMatrix    **S_ptr);
HYPRE_Int hypre_BoomerAMGCreateSHost_flt (hypre_ParCSRMatrix *A, hypre_float strength_threshold,
                                     hypre_float max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr);
HYPRE_Int hypre_BoomerAMGCreateSHost_dbl (hypre_ParCSRMatrix *A, hypre_double strength_threshold,
                                     hypre_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr);
HYPRE_Int hypre_BoomerAMGCreateSHost_long_dbl (hypre_ParCSRMatrix *A, hypre_long_double strength_threshold,
                                     hypre_long_double max_row_sum, HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_ParCSRMatrix **S_ptr);
HYPRE_Int hypre_BoomerAMG_LNExpandInterp_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, hypre_IntArray **coarse_dof_func,
                                           HYPRE_Int *CF_marker, HYPRE_Int level, hypre_float *weights, HYPRE_Int num_smooth_vecs,
                                           hypre_ParVector **smooth_vecs, hypre_float abs_trunc, HYPRE_Int q_max,
                                           HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMG_LNExpandInterp_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, hypre_IntArray **coarse_dof_func,
                                           HYPRE_Int *CF_marker, HYPRE_Int level, hypre_double *weights, HYPRE_Int num_smooth_vecs,
                                           hypre_ParVector **smooth_vecs, hypre_double abs_trunc, HYPRE_Int q_max,
                                           HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMG_LNExpandInterp_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, hypre_IntArray **coarse_dof_func,
                                           HYPRE_Int *CF_marker, HYPRE_Int level, hypre_long_double *weights, HYPRE_Int num_smooth_vecs,
                                           hypre_ParVector **smooth_vecs, hypre_long_double abs_trunc, HYPRE_Int q_max,
                                           HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMGCoarsenInterpVectors_flt  ( hypre_ParCSRMatrix *P, HYPRE_Int num_smooth_vecs,
                                                hypre_ParVector **smooth_vecs, HYPRE_Int *CF_marker, hypre_ParVector ***new_smooth_vecs,
                                                HYPRE_Int expand_level, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMGCoarsenInterpVectors_dbl  ( hypre_ParCSRMatrix *P, HYPRE_Int num_smooth_vecs,
                                                hypre_ParVector **smooth_vecs, HYPRE_Int *CF_marker, hypre_ParVector ***new_smooth_vecs,
                                                HYPRE_Int expand_level, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMGCoarsenInterpVectors_long_dbl  ( hypre_ParCSRMatrix *P, HYPRE_Int num_smooth_vecs,
                                                hypre_ParVector **smooth_vecs, HYPRE_Int *CF_marker, hypre_ParVector ***new_smooth_vecs,
                                                HYPRE_Int expand_level, HYPRE_Int num_functions );
HYPRE_Int hypre_BoomerAMG_GMExpandInterp_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_Int num_smooth_vecs, hypre_ParVector **smooth_vecs, HYPRE_Int *nf, HYPRE_Int *dof_func,
                                           hypre_IntArray **coarse_dof_func, HYPRE_Int variant, HYPRE_Int level, hypre_float abs_trunc,
                                           hypre_float *weights, HYPRE_Int q_max, HYPRE_Int *CF_marker, HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMG_GMExpandInterp_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_Int num_smooth_vecs, hypre_ParVector **smooth_vecs, HYPRE_Int *nf, HYPRE_Int *dof_func,
                                           hypre_IntArray **coarse_dof_func, HYPRE_Int variant, HYPRE_Int level, hypre_double abs_trunc,
                                           hypre_double *weights, HYPRE_Int q_max, HYPRE_Int *CF_marker, HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMG_GMExpandInterp_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **P,
                                           HYPRE_Int num_smooth_vecs, hypre_ParVector **smooth_vecs, HYPRE_Int *nf, HYPRE_Int *dof_func,
                                           hypre_IntArray **coarse_dof_func, HYPRE_Int variant, HYPRE_Int level, hypre_long_double abs_trunc,
                                           hypre_long_double *weights, HYPRE_Int q_max, HYPRE_Int *CF_marker, HYPRE_Int interp_vec_first_level );
HYPRE_Int hypre_BoomerAMGRefineInterp_flt  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                        HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                        HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGRefineInterp_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                        HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                        HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGRefineInterp_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
                                        HYPRE_BigInt *num_cpts_global, HYPRE_Int *nf, HYPRE_Int *dof_func, HYPRE_Int *CF_marker,
                                        HYPRE_Int level );
HYPRE_Int hypre_BoomerAMGSmoothInterpVectors_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int num_smooth_vecs,
                                               hypre_ParVector **smooth_vecs, HYPRE_Int smooth_steps );
HYPRE_Int hypre_BoomerAMGSmoothInterpVectors_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int num_smooth_vecs,
                                               hypre_ParVector **smooth_vecs, HYPRE_Int smooth_steps );
HYPRE_Int hypre_BoomerAMGSmoothInterpVectors_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int num_smooth_vecs,
                                               hypre_ParVector **smooth_vecs, HYPRE_Int smooth_steps );
HYPRE_Int hypre_BoomerAMGBuildPartialExtInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_float trunc_factor,
                                                 HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialExtInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_double trunc_factor,
                                                 HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialExtInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                                 HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialExtPIInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                   hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                   HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_float trunc_factor,
                                                   HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialExtPIInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                   hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                   HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_double trunc_factor,
                                                   HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialExtPIInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                   hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                   HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                                   HYPRE_Int max_elmts, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialStdInterp_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_float trunc_factor,
                                                 HYPRE_Int max_elmts, HYPRE_Int sep_weight, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialStdInterp_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_double trunc_factor,
                                                 HYPRE_Int max_elmts, HYPRE_Int sep_weight, hypre_ParCSRMatrix **P_ptr );
HYPRE_Int hypre_BoomerAMGBuildPartialStdInterp_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int *CF_marker,
                                                 hypre_ParCSRMatrix *S, HYPRE_BigInt *num_cpts_global, HYPRE_BigInt *num_old_cpts_global,
                                                 HYPRE_Int num_functions, HYPRE_Int *dof_func, HYPRE_Int debug_flag, hypre_long_double trunc_factor,
                                                 HYPRE_Int max_elmts, HYPRE_Int sep_weight, hypre_ParCSRMatrix **P_ptr );
hypre_float afun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double afun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double afun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float bfun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double bfun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double bfun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float bndfun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double bndfun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double bndfun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float cfun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double cfun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double cfun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float dfun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double dfun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double dfun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float efun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double efun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double efun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float ffun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double ffun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double ffun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
HYPRE_ParCSRMatrix GenerateVarDifConv_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                        HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                        hypre_float eps, HYPRE_ParVector *rhs_ptr );
HYPRE_ParCSRMatrix GenerateVarDifConv_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                        HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                        hypre_double eps, HYPRE_ParVector *rhs_ptr );
HYPRE_ParCSRMatrix GenerateVarDifConv_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                        HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                        hypre_long_double eps, HYPRE_ParVector *rhs_ptr );
hypre_float gfun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double gfun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double gfun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float rfun_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double rfun_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double rfun_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float afun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double afun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double afun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float bfun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double bfun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double bfun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float bndfun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double bndfun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double bndfun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float cfun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double cfun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double cfun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float dfun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double dfun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double dfun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float efun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double efun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double efun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float ffun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double ffun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double ffun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
HYPRE_ParCSRMatrix GenerateRSVarDifConv_flt  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          hypre_float eps, HYPRE_ParVector *rhs_ptr, HYPRE_Int type );
HYPRE_ParCSRMatrix GenerateRSVarDifConv_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          hypre_double eps, HYPRE_ParVector *rhs_ptr, HYPRE_Int type );
HYPRE_ParCSRMatrix GenerateRSVarDifConv_long_dbl  ( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny,
                                          HYPRE_BigInt nz, HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R, HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          hypre_long_double eps, HYPRE_ParVector *rhs_ptr, HYPRE_Int type );
hypre_float gfun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double gfun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double gfun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
hypre_float rfun_rs_flt  ( hypre_float xx, hypre_float yy, hypre_float zz );
hypre_double rfun_rs_dbl  ( hypre_double xx, hypre_double yy, hypre_double zz );
hypre_long_double rfun_rs_long_dbl  ( hypre_long_double xx, hypre_long_double yy, hypre_long_double zz );
HYPRE_Int hypre_AdSchwarzCFSolve_flt  ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                   hypre_CSRMatrix *domain_structure, hypre_float *scale, hypre_ParVector *par_x,
                                   hypre_ParVector *par_aux, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AdSchwarzCFSolve_dbl  ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                   hypre_CSRMatrix *domain_structure, hypre_double *scale, hypre_ParVector *par_x,
                                   hypre_ParVector *par_aux, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AdSchwarzCFSolve_long_dbl  ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                   hypre_CSRMatrix *domain_structure, hypre_long_double *scale, hypre_ParVector *par_x,
                                   hypre_ParVector *par_aux, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AdSchwarzSolve_flt  ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                 hypre_CSRMatrix *domain_structure, hypre_float *scale, hypre_ParVector *par_x,
                                 hypre_ParVector *par_aux, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AdSchwarzSolve_dbl  ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                 hypre_CSRMatrix *domain_structure, hypre_double *scale, hypre_ParVector *par_x,
                                 hypre_ParVector *par_aux, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AdSchwarzSolve_long_dbl  ( hypre_ParCSRMatrix *par_A, hypre_ParVector *par_rhs,
                                 hypre_CSRMatrix *domain_structure, hypre_long_double *scale, hypre_ParVector *par_x,
                                 hypre_ParVector *par_aux, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AMGCreateDomainDof_flt  ( hypre_CSRMatrix *A, HYPRE_Int domain_type, HYPRE_Int overlap,
                                     HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_CSRMatrix **domain_structure_pointer,
                                     HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AMGCreateDomainDof_dbl  ( hypre_CSRMatrix *A, HYPRE_Int domain_type, HYPRE_Int overlap,
                                     HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_CSRMatrix **domain_structure_pointer,
                                     HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AMGCreateDomainDof_long_dbl  ( hypre_CSRMatrix *A, HYPRE_Int domain_type, HYPRE_Int overlap,
                                     HYPRE_Int num_functions, HYPRE_Int *dof_func, hypre_CSRMatrix **domain_structure_pointer,
                                     HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_AMGeAgglomerate_flt  ( HYPRE_Int *i_AE_element, HYPRE_Int *j_AE_element,
                                  HYPRE_Int *i_face_face, HYPRE_Int *j_face_face, HYPRE_Int *w_face_face, HYPRE_Int *i_face_element,
                                  HYPRE_Int *j_face_element, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_to_prefer_weight, HYPRE_Int *i_face_weight, HYPRE_Int num_faces,
                                  HYPRE_Int num_elements, HYPRE_Int *num_AEs_pointer );
HYPRE_Int hypre_AMGeAgglomerate_dbl  ( HYPRE_Int *i_AE_element, HYPRE_Int *j_AE_element,
                                  HYPRE_Int *i_face_face, HYPRE_Int *j_face_face, HYPRE_Int *w_face_face, HYPRE_Int *i_face_element,
                                  HYPRE_Int *j_face_element, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_to_prefer_weight, HYPRE_Int *i_face_weight, HYPRE_Int num_faces,
                                  HYPRE_Int num_elements, HYPRE_Int *num_AEs_pointer );
HYPRE_Int hypre_AMGeAgglomerate_long_dbl  ( HYPRE_Int *i_AE_element, HYPRE_Int *j_AE_element,
                                  HYPRE_Int *i_face_face, HYPRE_Int *j_face_face, HYPRE_Int *w_face_face, HYPRE_Int *i_face_element,
                                  HYPRE_Int *j_face_element, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_to_prefer_weight, HYPRE_Int *i_face_weight, HYPRE_Int num_faces,
                                  HYPRE_Int num_elements, HYPRE_Int *num_AEs_pointer );
HYPRE_Int hypre_AMGNodalSchwarzSmoother_flt  ( hypre_CSRMatrix *A, HYPRE_Int num_functions,
                                          HYPRE_Int option, hypre_CSRMatrix **domain_structure_pointer );
HYPRE_Int hypre_AMGNodalSchwarzSmoother_dbl  ( hypre_CSRMatrix *A, HYPRE_Int num_functions,
                                          HYPRE_Int option, hypre_CSRMatrix **domain_structure_pointer );
HYPRE_Int hypre_AMGNodalSchwarzSmoother_long_dbl  ( hypre_CSRMatrix *A, HYPRE_Int num_functions,
                                          HYPRE_Int option, hypre_CSRMatrix **domain_structure_pointer );
HYPRE_Int hypre_GenerateScale_flt  ( hypre_CSRMatrix *domain_structure, HYPRE_Int num_variables,
                                hypre_float relaxation_weight, hypre_float **scale_pointer );
HYPRE_Int hypre_GenerateScale_dbl  ( hypre_CSRMatrix *domain_structure, HYPRE_Int num_variables,
                                hypre_double relaxation_weight, hypre_double **scale_pointer );
HYPRE_Int hypre_GenerateScale_long_dbl  ( hypre_CSRMatrix *domain_structure, HYPRE_Int num_variables,
                                hypre_long_double relaxation_weight, hypre_long_double **scale_pointer );
HYPRE_Int hypre_matinv_flt  ( hypre_float *x, hypre_float *a, HYPRE_Int k );
HYPRE_Int hypre_matinv_dbl  ( hypre_double *x, hypre_double *a, HYPRE_Int k );
HYPRE_Int hypre_matinv_long_dbl  ( hypre_long_double *x, hypre_long_double *a, HYPRE_Int k );
HYPRE_Int hypre_move_entry_flt  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                             HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_move_entry_dbl  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                             HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_move_entry_long_dbl  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                             HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_MPSchwarzCFFWSolve_flt  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                     hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_float relax_wt,
                                     hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                     HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFFWSolve_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                     hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_double relax_wt,
                                     hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                     HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFFWSolve_long_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                     hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_long_double relax_wt,
                                     hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                     HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFSolve_flt  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_float relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFSolve_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_double relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzCFSolve_long_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_long_double relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *CF_marker, HYPRE_Int rlx_pt, HYPRE_Int *pivots,
                                   HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzFWSolve_flt  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_float relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzFWSolve_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_double relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzFWSolve_long_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                   hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_long_double relax_wt,
                                   hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzSolve_flt  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                 hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_float relax_wt,
                                 hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzSolve_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                 hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_double relax_wt,
                                 hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_MPSchwarzSolve_long_dbl  ( hypre_ParCSRMatrix *par_A, hypre_Vector *rhs_vector,
                                 hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x, hypre_long_double relax_wt,
                                 hypre_Vector *aux_vector, HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAdSchwarzSolve_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *F,
                                    hypre_CSRMatrix *domain_structure, hypre_float *scale, hypre_ParVector *X, hypre_ParVector *Vtemp,
                                    HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAdSchwarzSolve_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *F,
                                    hypre_CSRMatrix *domain_structure, hypre_double *scale, hypre_ParVector *X, hypre_ParVector *Vtemp,
                                    HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAdSchwarzSolve_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *F,
                                    hypre_CSRMatrix *domain_structure, hypre_long_double *scale, hypre_ParVector *X, hypre_ParVector *Vtemp,
                                    HYPRE_Int *pivots, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAMGCreateDomainDof_flt  ( hypre_ParCSRMatrix *A, HYPRE_Int domain_type,
                                        HYPRE_Int overlap, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        hypre_CSRMatrix **domain_structure_pointer, HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAMGCreateDomainDof_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int domain_type,
                                        HYPRE_Int overlap, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        hypre_CSRMatrix **domain_structure_pointer, HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParAMGCreateDomainDof_long_dbl  ( hypre_ParCSRMatrix *A, HYPRE_Int domain_type,
                                        HYPRE_Int overlap, HYPRE_Int num_functions, HYPRE_Int *dof_func,
                                        hypre_CSRMatrix **domain_structure_pointer, HYPRE_Int **piv_pointer, HYPRE_Int use_nonsymm );
HYPRE_Int hypre_parCorrRes_flt  ( hypre_ParCSRMatrix *A, hypre_ParVector *x, hypre_Vector *rhs,
                             hypre_Vector **tmp_ptr );
HYPRE_Int hypre_parCorrRes_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *x, hypre_Vector *rhs,
                             hypre_Vector **tmp_ptr );
HYPRE_Int hypre_parCorrRes_long_dbl  ( hypre_ParCSRMatrix *A, hypre_ParVector *x, hypre_Vector *rhs,
                             hypre_Vector **tmp_ptr );
HYPRE_Int hypre_ParGenerateHybridScale_flt  ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                         hypre_CSRMatrix **A_boundary_pointer, hypre_float **scale_pointer );
HYPRE_Int hypre_ParGenerateHybridScale_dbl  ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                         hypre_CSRMatrix **A_boundary_pointer, hypre_double **scale_pointer );
HYPRE_Int hypre_ParGenerateHybridScale_long_dbl  ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                         hypre_CSRMatrix **A_boundary_pointer, hypre_long_double **scale_pointer );
HYPRE_Int hypre_ParGenerateScale_flt  ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                   hypre_float relaxation_weight, hypre_float **scale_pointer );
HYPRE_Int hypre_ParGenerateScale_dbl  ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                   hypre_double relaxation_weight, hypre_double **scale_pointer );
HYPRE_Int hypre_ParGenerateScale_long_dbl  ( hypre_ParCSRMatrix *A, hypre_CSRMatrix *domain_structure,
                                   hypre_long_double relaxation_weight, hypre_long_double **scale_pointer );
HYPRE_Int hypre_ParMPSchwarzSolve_flt  ( hypre_ParCSRMatrix *par_A, hypre_CSRMatrix *A_boundary,
                                    hypre_ParVector *rhs_vector, hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x,
                                    hypre_float relax_wt, hypre_float *scale, hypre_ParVector *Vtemp, HYPRE_Int *pivots,
                                    HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParMPSchwarzSolve_dbl  ( hypre_ParCSRMatrix *par_A, hypre_CSRMatrix *A_boundary,
                                    hypre_ParVector *rhs_vector, hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x,
                                    hypre_double relax_wt, hypre_double *scale, hypre_ParVector *Vtemp, HYPRE_Int *pivots,
                                    HYPRE_Int use_nonsymm );
HYPRE_Int hypre_ParMPSchwarzSolve_long_dbl  ( hypre_ParCSRMatrix *par_A, hypre_CSRMatrix *A_boundary,
                                    hypre_ParVector *rhs_vector, hypre_CSRMatrix *domain_structure, hypre_ParVector *par_x,
                                    hypre_long_double relax_wt, hypre_long_double *scale, hypre_ParVector *Vtemp, HYPRE_Int *pivots,
                                    HYPRE_Int use_nonsymm );
HYPRE_Int hypre_remove_entry_flt  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_remove_entry_dbl  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_remove_entry_long_dbl  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_update_entry_flt  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_update_entry_dbl  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int hypre_update_entry_long_dbl  ( HYPRE_Int weight, HYPRE_Int *weight_max, HYPRE_Int *previous,
                               HYPRE_Int *next, HYPRE_Int *first, HYPRE_Int *last, HYPRE_Int head, HYPRE_Int tail, HYPRE_Int i );
HYPRE_Int matrix_matrix_product_flt  ( HYPRE_Int **i_element_edge_pointer,
                                  HYPRE_Int **j_element_edge_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_edge, HYPRE_Int *j_face_edge, HYPRE_Int num_elements, HYPRE_Int num_faces,
                                  HYPRE_Int num_edges );
HYPRE_Int matrix_matrix_product_dbl  ( HYPRE_Int **i_element_edge_pointer,
                                  HYPRE_Int **j_element_edge_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_edge, HYPRE_Int *j_face_edge, HYPRE_Int num_elements, HYPRE_Int num_faces,
                                  HYPRE_Int num_edges );
HYPRE_Int matrix_matrix_product_long_dbl  ( HYPRE_Int **i_element_edge_pointer,
                                  HYPRE_Int **j_element_edge_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                  HYPRE_Int *i_face_edge, HYPRE_Int *j_face_edge, HYPRE_Int num_elements, HYPRE_Int num_faces,
                                  HYPRE_Int num_edges );
HYPRE_Int transpose_matrix_create_flt  ( HYPRE_Int **i_face_element_pointer,
                                    HYPRE_Int **j_face_element_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                    HYPRE_Int num_elements, HYPRE_Int num_faces );
HYPRE_Int transpose_matrix_create_dbl  ( HYPRE_Int **i_face_element_pointer,
                                    HYPRE_Int **j_face_element_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                    HYPRE_Int num_elements, HYPRE_Int num_faces );
HYPRE_Int transpose_matrix_create_long_dbl  ( HYPRE_Int **i_face_element_pointer,
                                    HYPRE_Int **j_face_element_pointer, HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
                                    HYPRE_Int num_elements, HYPRE_Int num_faces );

#endif

#endif
