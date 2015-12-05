/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/

#include "_hypre_utilities.h"

#include "HYPRE_struct_ls.h"

#ifndef hypre_STRUCT_LS_HEADER
#define hypre_STRUCT_LS_HEADER

#include "_hypre_struct_mv.h"
#include "krylov.h"

#include "temp_multivector.h"
 /* ... needed to make sense of functions in HYPRE_parcsr_int.c */
#include "HYPRE_MatvecFunctions.h"
 /* ... needed to make sense of functions in HYPRE_parcsr_int.c */

#ifdef __cplusplus
extern "C" {
#endif


/* coarsen.c */
HYPRE_Int hypre_StructMapFineToCoarse ( hypre_Index findex , hypre_Index index , hypre_Index stride , hypre_Index cindex );
HYPRE_Int hypre_StructMapCoarseToFine ( hypre_Index cindex , hypre_Index index , hypre_Index stride , hypre_Index findex );
HYPRE_Int hypre_StructCoarsen ( hypre_StructGrid *fgrid , hypre_Index index , hypre_Index stride , HYPRE_Int prune , hypre_StructGrid **cgrid_ptr );
HYPRE_Int hypre_Merge ( HYPRE_Int **arrays , HYPRE_Int *sizes , HYPRE_Int size , HYPRE_Int **mergei_ptr , HYPRE_Int **mergej_ptr );

/* cyclic_reduction.c */
void *hypre_CyclicReductionCreate ( MPI_Comm comm );
hypre_StructMatrix *hypre_CycRedCreateCoarseOp ( hypre_StructMatrix *A , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_CycRedSetupCoarseOp ( hypre_StructMatrix *A , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride );
HYPRE_Int hypre_CyclicReductionSetup ( void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_CyclicReduction ( void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_CyclicReductionSetBase ( void *cyc_red_vdata , hypre_Index base_index , hypre_Index base_stride );
HYPRE_Int hypre_CyclicReductionDestroy ( void *cyc_red_vdata );

/* general.c */
HYPRE_Int hypre_Log2 ( HYPRE_Int p );

/* hybrid.c */
void *hypre_HybridCreate ( MPI_Comm comm );
HYPRE_Int hypre_HybridDestroy ( void *hybrid_vdata );
HYPRE_Int hypre_HybridSetTol ( void *hybrid_vdata , double tol );
HYPRE_Int hypre_HybridSetConvergenceTol ( void *hybrid_vdata , double cf_tol );
HYPRE_Int hypre_HybridSetDSCGMaxIter ( void *hybrid_vdata , HYPRE_Int dscg_max_its );
HYPRE_Int hypre_HybridSetPCGMaxIter ( void *hybrid_vdata , HYPRE_Int pcg_max_its );
HYPRE_Int hypre_HybridSetPCGAbsoluteTolFactor ( void *hybrid_vdata , double pcg_atolf );
HYPRE_Int hypre_HybridSetTwoNorm ( void *hybrid_vdata , HYPRE_Int two_norm );
HYPRE_Int hypre_HybridSetStopCrit ( void *hybrid_vdata , HYPRE_Int stop_crit );
HYPRE_Int hypre_HybridSetRelChange ( void *hybrid_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_HybridSetSolverType ( void *hybrid_vdata , HYPRE_Int solver_type );
HYPRE_Int hypre_HybridSetKDim ( void *hybrid_vdata , HYPRE_Int k_dim );
HYPRE_Int hypre_HybridSetPrecond ( void *pcg_vdata , HYPRE_Int (*pcg_precond_solve )(), HYPRE_Int (*pcg_precond_setup )(), void *pcg_precond );
HYPRE_Int hypre_HybridSetLogging ( void *hybrid_vdata , HYPRE_Int logging );
HYPRE_Int hypre_HybridSetPrintLevel ( void *hybrid_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_HybridGetNumIterations ( void *hybrid_vdata , HYPRE_Int *num_its );
HYPRE_Int hypre_HybridGetDSCGNumIterations ( void *hybrid_vdata , HYPRE_Int *dscg_num_its );
HYPRE_Int hypre_HybridGetPCGNumIterations ( void *hybrid_vdata , HYPRE_Int *pcg_num_its );
HYPRE_Int hypre_HybridGetFinalRelativeResidualNorm ( void *hybrid_vdata , double *final_rel_res_norm );
HYPRE_Int hypre_HybridSetup ( void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_HybridSolve ( void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* HYPRE_struct_bicgstab.c */
HYPRE_Int HYPRE_StructBiCGSTABCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructBiCGSTABDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructBiCGSTABSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructBiCGSTABSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructBiCGSTABSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructBiCGSTABSetAbsoluteTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructBiCGSTABSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructBiCGSTABSetPrecond ( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
HYPRE_Int HYPRE_StructBiCGSTABSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructBiCGSTABSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int level );
HYPRE_Int HYPRE_StructBiCGSTABGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );
HYPRE_Int HYPRE_StructBiCGSTABGetResidual ( HYPRE_StructSolver solver , void **residual );

/* HYPRE_struct_flexgmres.c */
HYPRE_Int HYPRE_StructFlexGMRESCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructFlexGMRESDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructFlexGMRESSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructFlexGMRESSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructFlexGMRESSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructFlexGMRESSetAbsoluteTol ( HYPRE_StructSolver solver , double atol );
HYPRE_Int HYPRE_StructFlexGMRESSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructFlexGMRESSetKDim ( HYPRE_StructSolver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_StructFlexGMRESSetPrecond ( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
HYPRE_Int HYPRE_StructFlexGMRESSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructFlexGMRESSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructFlexGMRESGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );
HYPRE_Int HYPRE_StructFlexGMRESSetModifyPC ( HYPRE_StructSolver solver , HYPRE_PtrToModifyPCFcn modify_pc );

/* HYPRE_struct_gmres.c */
HYPRE_Int HYPRE_StructGMRESCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructGMRESDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructGMRESSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructGMRESSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructGMRESSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructGMRESSetAbsoluteTol ( HYPRE_StructSolver solver , double atol );
HYPRE_Int HYPRE_StructGMRESSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructGMRESSetKDim ( HYPRE_StructSolver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_StructGMRESSetPrecond ( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
HYPRE_Int HYPRE_StructGMRESSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructGMRESSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructGMRESGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructGMRESGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_hybrid.c */
HYPRE_Int HYPRE_StructHybridCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructHybridDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructHybridSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructHybridSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructHybridSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructHybridSetConvergenceTol ( HYPRE_StructSolver solver , double cf_tol );
HYPRE_Int HYPRE_StructHybridSetDSCGMaxIter ( HYPRE_StructSolver solver , HYPRE_Int dscg_max_its );
HYPRE_Int HYPRE_StructHybridSetPCGMaxIter ( HYPRE_StructSolver solver , HYPRE_Int pcg_max_its );
HYPRE_Int HYPRE_StructHybridSetPCGAbsoluteTolFactor ( HYPRE_StructSolver solver , double pcg_atolf );
HYPRE_Int HYPRE_StructHybridSetTwoNorm ( HYPRE_StructSolver solver , HYPRE_Int two_norm );
HYPRE_Int HYPRE_StructHybridSetStopCrit ( HYPRE_StructSolver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_StructHybridSetRelChange ( HYPRE_StructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_StructHybridSetSolverType ( HYPRE_StructSolver solver , HYPRE_Int solver_type );
HYPRE_Int HYPRE_StructHybridSetKDim ( HYPRE_StructSolver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_StructHybridSetPrecond ( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
HYPRE_Int HYPRE_StructHybridSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructHybridSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructHybridGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_its );
HYPRE_Int HYPRE_StructHybridGetDSCGNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *dscg_num_its );
HYPRE_Int HYPRE_StructHybridGetPCGNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *pcg_num_its );
HYPRE_Int HYPRE_StructHybridGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_int.c */
HYPRE_Int hypre_StructVectorSetRandomValues ( hypre_StructVector *vector , HYPRE_Int seed );
HYPRE_Int hypre_StructSetRandomValues ( void *v , HYPRE_Int seed );
HYPRE_Int HYPRE_StructSetupInterpreter ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_StructSetupMatvec ( HYPRE_MatvecFunctions *mv );

/* HYPRE_struct_jacobi.c */
HYPRE_Int HYPRE_StructJacobiCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructJacobiDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructJacobiSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructJacobiSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructJacobiSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructJacobiGetTol ( HYPRE_StructSolver solver , double *tol );
HYPRE_Int HYPRE_StructJacobiSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructJacobiGetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_StructJacobiSetZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructJacobiGetZeroGuess ( HYPRE_StructSolver solver , HYPRE_Int *zeroguess );
HYPRE_Int HYPRE_StructJacobiSetNonZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructJacobiGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructJacobiGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_lgmres.c */
HYPRE_Int HYPRE_StructLGMRESCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructLGMRESDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructLGMRESSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructLGMRESSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructLGMRESSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructLGMRESSetAbsoluteTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructLGMRESSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructLGMRESSetKDim ( HYPRE_StructSolver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_StructLGMRESSetAugDim ( HYPRE_StructSolver solver , HYPRE_Int aug_dim );
HYPRE_Int HYPRE_StructLGMRESSetPrecond ( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
HYPRE_Int HYPRE_StructLGMRESSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructLGMRESSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructLGMRESGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructLGMRESGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_pcg.c */
HYPRE_Int HYPRE_StructPCGCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructPCGDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructPCGSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructPCGSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructPCGSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructPCGSetAbsoluteTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructPCGSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructPCGSetTwoNorm ( HYPRE_StructSolver solver , HYPRE_Int two_norm );
HYPRE_Int HYPRE_StructPCGSetRelChange ( HYPRE_StructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_StructPCGSetPrecond ( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
HYPRE_Int HYPRE_StructPCGSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructPCGSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructPCGGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructPCGGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );
HYPRE_Int HYPRE_StructDiagScaleSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructDiagScale ( HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx );

/* HYPRE_struct_pfmg.c */
HYPRE_Int HYPRE_StructPFMGCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructPFMGDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructPFMGSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructPFMGSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructPFMGSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructPFMGGetTol ( HYPRE_StructSolver solver , double *tol );
HYPRE_Int HYPRE_StructPFMGSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructPFMGGetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_StructPFMGSetMaxLevels ( HYPRE_StructSolver solver , HYPRE_Int max_levels );
HYPRE_Int HYPRE_StructPFMGGetMaxLevels ( HYPRE_StructSolver solver , HYPRE_Int *max_levels );
HYPRE_Int HYPRE_StructPFMGSetRelChange ( HYPRE_StructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_StructPFMGGetRelChange ( HYPRE_StructSolver solver , HYPRE_Int *rel_change );
HYPRE_Int HYPRE_StructPFMGSetZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructPFMGGetZeroGuess ( HYPRE_StructSolver solver , HYPRE_Int *zeroguess );
HYPRE_Int HYPRE_StructPFMGSetNonZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructPFMGSetRelaxType ( HYPRE_StructSolver solver , HYPRE_Int relax_type );
HYPRE_Int HYPRE_StructPFMGGetRelaxType ( HYPRE_StructSolver solver , HYPRE_Int *relax_type );
HYPRE_Int HYPRE_StructPFMGSetJacobiWeight ( HYPRE_StructSolver solver , double weight );
HYPRE_Int HYPRE_StructPFMGGetJacobiWeight ( HYPRE_StructSolver solver , double *weight );
HYPRE_Int HYPRE_StructPFMGSetRAPType ( HYPRE_StructSolver solver , HYPRE_Int rap_type );
HYPRE_Int HYPRE_StructPFMGGetRAPType ( HYPRE_StructSolver solver , HYPRE_Int *rap_type );
HYPRE_Int HYPRE_StructPFMGSetNumPreRelax ( HYPRE_StructSolver solver , HYPRE_Int num_pre_relax );
HYPRE_Int HYPRE_StructPFMGGetNumPreRelax ( HYPRE_StructSolver solver , HYPRE_Int *num_pre_relax );
HYPRE_Int HYPRE_StructPFMGSetNumPostRelax ( HYPRE_StructSolver solver , HYPRE_Int num_post_relax );
HYPRE_Int HYPRE_StructPFMGGetNumPostRelax ( HYPRE_StructSolver solver , HYPRE_Int *num_post_relax );
HYPRE_Int HYPRE_StructPFMGSetSkipRelax ( HYPRE_StructSolver solver , HYPRE_Int skip_relax );
HYPRE_Int HYPRE_StructPFMGGetSkipRelax ( HYPRE_StructSolver solver , HYPRE_Int *skip_relax );
HYPRE_Int HYPRE_StructPFMGSetDxyz ( HYPRE_StructSolver solver , double *dxyz );
HYPRE_Int HYPRE_StructPFMGSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructPFMGGetLogging ( HYPRE_StructSolver solver , HYPRE_Int *logging );
HYPRE_Int HYPRE_StructPFMGSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructPFMGGetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int *print_level );
HYPRE_Int HYPRE_StructPFMGGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructPFMGGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_smg.c */
HYPRE_Int HYPRE_StructSMGCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructSMGDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructSMGSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructSMGSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructSMGSetMemoryUse ( HYPRE_StructSolver solver , HYPRE_Int memory_use );
HYPRE_Int HYPRE_StructSMGGetMemoryUse ( HYPRE_StructSolver solver , HYPRE_Int *memory_use );
HYPRE_Int HYPRE_StructSMGSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructSMGGetTol ( HYPRE_StructSolver solver , double *tol );
HYPRE_Int HYPRE_StructSMGSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructSMGGetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int *max_iter );
HYPRE_Int HYPRE_StructSMGSetRelChange ( HYPRE_StructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_StructSMGGetRelChange ( HYPRE_StructSolver solver , HYPRE_Int *rel_change );
HYPRE_Int HYPRE_StructSMGSetZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructSMGGetZeroGuess ( HYPRE_StructSolver solver , HYPRE_Int *zeroguess );
HYPRE_Int HYPRE_StructSMGSetNonZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructSMGSetNumPreRelax ( HYPRE_StructSolver solver , HYPRE_Int num_pre_relax );
HYPRE_Int HYPRE_StructSMGGetNumPreRelax ( HYPRE_StructSolver solver , HYPRE_Int *num_pre_relax );
HYPRE_Int HYPRE_StructSMGSetNumPostRelax ( HYPRE_StructSolver solver , HYPRE_Int num_post_relax );
HYPRE_Int HYPRE_StructSMGGetNumPostRelax ( HYPRE_StructSolver solver , HYPRE_Int *num_post_relax );
HYPRE_Int HYPRE_StructSMGSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructSMGGetLogging ( HYPRE_StructSolver solver , HYPRE_Int *logging );
HYPRE_Int HYPRE_StructSMGSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructSMGGetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int *print_level );
HYPRE_Int HYPRE_StructSMGGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructSMGGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_sparse_msg.c */
HYPRE_Int HYPRE_StructSparseMSGCreate ( MPI_Comm comm , HYPRE_StructSolver *solver );
HYPRE_Int HYPRE_StructSparseMSGDestroy ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructSparseMSGSetup ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructSparseMSGSolve ( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
HYPRE_Int HYPRE_StructSparseMSGSetTol ( HYPRE_StructSolver solver , double tol );
HYPRE_Int HYPRE_StructSparseMSGSetMaxIter ( HYPRE_StructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_StructSparseMSGSetJump ( HYPRE_StructSolver solver , HYPRE_Int jump );
HYPRE_Int HYPRE_StructSparseMSGSetRelChange ( HYPRE_StructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_StructSparseMSGSetZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructSparseMSGSetNonZeroGuess ( HYPRE_StructSolver solver );
HYPRE_Int HYPRE_StructSparseMSGSetRelaxType ( HYPRE_StructSolver solver , HYPRE_Int relax_type );
HYPRE_Int HYPRE_StructSparseMSGSetJacobiWeight ( HYPRE_StructSolver solver , double weight );
HYPRE_Int HYPRE_StructSparseMSGSetNumPreRelax ( HYPRE_StructSolver solver , HYPRE_Int num_pre_relax );
HYPRE_Int HYPRE_StructSparseMSGSetNumPostRelax ( HYPRE_StructSolver solver , HYPRE_Int num_post_relax );
HYPRE_Int HYPRE_StructSparseMSGSetNumFineRelax ( HYPRE_StructSolver solver , HYPRE_Int num_fine_relax );
HYPRE_Int HYPRE_StructSparseMSGSetLogging ( HYPRE_StructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_StructSparseMSGSetPrintLevel ( HYPRE_StructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_StructSparseMSGGetNumIterations ( HYPRE_StructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm ( HYPRE_StructSolver solver , double *norm );

/* jacobi.c */
void *hypre_JacobiCreate ( MPI_Comm comm );
HYPRE_Int hypre_JacobiDestroy ( void *jacobi_vdata );
HYPRE_Int hypre_JacobiSetup ( void *jacobi_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_JacobiSolve ( void *jacobi_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_JacobiSetTol ( void *jacobi_vdata , double tol );
HYPRE_Int hypre_JacobiGetTol ( void *jacobi_vdata , double *tol );
HYPRE_Int hypre_JacobiSetMaxIter ( void *jacobi_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_JacobiGetMaxIter ( void *jacobi_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_JacobiSetZeroGuess ( void *jacobi_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_JacobiGetZeroGuess ( void *jacobi_vdata , HYPRE_Int *zero_guess );
HYPRE_Int hypre_JacobiGetNumIterations ( void *jacobi_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_JacobiSetTempVec ( void *jacobi_vdata , hypre_StructVector *t );
HYPRE_Int hypre_JacobiGetFinalRelativeResidualNorm ( void *jacobi_vdata , double *norm );

/* pcg_struct.c */
char *hypre_StructKrylovCAlloc ( HYPRE_Int count , HYPRE_Int elt_size );
HYPRE_Int hypre_StructKrylovFree ( char *ptr );
void *hypre_StructKrylovCreateVector ( void *vvector );
void *hypre_StructKrylovCreateVectorArray ( HYPRE_Int n , void *vvector );
HYPRE_Int hypre_StructKrylovDestroyVector ( void *vvector );
void *hypre_StructKrylovMatvecCreate ( void *A , void *x );
HYPRE_Int hypre_StructKrylovMatvec ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
HYPRE_Int hypre_StructKrylovMatvecDestroy ( void *matvec_data );
double hypre_StructKrylovInnerProd ( void *x , void *y );
HYPRE_Int hypre_StructKrylovCopyVector ( void *x , void *y );
HYPRE_Int hypre_StructKrylovClearVector ( void *x );
HYPRE_Int hypre_StructKrylovScaleVector ( double alpha , void *x );
HYPRE_Int hypre_StructKrylovAxpy ( double alpha , void *x , void *y );
HYPRE_Int hypre_StructKrylovIdentitySetup ( void *vdata , void *A , void *b , void *x );
HYPRE_Int hypre_StructKrylovIdentity ( void *vdata , void *A , void *b , void *x );
HYPRE_Int hypre_StructKrylovCommInfo ( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs );

/* pfmg2_setup_rap.c */
hypre_StructMatrix *hypre_PFMG2CreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_PFMG2BuildRAPSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );

/* pfmg3_setup_rap.c */
hypre_StructMatrix *hypre_PFMG3CreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_PFMG3BuildRAPSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1 ( HYPRE_Int ci , HYPRE_Int fi , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );

/* pfmg.c */
void *hypre_PFMGCreate ( MPI_Comm comm );
HYPRE_Int hypre_PFMGDestroy ( void *pfmg_vdata );
HYPRE_Int hypre_PFMGSetTol ( void *pfmg_vdata , double tol );
HYPRE_Int hypre_PFMGGetTol ( void *pfmg_vdata , double *tol );
HYPRE_Int hypre_PFMGSetMaxIter ( void *pfmg_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGGetMaxIter ( void *pfmg_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_PFMGSetMaxLevels ( void *pfmg_vdata , HYPRE_Int max_levels );
HYPRE_Int hypre_PFMGGetMaxLevels ( void *pfmg_vdata , HYPRE_Int *max_levels );
HYPRE_Int hypre_PFMGSetRelChange ( void *pfmg_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_PFMGGetRelChange ( void *pfmg_vdata , HYPRE_Int *rel_change );
HYPRE_Int hypre_PFMGSetZeroGuess ( void *pfmg_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGGetZeroGuess ( void *pfmg_vdata , HYPRE_Int *zero_guess );
HYPRE_Int hypre_PFMGSetRelaxType ( void *pfmg_vdata , HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGGetRelaxType ( void *pfmg_vdata , HYPRE_Int *relax_type );
HYPRE_Int hypre_PFMGSetJacobiWeight ( void *pfmg_vdata , double weight );
HYPRE_Int hypre_PFMGGetJacobiWeight ( void *pfmg_vdata , double *weight );
HYPRE_Int hypre_PFMGSetRAPType ( void *pfmg_vdata , HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGGetRAPType ( void *pfmg_vdata , HYPRE_Int *rap_type );
HYPRE_Int hypre_PFMGSetNumPreRelax ( void *pfmg_vdata , HYPRE_Int num_pre_relax );
HYPRE_Int hypre_PFMGGetNumPreRelax ( void *pfmg_vdata , HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_PFMGSetNumPostRelax ( void *pfmg_vdata , HYPRE_Int num_post_relax );
HYPRE_Int hypre_PFMGGetNumPostRelax ( void *pfmg_vdata , HYPRE_Int *num_post_relax );
HYPRE_Int hypre_PFMGSetSkipRelax ( void *pfmg_vdata , HYPRE_Int skip_relax );
HYPRE_Int hypre_PFMGGetSkipRelax ( void *pfmg_vdata , HYPRE_Int *skip_relax );
HYPRE_Int hypre_PFMGSetDxyz ( void *pfmg_vdata , double *dxyz );
HYPRE_Int hypre_PFMGSetLogging ( void *pfmg_vdata , HYPRE_Int logging );
HYPRE_Int hypre_PFMGGetLogging ( void *pfmg_vdata , HYPRE_Int *logging );
HYPRE_Int hypre_PFMGSetPrintLevel ( void *pfmg_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_PFMGGetPrintLevel ( void *pfmg_vdata , HYPRE_Int *print_level );
HYPRE_Int hypre_PFMGGetNumIterations ( void *pfmg_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_PFMGPrintLogging ( void *pfmg_vdata , HYPRE_Int myid );
HYPRE_Int hypre_PFMGGetFinalRelativeResidualNorm ( void *pfmg_vdata , double *relative_residual_norm );

/* pfmg_relax.c */
void *hypre_PFMGRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_PFMGRelaxDestroy ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelax ( void *pfmg_relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelaxSetup ( void *pfmg_relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_PFMGRelaxSetType ( void *pfmg_relax_vdata , HYPRE_Int relax_type );
HYPRE_Int hypre_PFMGRelaxSetJacobiWeight ( void *pfmg_relax_vdata , double weight );
HYPRE_Int hypre_PFMGRelaxSetPreRelax ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetPostRelax ( void *pfmg_relax_vdata );
HYPRE_Int hypre_PFMGRelaxSetTol ( void *pfmg_relax_vdata , double tol );
HYPRE_Int hypre_PFMGRelaxSetMaxIter ( void *pfmg_relax_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_PFMGRelaxSetZeroGuess ( void *pfmg_relax_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_PFMGRelaxSetTempVec ( void *pfmg_relax_vdata , hypre_StructVector *t );

/* pfmg_setup.c */
HYPRE_Int hypre_PFMGSetup ( void *pfmg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_PFMGComputeDxyz ( hypre_StructMatrix *A , double *dxyz , double *mean , double *deviation );
HYPRE_Int hypre_ZeroDiagonal ( hypre_StructMatrix *A );

/* pfmg_setup_interp.c */
hypre_StructMatrix *hypre_PFMGCreateInterpOp ( hypre_StructMatrix *A , hypre_StructGrid *cgrid , HYPRE_Int cdir , HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp ( hypre_StructMatrix *A , HYPRE_Int cdir , hypre_Index findex , hypre_Index stride , hypre_StructMatrix *P , HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupInterpOp_CC0 ( HYPRE_Int i , hypre_StructMatrix *A , hypre_Box *A_dbox , HYPRE_Int cdir , hypre_Index stride , hypre_Index stridec , hypre_Index start , hypre_IndexRef startc , hypre_Index loop_size , hypre_Box *P_dbox , HYPRE_Int Pstenc0 , HYPRE_Int Pstenc1 , double *Pp0 , double *Pp1 , HYPRE_Int rap_type , HYPRE_Int si0 , HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC1 ( HYPRE_Int i , hypre_StructMatrix *A , hypre_Box *A_dbox , HYPRE_Int cdir , hypre_Index stride , hypre_Index stridec , hypre_Index start , hypre_IndexRef startc , hypre_Index loop_size , hypre_Box *P_dbox , HYPRE_Int Pstenc0 , HYPRE_Int Pstenc1 , double *Pp0 , double *Pp1 , HYPRE_Int rap_type , HYPRE_Int si0 , HYPRE_Int si1 );
HYPRE_Int hypre_PFMGSetupInterpOp_CC2 ( HYPRE_Int i , hypre_StructMatrix *A , hypre_Box *A_dbox , HYPRE_Int cdir , hypre_Index stride , hypre_Index stridec , hypre_Index start , hypre_IndexRef startc , hypre_Index loop_size , hypre_Box *P_dbox , HYPRE_Int Pstenc0 , HYPRE_Int Pstenc1 , double *Pp0 , double *Pp1 , HYPRE_Int rap_type , HYPRE_Int si0 , HYPRE_Int si1 );

/* pfmg_setup_rap5.c */
hypre_StructMatrix *hypre_PFMGCreateCoarseOp5 ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_PFMGBuildCoarseOp5 ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_onebox_CC0 ( HYPRE_Int fi , HYPRE_Int ci , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_onebox_CC1 ( HYPRE_Int fi , HYPRE_Int ci , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
HYPRE_Int hypre_PFMGBuildCoarseOp5_onebox_CC2 ( HYPRE_Int fi , HYPRE_Int ci , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );

/* pfmg_setup_rap7.c */
hypre_StructMatrix *hypre_PFMGCreateCoarseOp7 ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_PFMGBuildCoarseOp7 ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );

/* pfmg_setup_rap.c */
hypre_StructMatrix *hypre_PFMGCreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir , HYPRE_Int rap_type );
HYPRE_Int hypre_PFMGSetupRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , HYPRE_Int rap_type , hypre_StructMatrix *Ac );

/* pfmg_solve.c */
HYPRE_Int hypre_PFMGSolve ( void *pfmg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* point_relax.c */
void *hypre_PointRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_PointRelaxDestroy ( void *relax_vdata );
HYPRE_Int hypre_PointRelaxSetup ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_PointRelax ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_PointRelax_core0 ( void *relax_vdata , hypre_StructMatrix *A , HYPRE_Int constant_coefficient , hypre_Box *compute_box , double *bp , double *xp , double *tp , HYPRE_Int boxarray_id , hypre_Box *A_data_box , hypre_Box *b_data_box , hypre_Box *x_data_box , hypre_Box *t_data_box , hypre_IndexRef stride );
HYPRE_Int hypre_PointRelax_core12 ( void *relax_vdata , hypre_StructMatrix *A , HYPRE_Int constant_coefficient , hypre_Box *compute_box , double *bp , double *xp , double *tp , HYPRE_Int boxarray_id , hypre_Box *A_data_box , hypre_Box *b_data_box , hypre_Box *x_data_box , hypre_Box *t_data_box , hypre_IndexRef stride );
HYPRE_Int hypre_PointRelaxSetTol ( void *relax_vdata , double tol );
HYPRE_Int hypre_PointRelaxGetTol ( void *relax_vdata , double *tol );
HYPRE_Int hypre_PointRelaxSetMaxIter ( void *relax_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_PointRelaxGetMaxIter ( void *relax_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_PointRelaxSetZeroGuess ( void *relax_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_PointRelaxGetZeroGuess ( void *relax_vdata , HYPRE_Int *zero_guess );
HYPRE_Int hypre_PointRelaxGetNumIterations ( void *relax_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_PointRelaxSetWeight ( void *relax_vdata , double weight );
HYPRE_Int hypre_PointRelaxSetNumPointsets ( void *relax_vdata , HYPRE_Int num_pointsets );
HYPRE_Int hypre_PointRelaxSetPointset ( void *relax_vdata , HYPRE_Int pointset , HYPRE_Int pointset_size , hypre_Index pointset_stride , hypre_Index *pointset_indices );
HYPRE_Int hypre_PointRelaxSetPointsetRank ( void *relax_vdata , HYPRE_Int pointset , HYPRE_Int pointset_rank );
HYPRE_Int hypre_PointRelaxSetTempVec ( void *relax_vdata , hypre_StructVector *t );
HYPRE_Int hypre_PointRelaxGetFinalRelativeResidualNorm ( void *relax_vdata , double *norm );
HYPRE_Int hypre_relax_wtx ( void *relax_vdata , HYPRE_Int pointset , hypre_StructVector *t , hypre_StructVector *x );
HYPRE_Int hypre_relax_copy ( void *relax_vdata , HYPRE_Int pointset , hypre_StructVector *t , hypre_StructVector *x );

/* red_black_constantcoef_gs.c */
HYPRE_Int hypre_RedBlackConstantCoefGS ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* red_black_gs.c */
void *hypre_RedBlackGSCreate ( MPI_Comm comm );
HYPRE_Int hypre_RedBlackGSDestroy ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetup ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGS ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_RedBlackGSSetTol ( void *relax_vdata , double tol );
HYPRE_Int hypre_RedBlackGSSetMaxIter ( void *relax_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_RedBlackGSSetZeroGuess ( void *relax_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_RedBlackGSSetStartRed ( void *relax_vdata );
HYPRE_Int hypre_RedBlackGSSetStartBlack ( void *relax_vdata );

/* semi.c */
HYPRE_Int hypre_StructInterpAssemble ( hypre_StructMatrix *A , hypre_StructMatrix *P , HYPRE_Int P_stored_as_transpose , HYPRE_Int cdir , hypre_Index index , hypre_Index stride );

/* semi_interp.c */
void *hypre_SemiInterpCreate ( void );
HYPRE_Int hypre_SemiInterpSetup ( void *interp_vdata , hypre_StructMatrix *P , HYPRE_Int P_stored_as_transpose , hypre_StructVector *xc , hypre_StructVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
HYPRE_Int hypre_SemiInterp ( void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e );
HYPRE_Int hypre_SemiInterpDestroy ( void *interp_vdata );

/* semi_restrict.c */
void *hypre_SemiRestrictCreate ( void );
HYPRE_Int hypre_SemiRestrictSetup ( void *restrict_vdata , hypre_StructMatrix *R , HYPRE_Int R_stored_as_transpose , hypre_StructVector *r , hypre_StructVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
HYPRE_Int hypre_SemiRestrict ( void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc );
HYPRE_Int hypre_SemiRestrictDestroy ( void *restrict_vdata );

/* semi_setup_rap.c */
hypre_StructMatrix *hypre_SemiCreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir , HYPRE_Int P_stored_as_transpose );
HYPRE_Int hypre_SemiBuildRAP ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , HYPRE_Int P_stored_as_transpose , hypre_StructMatrix *RAP );

/* smg2_setup_rap.c */
hypre_StructMatrix *hypre_SMG2CreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMG2BuildRAPSym ( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
HYPRE_Int hypre_SMG2BuildRAPNoSym ( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicSym ( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
HYPRE_Int hypre_SMG2RAPPeriodicNoSym ( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );

/* smg3_setup_rap.c */
hypre_StructMatrix *hypre_SMG3CreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMG3BuildRAPSym ( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
HYPRE_Int hypre_SMG3BuildRAPNoSym ( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicSym ( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
HYPRE_Int hypre_SMG3RAPPeriodicNoSym ( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );

/* smg_axpy.c */
HYPRE_Int hypre_SMGAxpy ( double alpha , hypre_StructVector *x , hypre_StructVector *y , hypre_Index base_index , hypre_Index base_stride );

/* smg.c */
void *hypre_SMGCreate ( MPI_Comm comm );
HYPRE_Int hypre_SMGDestroy ( void *smg_vdata );
HYPRE_Int hypre_SMGSetMemoryUse ( void *smg_vdata , HYPRE_Int memory_use );
HYPRE_Int hypre_SMGGetMemoryUse ( void *smg_vdata , HYPRE_Int *memory_use );
HYPRE_Int hypre_SMGSetTol ( void *smg_vdata , double tol );
HYPRE_Int hypre_SMGGetTol ( void *smg_vdata , double *tol );
HYPRE_Int hypre_SMGSetMaxIter ( void *smg_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_SMGGetMaxIter ( void *smg_vdata , HYPRE_Int *max_iter );
HYPRE_Int hypre_SMGSetRelChange ( void *smg_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_SMGGetRelChange ( void *smg_vdata , HYPRE_Int *rel_change );
HYPRE_Int hypre_SMGSetZeroGuess ( void *smg_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGGetZeroGuess ( void *smg_vdata , HYPRE_Int *zero_guess );
HYPRE_Int hypre_SMGSetNumPreRelax ( void *smg_vdata , HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGGetNumPreRelax ( void *smg_vdata , HYPRE_Int *num_pre_relax );
HYPRE_Int hypre_SMGSetNumPostRelax ( void *smg_vdata , HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGGetNumPostRelax ( void *smg_vdata , HYPRE_Int *num_post_relax );
HYPRE_Int hypre_SMGSetBase ( void *smg_vdata , hypre_Index base_index , hypre_Index base_stride );
HYPRE_Int hypre_SMGSetLogging ( void *smg_vdata , HYPRE_Int logging );
HYPRE_Int hypre_SMGGetLogging ( void *smg_vdata , HYPRE_Int *logging );
HYPRE_Int hypre_SMGSetPrintLevel ( void *smg_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_SMGGetPrintLevel ( void *smg_vdata , HYPRE_Int *print_level );
HYPRE_Int hypre_SMGGetNumIterations ( void *smg_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_SMGPrintLogging ( void *smg_vdata , HYPRE_Int myid );
HYPRE_Int hypre_SMGGetFinalRelativeResidualNorm ( void *smg_vdata , double *relative_residual_norm );
HYPRE_Int hypre_SMGSetStructVectorConstantValues ( hypre_StructVector *vector , double values , hypre_BoxArray *box_array , hypre_Index stride );

/* smg_relax.c */
void *hypre_SMGRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_SMGRelaxDestroyTempVec ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyARem ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroyASol ( void *relax_vdata );
HYPRE_Int hypre_SMGRelaxDestroy ( void *relax_vdata );
HYPRE_Int hypre_SMGRelax ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetup ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupTempVec ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupARem ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetupASol ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
HYPRE_Int hypre_SMGRelaxSetTempVec ( void *relax_vdata , hypre_StructVector *temp_vec );
HYPRE_Int hypre_SMGRelaxSetMemoryUse ( void *relax_vdata , HYPRE_Int memory_use );
HYPRE_Int hypre_SMGRelaxSetTol ( void *relax_vdata , double tol );
HYPRE_Int hypre_SMGRelaxSetMaxIter ( void *relax_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_SMGRelaxSetZeroGuess ( void *relax_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_SMGRelaxSetNumSpaces ( void *relax_vdata , HYPRE_Int num_spaces );
HYPRE_Int hypre_SMGRelaxSetNumPreSpaces ( void *relax_vdata , HYPRE_Int num_pre_spaces );
HYPRE_Int hypre_SMGRelaxSetNumRegSpaces ( void *relax_vdata , HYPRE_Int num_reg_spaces );
HYPRE_Int hypre_SMGRelaxSetSpace ( void *relax_vdata , HYPRE_Int i , HYPRE_Int space_index , HYPRE_Int space_stride );
HYPRE_Int hypre_SMGRelaxSetRegSpaceRank ( void *relax_vdata , HYPRE_Int i , HYPRE_Int reg_space_rank );
HYPRE_Int hypre_SMGRelaxSetPreSpaceRank ( void *relax_vdata , HYPRE_Int i , HYPRE_Int pre_space_rank );
HYPRE_Int hypre_SMGRelaxSetBase ( void *relax_vdata , hypre_Index base_index , hypre_Index base_stride );
HYPRE_Int hypre_SMGRelaxSetNumPreRelax ( void *relax_vdata , HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SMGRelaxSetNumPostRelax ( void *relax_vdata , HYPRE_Int num_post_relax );
HYPRE_Int hypre_SMGRelaxSetNewMatrixStencil ( void *relax_vdata , hypre_StructStencil *diff_stencil );
HYPRE_Int hypre_SMGRelaxSetupBaseBoxArray ( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* smg_residual.c */
void *hypre_SMGResidualCreate ( void );
HYPRE_Int hypre_SMGResidualSetup ( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
HYPRE_Int hypre_SMGResidual ( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
HYPRE_Int hypre_SMGResidualSetBase ( void *residual_vdata , hypre_Index base_index , hypre_Index base_stride );
HYPRE_Int hypre_SMGResidualDestroy ( void *residual_vdata );

/* smg_residual_unrolled.c */
void *hypre_SMGResidualCreate ( void );
HYPRE_Int hypre_SMGResidualSetup ( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
HYPRE_Int hypre_SMGResidual ( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
HYPRE_Int hypre_SMGResidualSetBase ( void *residual_vdata , hypre_Index base_index , hypre_Index base_stride );
HYPRE_Int hypre_SMGResidualDestroy ( void *residual_vdata );

/* smg_setup.c */
HYPRE_Int hypre_SMGSetup ( void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* smg_setup_interp.c */
hypre_StructMatrix *hypre_SMGCreateInterpOp ( hypre_StructMatrix *A , hypre_StructGrid *cgrid , HYPRE_Int cdir );
HYPRE_Int hypre_SMGSetupInterpOp ( void *relax_data , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x , hypre_StructMatrix *PT , HYPRE_Int cdir , hypre_Index cindex , hypre_Index findex , hypre_Index stride );

/* smg_setup_rap.c */
hypre_StructMatrix *hypre_SMGCreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructGrid *coarse_grid );
HYPRE_Int hypre_SMGSetupRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride );

/* smg_setup_restrict.c */
hypre_StructMatrix *hypre_SMGCreateRestrictOp ( hypre_StructMatrix *A , hypre_StructGrid *cgrid , HYPRE_Int cdir );
HYPRE_Int hypre_SMGSetupRestrictOp ( hypre_StructMatrix *A , hypre_StructMatrix *R , hypre_StructVector *temp_vec , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride );

/* smg_solve.c */
HYPRE_Int hypre_SMGSolve ( void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* sparse_msg2_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSG2CreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSG2BuildRAPSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG2BuildRAPNoSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );

/* sparse_msg3_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSG3CreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSG3BuildRAPSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );
HYPRE_Int hypre_SparseMSG3BuildRAPNoSym ( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );

/* sparse_msg.c */
void *hypre_SparseMSGCreate ( MPI_Comm comm );
HYPRE_Int hypre_SparseMSGDestroy ( void *smsg_vdata );
HYPRE_Int hypre_SparseMSGSetTol ( void *smsg_vdata , double tol );
HYPRE_Int hypre_SparseMSGSetMaxIter ( void *smsg_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_SparseMSGSetJump ( void *smsg_vdata , HYPRE_Int jump );
HYPRE_Int hypre_SparseMSGSetRelChange ( void *smsg_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_SparseMSGSetZeroGuess ( void *smsg_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_SparseMSGSetRelaxType ( void *smsg_vdata , HYPRE_Int relax_type );
HYPRE_Int hypre_SparseMSGSetJacobiWeight ( void *smsg_vdata , double weight );
HYPRE_Int hypre_SparseMSGSetNumPreRelax ( void *smsg_vdata , HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SparseMSGSetNumPostRelax ( void *smsg_vdata , HYPRE_Int num_post_relax );
HYPRE_Int hypre_SparseMSGSetNumFineRelax ( void *smsg_vdata , HYPRE_Int num_fine_relax );
HYPRE_Int hypre_SparseMSGSetLogging ( void *smsg_vdata , HYPRE_Int logging );
HYPRE_Int hypre_SparseMSGSetPrintLevel ( void *smsg_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_SparseMSGGetNumIterations ( void *smsg_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_SparseMSGPrintLogging ( void *smsg_vdata , HYPRE_Int myid );
HYPRE_Int hypre_SparseMSGGetFinalRelativeResidualNorm ( void *smsg_vdata , double *relative_residual_norm );

/* sparse_msg_filter.c */
HYPRE_Int hypre_SparseMSGFilterSetup ( hypre_StructMatrix *A , HYPRE_Int *num_grids , HYPRE_Int lx , HYPRE_Int ly , HYPRE_Int lz , HYPRE_Int jump , hypre_StructVector *visitx , hypre_StructVector *visity , hypre_StructVector *visitz );
HYPRE_Int hypre_SparseMSGFilter ( hypre_StructVector *visit , hypre_StructVector *e , HYPRE_Int lx , HYPRE_Int ly , HYPRE_Int lz , HYPRE_Int jump );

/* sparse_msg_interp.c */
void *hypre_SparseMSGInterpCreate ( void );
HYPRE_Int hypre_SparseMSGInterpSetup ( void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride , hypre_Index strideP );
HYPRE_Int hypre_SparseMSGInterp ( void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e );
HYPRE_Int hypre_SparseMSGInterpDestroy ( void *interp_vdata );

/* sparse_msg_restrict.c */
void *hypre_SparseMSGRestrictCreate ( void );
HYPRE_Int hypre_SparseMSGRestrictSetup ( void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride , hypre_Index strideR );
HYPRE_Int hypre_SparseMSGRestrict ( void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc );
HYPRE_Int hypre_SparseMSGRestrictDestroy ( void *restrict_vdata );

/* sparse_msg_setup.c */
HYPRE_Int hypre_SparseMSGSetup ( void *smsg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* sparse_msg_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSGCreateRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_SparseMSGSetupRAPOp ( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *Ac );

/* sparse_msg_solve.c */
HYPRE_Int hypre_SparseMSGSolve ( void *smsg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

#ifdef __cplusplus
}
#endif

#endif

