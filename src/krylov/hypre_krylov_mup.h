
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

#ifndef HYPRE_KRYLOV_MUP_HEADER
#define HYPRE_KRYLOV_MUP_HEADER

#include "krylov.h"

#if defined (HYPRE_MIXED_PRECISION)

void *hypre_BiCGSTABCreate_flt( hypre_BiCGSTABFunctions *bicgstab_functions );
void *hypre_BiCGSTABCreate_dbl( hypre_BiCGSTABFunctions *bicgstab_functions );
void *hypre_BiCGSTABCreate_long_dbl( hypre_BiCGSTABFunctions *bicgstab_functions );
HYPRE_Int hypre_BiCGSTABDestroy_flt  ( void *bicgstab_vdata );
HYPRE_Int hypre_BiCGSTABDestroy_dbl  ( void *bicgstab_vdata );
HYPRE_Int hypre_BiCGSTABDestroy_long_dbl  ( void *bicgstab_vdata );
hypre_BiCGSTABFunctions *hypre_BiCGSTABFunctionsCreate_flt(
      void *(*CreateVector)  ( void *vvector ),
      HYPRE_Int  (*DestroyVector) ( void *vvector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int  (*Matvec)        ( void *matvec_data, hypre_float alpha, void *A,
                                    void *x, hypre_float beta, void *y ),
      HYPRE_Int  (*MatvecDestroy) ( void *matvec_data ),
      hypre_float (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int  (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int  (*ClearVector)   ( void *x ),
      HYPRE_Int  (*ScaleVector)   ( hypre_float alpha, void *x ),
      HYPRE_Int  (*Axpy)          ( hypre_float alpha, void *x, void *y ),
      HYPRE_Int  (*CommInfo)      ( void *A, HYPRE_Int *my_id,
                                    HYPRE_Int *num_procs ),
      HYPRE_Int  (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int  (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_BiCGSTABFunctions *hypre_BiCGSTABFunctionsCreate_dbl(
      void *(*CreateVector)  ( void *vvector ),
      HYPRE_Int  (*DestroyVector) ( void *vvector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int  (*Matvec)        ( void *matvec_data, hypre_double alpha, void *A,
                                    void *x, hypre_double beta, void *y ),
      HYPRE_Int  (*MatvecDestroy) ( void *matvec_data ),
      hypre_double (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int  (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int  (*ClearVector)   ( void *x ),
      HYPRE_Int  (*ScaleVector)   ( hypre_double alpha, void *x ),
      HYPRE_Int  (*Axpy)          ( hypre_double alpha, void *x, void *y ),
      HYPRE_Int  (*CommInfo)      ( void *A, HYPRE_Int *my_id,
                                    HYPRE_Int *num_procs ),
      HYPRE_Int  (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int  (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_BiCGSTABFunctions *hypre_BiCGSTABFunctionsCreate_long_dbl(
      void *(*CreateVector)  ( void *vvector ),
      HYPRE_Int  (*DestroyVector) ( void *vvector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int  (*Matvec)        ( void *matvec_data, hypre_long_double alpha, void *A,
                                    void *x, hypre_long_double beta, void *y ),
      HYPRE_Int  (*MatvecDestroy) ( void *matvec_data ),
      hypre_long_double (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int  (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int  (*ClearVector)   ( void *x ),
      HYPRE_Int  (*ScaleVector)   ( hypre_long_double alpha, void *x ),
      HYPRE_Int  (*Axpy)          ( hypre_long_double alpha, void *x, void *y ),
      HYPRE_Int  (*CommInfo)      ( void *A, HYPRE_Int *my_id,
                                    HYPRE_Int *num_procs ),
      HYPRE_Int  (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int  (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
HYPRE_Int hypre_BiCGSTABGetConverged_flt  ( void *bicgstab_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_BiCGSTABGetConverged_dbl  ( void *bicgstab_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_BiCGSTABGetConverged_long_dbl  ( void *bicgstab_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_BiCGSTABGetFinalRelativeResidualNorm_flt  ( void *bicgstab_vdata,
                                                       hypre_float *relative_residual_norm );
HYPRE_Int hypre_BiCGSTABGetFinalRelativeResidualNorm_dbl  ( void *bicgstab_vdata,
                                                       hypre_double *relative_residual_norm );
HYPRE_Int hypre_BiCGSTABGetFinalRelativeResidualNorm_long_dbl  ( void *bicgstab_vdata,
                                                       hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_BiCGSTABGetNumIterations_flt  ( void *bicgstab_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BiCGSTABGetNumIterations_dbl  ( void *bicgstab_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BiCGSTABGetNumIterations_long_dbl  ( void *bicgstab_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_BiCGSTABGetPrecond_flt  ( void *bicgstab_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_BiCGSTABGetPrecond_dbl  ( void *bicgstab_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_BiCGSTABGetPrecond_long_dbl  ( void *bicgstab_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_BiCGSTABGetResidual_flt  ( void *bicgstab_vdata, void **residual );
HYPRE_Int hypre_BiCGSTABGetResidual_dbl  ( void *bicgstab_vdata, void **residual );
HYPRE_Int hypre_BiCGSTABGetResidual_long_dbl  ( void *bicgstab_vdata, void **residual );
HYPRE_Int hypre_BiCGSTABSetAbsoluteTol_flt  ( void *bicgstab_vdata, hypre_float a_tol );
HYPRE_Int hypre_BiCGSTABSetAbsoluteTol_dbl  ( void *bicgstab_vdata, hypre_double a_tol );
HYPRE_Int hypre_BiCGSTABSetAbsoluteTol_long_dbl  ( void *bicgstab_vdata, hypre_long_double a_tol );
HYPRE_Int hypre_BiCGSTABSetConvergenceFactorTol_flt  ( void *bicgstab_vdata, hypre_float cf_tol );
HYPRE_Int hypre_BiCGSTABSetConvergenceFactorTol_dbl  ( void *bicgstab_vdata, hypre_double cf_tol );
HYPRE_Int hypre_BiCGSTABSetConvergenceFactorTol_long_dbl  ( void *bicgstab_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_BiCGSTABSetHybrid_flt  ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetHybrid_dbl  ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetHybrid_long_dbl  ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetLogging_flt  ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetLogging_dbl  ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetLogging_long_dbl  ( void *bicgstab_vdata, HYPRE_Int logging );
HYPRE_Int hypre_BiCGSTABSetMaxIter_flt  ( void *bicgstab_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_BiCGSTABSetMaxIter_dbl  ( void *bicgstab_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_BiCGSTABSetMaxIter_long_dbl  ( void *bicgstab_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_BiCGSTABSetMinIter_flt  ( void *bicgstab_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_BiCGSTABSetMinIter_dbl  ( void *bicgstab_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_BiCGSTABSetMinIter_long_dbl  ( void *bicgstab_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_BiCGSTABSetPrecond_flt  ( void *bicgstab_vdata, HYPRE_Int (*precond )(void*, void*,
                                                                                 void*,
                                                                                 void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_BiCGSTABSetPrecond_dbl  ( void *bicgstab_vdata, HYPRE_Int (*precond )(void*, void*,
                                                                                 void*,
                                                                                 void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_BiCGSTABSetPrecond_long_dbl  ( void *bicgstab_vdata, HYPRE_Int (*precond )(void*, void*,
                                                                                 void*,
                                                                                 void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_BiCGSTABSetPrintLevel_flt  ( void *bicgstab_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_BiCGSTABSetPrintLevel_dbl  ( void *bicgstab_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_BiCGSTABSetPrintLevel_long_dbl  ( void *bicgstab_vdata, HYPRE_Int print_level );
HYPRE_Int hypre_BiCGSTABSetStopCrit_flt  ( void *bicgstab_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_BiCGSTABSetStopCrit_dbl  ( void *bicgstab_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_BiCGSTABSetStopCrit_long_dbl  ( void *bicgstab_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_BiCGSTABSetTol_flt  ( void *bicgstab_vdata, hypre_float tol );
HYPRE_Int hypre_BiCGSTABSetTol_dbl  ( void *bicgstab_vdata, hypre_double tol );
HYPRE_Int hypre_BiCGSTABSetTol_long_dbl  ( void *bicgstab_vdata, hypre_long_double tol );
HYPRE_Int hypre_BiCGSTABSetup_flt  ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSetup_dbl  ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSetup_long_dbl  ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSolve_flt  ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSolve_dbl  ( void *bicgstab_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_BiCGSTABSolve_long_dbl  ( void *bicgstab_vdata, void *A, void *b, void *x );
void *hypre_CGNRCreate_flt( hypre_CGNRFunctions *cgnr_functions );
void *hypre_CGNRCreate_dbl( hypre_CGNRFunctions *cgnr_functions );
void *hypre_CGNRCreate_long_dbl( hypre_CGNRFunctions *cgnr_functions );
HYPRE_Int hypre_CGNRDestroy_flt  ( void *cgnr_vdata );
HYPRE_Int hypre_CGNRDestroy_dbl  ( void *cgnr_vdata );
HYPRE_Int hypre_CGNRDestroy_long_dbl  ( void *cgnr_vdata );
hypre_CGNRFunctions *hypre_CGNRFunctionsCreate_flt(
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_float alpha, void *A,
                                      void *x, hypre_float beta, void *y ),
      HYPRE_Int    (*MatvecT)       ( void *matvec_data, hypre_float alpha, void *A,
                                      void *x, hypre_float beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_float   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_float alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_float alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*PrecondT)      ( void *vdata, void *A, void *b, void *x )
   );
hypre_CGNRFunctions *hypre_CGNRFunctionsCreate_dbl(
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_double alpha, void *A,
                                      void *x, hypre_double beta, void *y ),
      HYPRE_Int    (*MatvecT)       ( void *matvec_data, hypre_double alpha, void *A,
                                      void *x, hypre_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*PrecondT)      ( void *vdata, void *A, void *b, void *x )
   );
hypre_CGNRFunctions *hypre_CGNRFunctionsCreate_long_dbl(
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_long_double alpha, void *A,
                                      void *x, hypre_long_double beta, void *y ),
      HYPRE_Int    (*MatvecT)       ( void *matvec_data, hypre_long_double alpha, void *A,
                                      void *x, hypre_long_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_long_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_long_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_long_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*PrecondT)      ( void *vdata, void *A, void *b, void *x )
   );
HYPRE_Int hypre_CGNRGetFinalRelativeResidualNorm_flt  ( void *cgnr_vdata,
                                                   hypre_float *relative_residual_norm );
HYPRE_Int hypre_CGNRGetFinalRelativeResidualNorm_dbl  ( void *cgnr_vdata,
                                                   hypre_double *relative_residual_norm );
HYPRE_Int hypre_CGNRGetFinalRelativeResidualNorm_long_dbl  ( void *cgnr_vdata,
                                                   hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_CGNRGetNumIterations_flt  ( void *cgnr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_CGNRGetNumIterations_dbl  ( void *cgnr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_CGNRGetNumIterations_long_dbl  ( void *cgnr_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_CGNRGetPrecond_flt  ( void *cgnr_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_CGNRGetPrecond_dbl  ( void *cgnr_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_CGNRGetPrecond_long_dbl  ( void *cgnr_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_CGNRSetLogging_flt  ( void *cgnr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_CGNRSetLogging_dbl  ( void *cgnr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_CGNRSetLogging_long_dbl  ( void *cgnr_vdata, HYPRE_Int logging );
HYPRE_Int hypre_CGNRSetMaxIter_flt  ( void *cgnr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_CGNRSetMaxIter_dbl  ( void *cgnr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_CGNRSetMaxIter_long_dbl  ( void *cgnr_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_CGNRSetMinIter_flt  ( void *cgnr_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_CGNRSetMinIter_dbl  ( void *cgnr_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_CGNRSetMinIter_long_dbl  ( void *cgnr_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_CGNRSetPrecond_flt  ( void *cgnr_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                         void*),
                                 HYPRE_Int (*precondT )(void*, void*, void*, void*), HYPRE_Int (*precond_setup )(void*, void*, void*,
                                       void*), void *precond_data );
HYPRE_Int hypre_CGNRSetPrecond_dbl  ( void *cgnr_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                         void*),
                                 HYPRE_Int (*precondT )(void*, void*, void*, void*), HYPRE_Int (*precond_setup )(void*, void*, void*,
                                       void*), void *precond_data );
HYPRE_Int hypre_CGNRSetPrecond_long_dbl  ( void *cgnr_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                         void*),
                                 HYPRE_Int (*precondT )(void*, void*, void*, void*), HYPRE_Int (*precond_setup )(void*, void*, void*,
                                       void*), void *precond_data );
HYPRE_Int hypre_CGNRSetStopCrit_flt  ( void *cgnr_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_CGNRSetStopCrit_dbl  ( void *cgnr_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_CGNRSetStopCrit_long_dbl  ( void *cgnr_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_CGNRSetTol_flt  ( void *cgnr_vdata, hypre_float tol );
HYPRE_Int hypre_CGNRSetTol_dbl  ( void *cgnr_vdata, hypre_double tol );
HYPRE_Int hypre_CGNRSetTol_long_dbl  ( void *cgnr_vdata, hypre_long_double tol );
HYPRE_Int hypre_CGNRSetup_flt  ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSetup_dbl  ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSetup_long_dbl  ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSolve_flt  ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSolve_dbl  ( void *cgnr_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_CGNRSolve_long_dbl  ( void *cgnr_vdata, void *A, void *b, void *x );
void *hypre_COGMRESCreate_flt( hypre_COGMRESFunctions *gmres_functions );
void *hypre_COGMRESCreate_dbl( hypre_COGMRESFunctions *gmres_functions );
void *hypre_COGMRESCreate_long_dbl( hypre_COGMRESFunctions *gmres_functions );
HYPRE_Int hypre_COGMRESDestroy_flt  ( void *gmres_vdata );
HYPRE_Int hypre_COGMRESDestroy_dbl  ( void *gmres_vdata );
HYPRE_Int hypre_COGMRESDestroy_long_dbl  ( void *gmres_vdata );
hypre_COGMRESFunctions *hypre_COGMRESFunctionsCreate_flt(
      void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_float alpha, void *A, void *x,
                                      hypre_float beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_float   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result),
      HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, HYPRE_Int k, HYPRE_Int unroll,
                                      void *result_x, void *result_y),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_float alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_float alpha, void *x, void *y ),
      HYPRE_Int    (*MassAxpy)      ( hypre_float *alpha, void **x, void *y, HYPRE_Int k,
                                      HYPRE_Int unroll),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_COGMRESFunctions *hypre_COGMRESFunctionsCreate_dbl(
      void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_double alpha, void *A, void *x,
                                      hypre_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result),
      HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, HYPRE_Int k, HYPRE_Int unroll,
                                      void *result_x, void *result_y),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_double alpha, void *x, void *y ),
      HYPRE_Int    (*MassAxpy)      ( hypre_double *alpha, void **x, void *y, HYPRE_Int k,
                                      HYPRE_Int unroll),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_COGMRESFunctions *hypre_COGMRESFunctionsCreate_long_dbl(
      void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_long_double alpha, void *A, void *x,
                                      hypre_long_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_long_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result),
      HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, HYPRE_Int k, HYPRE_Int unroll,
                                      void *result_x, void *result_y),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_long_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_long_double alpha, void *x, void *y ),
      HYPRE_Int    (*MassAxpy)      ( hypre_long_double *alpha, void **x, void *y, HYPRE_Int k,
                                      HYPRE_Int unroll),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
HYPRE_Int hypre_COGMRESGetAbsoluteTol_flt  ( void *gmres_vdata, hypre_float *a_tol );
HYPRE_Int hypre_COGMRESGetAbsoluteTol_dbl  ( void *gmres_vdata, hypre_double *a_tol );
HYPRE_Int hypre_COGMRESGetAbsoluteTol_long_dbl  ( void *gmres_vdata, hypre_long_double *a_tol );
HYPRE_Int hypre_COGMRESGetCGS_flt  ( void *gmres_vdata, HYPRE_Int *cgs );
HYPRE_Int hypre_COGMRESGetCGS_dbl  ( void *gmres_vdata, HYPRE_Int *cgs );
HYPRE_Int hypre_COGMRESGetCGS_long_dbl  ( void *gmres_vdata, HYPRE_Int *cgs );
HYPRE_Int hypre_COGMRESGetConverged_flt  ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_COGMRESGetConverged_dbl  ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_COGMRESGetConverged_long_dbl  ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_COGMRESGetConvergenceFactorTol_flt  ( void *gmres_vdata, hypre_float *cf_tol );
HYPRE_Int hypre_COGMRESGetConvergenceFactorTol_dbl  ( void *gmres_vdata, hypre_double *cf_tol );
HYPRE_Int hypre_COGMRESGetConvergenceFactorTol_long_dbl  ( void *gmres_vdata, hypre_long_double *cf_tol );
HYPRE_Int hypre_COGMRESGetFinalRelativeResidualNorm_flt  ( void *gmres_vdata,
                                                      hypre_float *relative_residual_norm );
HYPRE_Int hypre_COGMRESGetFinalRelativeResidualNorm_dbl  ( void *gmres_vdata,
                                                      hypre_double *relative_residual_norm );
HYPRE_Int hypre_COGMRESGetFinalRelativeResidualNorm_long_dbl  ( void *gmres_vdata,
                                                      hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_COGMRESGetKDim_flt  ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_COGMRESGetKDim_dbl  ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_COGMRESGetKDim_long_dbl  ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_COGMRESGetLogging_flt  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetLogging_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetLogging_long_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetMaxIter_flt  ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_COGMRESGetMaxIter_dbl  ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_COGMRESGetMaxIter_long_dbl  ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_COGMRESGetMinIter_flt  ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_COGMRESGetMinIter_dbl  ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_COGMRESGetMinIter_long_dbl  ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_COGMRESGetNumIterations_flt  ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_COGMRESGetNumIterations_dbl  ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_COGMRESGetNumIterations_long_dbl  ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_COGMRESGetPrecond_flt  ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_COGMRESGetPrecond_dbl  ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_COGMRESGetPrecond_long_dbl  ( void *gmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_COGMRESGetPrintLevel_flt  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetPrintLevel_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetPrintLevel_long_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_COGMRESGetRelChange_flt  ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_COGMRESGetRelChange_dbl  ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_COGMRESGetRelChange_long_dbl  ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_COGMRESGetResidual_flt  ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_COGMRESGetResidual_dbl  ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_COGMRESGetResidual_long_dbl  ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_COGMRESGetSkipRealResidualCheck_flt  ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_COGMRESGetSkipRealResidualCheck_dbl  ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_COGMRESGetSkipRealResidualCheck_long_dbl  ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_COGMRESGetTol_flt  ( void *gmres_vdata, hypre_float *tol );
HYPRE_Int hypre_COGMRESGetTol_dbl  ( void *gmres_vdata, hypre_double *tol );
HYPRE_Int hypre_COGMRESGetTol_long_dbl  ( void *gmres_vdata, hypre_long_double *tol );
HYPRE_Int hypre_COGMRESGetUnroll_flt  ( void *gmres_vdata, HYPRE_Int *unroll );
HYPRE_Int hypre_COGMRESGetUnroll_dbl  ( void *gmres_vdata, HYPRE_Int *unroll );
HYPRE_Int hypre_COGMRESGetUnroll_long_dbl  ( void *gmres_vdata, HYPRE_Int *unroll );
HYPRE_Int hypre_COGMRESSetAbsoluteTol_flt  ( void *gmres_vdata, hypre_float a_tol );
HYPRE_Int hypre_COGMRESSetAbsoluteTol_dbl  ( void *gmres_vdata, hypre_double a_tol );
HYPRE_Int hypre_COGMRESSetAbsoluteTol_long_dbl  ( void *gmres_vdata, hypre_long_double a_tol );
HYPRE_Int hypre_COGMRESSetCGS_flt  ( void *gmres_vdata, HYPRE_Int cgs );
HYPRE_Int hypre_COGMRESSetCGS_dbl  ( void *gmres_vdata, HYPRE_Int cgs );
HYPRE_Int hypre_COGMRESSetCGS_long_dbl  ( void *gmres_vdata, HYPRE_Int cgs );
HYPRE_Int hypre_COGMRESSetConvergenceFactorTol_flt  ( void *gmres_vdata, hypre_float cf_tol );
HYPRE_Int hypre_COGMRESSetConvergenceFactorTol_dbl  ( void *gmres_vdata, hypre_double cf_tol );
HYPRE_Int hypre_COGMRESSetConvergenceFactorTol_long_dbl  ( void *gmres_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_COGMRESSetKDim_flt  ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_COGMRESSetKDim_dbl  ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_COGMRESSetKDim_long_dbl  ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_COGMRESSetLogging_flt  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESSetLogging_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESSetLogging_long_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESSetMaxIter_flt  ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_COGMRESSetMaxIter_dbl  ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_COGMRESSetMaxIter_long_dbl  ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_COGMRESSetMinIter_flt  ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_COGMRESSetMinIter_dbl  ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_COGMRESSetMinIter_long_dbl  ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_COGMRESSetModifyPC_flt  ( void *fgmres_vdata, HYPRE_Int (*modify_pc )(void *precond_data,
                                                                                 HYPRE_Int iteration, hypre_float rel_residual_norm));
HYPRE_Int hypre_COGMRESSetModifyPC_dbl  ( void *fgmres_vdata, HYPRE_Int (*modify_pc )(void *precond_data,
                                                                                 HYPRE_Int iteration, hypre_double rel_residual_norm));
HYPRE_Int hypre_COGMRESSetModifyPC_long_dbl  ( void *fgmres_vdata, HYPRE_Int (*modify_pc )(void *precond_data,
                                                                                 HYPRE_Int iteration, hypre_long_double rel_residual_norm));
HYPRE_Int hypre_COGMRESSetPrecond_flt  ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_COGMRESSetPrecond_dbl  ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_COGMRESSetPrecond_long_dbl  ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_COGMRESSetPrintLevel_flt  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESSetPrintLevel_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESSetPrintLevel_long_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_COGMRESSetRelChange_flt  ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_COGMRESSetRelChange_dbl  ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_COGMRESSetRelChange_long_dbl  ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_COGMRESSetSkipRealResidualCheck_flt  ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_COGMRESSetSkipRealResidualCheck_dbl  ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_COGMRESSetSkipRealResidualCheck_long_dbl  ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_COGMRESSetTol_flt  ( void *gmres_vdata, hypre_float tol );
HYPRE_Int hypre_COGMRESSetTol_dbl  ( void *gmres_vdata, hypre_double tol );
HYPRE_Int hypre_COGMRESSetTol_long_dbl  ( void *gmres_vdata, hypre_long_double tol );
HYPRE_Int hypre_COGMRESSetUnroll_flt  ( void *gmres_vdata, HYPRE_Int unroll );
HYPRE_Int hypre_COGMRESSetUnroll_dbl  ( void *gmres_vdata, HYPRE_Int unroll );
HYPRE_Int hypre_COGMRESSetUnroll_long_dbl  ( void *gmres_vdata, HYPRE_Int unroll );
HYPRE_Int hypre_COGMRESSetup_flt  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSetup_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSetup_long_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSolve_flt  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSolve_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_COGMRESSolve_long_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
void *hypre_FlexGMRESCreate_flt( hypre_FlexGMRESFunctions *fgmres_functions );
void *hypre_FlexGMRESCreate_dbl( hypre_FlexGMRESFunctions *fgmres_functions );
void *hypre_FlexGMRESCreate_long_dbl( hypre_FlexGMRESFunctions *fgmres_functions );
HYPRE_Int hypre_FlexGMRESDestroy_flt  ( void *fgmres_vdata );
HYPRE_Int hypre_FlexGMRESDestroy_dbl  ( void *fgmres_vdata );
HYPRE_Int hypre_FlexGMRESDestroy_long_dbl  ( void *fgmres_vdata );
hypre_FlexGMRESFunctions *hypre_FlexGMRESFunctionsCreate_flt(
      void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_float alpha, void *A,
                                      void *x, hypre_float beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_float   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_float alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_float alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_FlexGMRESFunctions *hypre_FlexGMRESFunctionsCreate_dbl(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_double alpha, void *A,
                                      void *x, hypre_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_FlexGMRESFunctions *hypre_FlexGMRESFunctionsCreate_long_dbl(
      void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_long_double alpha, void *A,
                                      void *x, hypre_long_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_long_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_long_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_long_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
HYPRE_Int hypre_FlexGMRESGetAbsoluteTol_flt  ( void *fgmres_vdata, hypre_float *a_tol );
HYPRE_Int hypre_FlexGMRESGetAbsoluteTol_dbl  ( void *fgmres_vdata, hypre_double *a_tol );
HYPRE_Int hypre_FlexGMRESGetAbsoluteTol_long_dbl  ( void *fgmres_vdata, hypre_long_double *a_tol );
HYPRE_Int hypre_FlexGMRESGetConverged_flt  ( void *fgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_FlexGMRESGetConverged_dbl  ( void *fgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_FlexGMRESGetConverged_long_dbl  ( void *fgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_FlexGMRESGetConvergenceFactorTol_flt  ( void *fgmres_vdata, hypre_float *cf_tol );
HYPRE_Int hypre_FlexGMRESGetConvergenceFactorTol_dbl  ( void *fgmres_vdata, hypre_double *cf_tol );
HYPRE_Int hypre_FlexGMRESGetConvergenceFactorTol_long_dbl  ( void *fgmres_vdata, hypre_long_double *cf_tol );
HYPRE_Int hypre_FlexGMRESGetFinalRelativeResidualNorm_flt  ( void *fgmres_vdata,
                                                        hypre_float *relative_residual_norm );
HYPRE_Int hypre_FlexGMRESGetFinalRelativeResidualNorm_dbl  ( void *fgmres_vdata,
                                                        hypre_double *relative_residual_norm );
HYPRE_Int hypre_FlexGMRESGetFinalRelativeResidualNorm_long_dbl  ( void *fgmres_vdata,
                                                        hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_FlexGMRESGetKDim_flt  ( void *fgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_FlexGMRESGetKDim_dbl  ( void *fgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_FlexGMRESGetKDim_long_dbl  ( void *fgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_FlexGMRESGetLogging_flt  ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetLogging_dbl  ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetLogging_long_dbl  ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetMaxIter_flt  ( void *fgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_FlexGMRESGetMaxIter_dbl  ( void *fgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_FlexGMRESGetMaxIter_long_dbl  ( void *fgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_FlexGMRESGetMinIter_flt  ( void *fgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_FlexGMRESGetMinIter_dbl  ( void *fgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_FlexGMRESGetMinIter_long_dbl  ( void *fgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_FlexGMRESGetNumIterations_flt  ( void *fgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FlexGMRESGetNumIterations_dbl  ( void *fgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FlexGMRESGetNumIterations_long_dbl  ( void *fgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_FlexGMRESGetPrecond_flt  ( void *fgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_FlexGMRESGetPrecond_dbl  ( void *fgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_FlexGMRESGetPrecond_long_dbl  ( void *fgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_FlexGMRESGetPrintLevel_flt  ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetPrintLevel_dbl  ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetPrintLevel_long_dbl  ( void *fgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_FlexGMRESGetResidual_flt  ( void *fgmres_vdata, void **residual );
HYPRE_Int hypre_FlexGMRESGetResidual_dbl  ( void *fgmres_vdata, void **residual );
HYPRE_Int hypre_FlexGMRESGetResidual_long_dbl  ( void *fgmres_vdata, void **residual );
HYPRE_Int hypre_FlexGMRESGetStopCrit_flt  ( void *fgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_FlexGMRESGetStopCrit_dbl  ( void *fgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_FlexGMRESGetStopCrit_long_dbl  ( void *fgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_FlexGMRESGetTol_flt  ( void *fgmres_vdata, hypre_float *tol );
HYPRE_Int hypre_FlexGMRESGetTol_dbl  ( void *fgmres_vdata, hypre_double *tol );
HYPRE_Int hypre_FlexGMRESGetTol_long_dbl  ( void *fgmres_vdata, hypre_long_double *tol );
HYPRE_Int hypre_FlexGMRESModifyPCDefault_flt  ( void *precond_data, HYPRE_Int iteration,
                                           hypre_float rel_residual_norm );
HYPRE_Int hypre_FlexGMRESModifyPCDefault_dbl  ( void *precond_data, HYPRE_Int iteration,
                                           hypre_double rel_residual_norm );
HYPRE_Int hypre_FlexGMRESModifyPCDefault_long_dbl  ( void *precond_data, HYPRE_Int iteration,
                                           hypre_long_double rel_residual_norm );
HYPRE_Int hypre_FlexGMRESSetAbsoluteTol_flt  ( void *fgmres_vdata, hypre_float a_tol );
HYPRE_Int hypre_FlexGMRESSetAbsoluteTol_dbl  ( void *fgmres_vdata, hypre_double a_tol );
HYPRE_Int hypre_FlexGMRESSetAbsoluteTol_long_dbl  ( void *fgmres_vdata, hypre_long_double a_tol );
HYPRE_Int hypre_FlexGMRESSetConvergenceFactorTol_flt  ( void *fgmres_vdata, hypre_float cf_tol );
HYPRE_Int hypre_FlexGMRESSetConvergenceFactorTol_dbl  ( void *fgmres_vdata, hypre_double cf_tol );
HYPRE_Int hypre_FlexGMRESSetConvergenceFactorTol_long_dbl  ( void *fgmres_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_FlexGMRESSetKDim_flt  ( void *fgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_FlexGMRESSetKDim_dbl  ( void *fgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_FlexGMRESSetKDim_long_dbl  ( void *fgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_FlexGMRESSetLogging_flt  ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESSetLogging_dbl  ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESSetLogging_long_dbl  ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESSetMaxIter_flt  ( void *fgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_FlexGMRESSetMaxIter_dbl  ( void *fgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_FlexGMRESSetMaxIter_long_dbl  ( void *fgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_FlexGMRESSetMinIter_flt  ( void *fgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_FlexGMRESSetMinIter_dbl  ( void *fgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_FlexGMRESSetMinIter_long_dbl  ( void *fgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_FlexGMRESSetModifyPC_flt  ( void *fgmres_vdata,
                                       HYPRE_Int (*modify_pc )(void *precond_data, HYPRE_Int iteration, hypre_float rel_residual_norm));
HYPRE_Int hypre_FlexGMRESSetModifyPC_dbl  ( void *fgmres_vdata,
                                       HYPRE_Int (*modify_pc )(void *precond_data, HYPRE_Int iteration, hypre_double rel_residual_norm));
HYPRE_Int hypre_FlexGMRESSetModifyPC_long_dbl  ( void *fgmres_vdata,
                                       HYPRE_Int (*modify_pc )(void *precond_data, HYPRE_Int iteration, hypre_long_double rel_residual_norm));
HYPRE_Int hypre_FlexGMRESSetPrecond_flt  ( void *fgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                                void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_FlexGMRESSetPrecond_dbl  ( void *fgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                                void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_FlexGMRESSetPrecond_long_dbl  ( void *fgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                                void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_FlexGMRESSetPrintLevel_flt  ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESSetPrintLevel_dbl  ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESSetPrintLevel_long_dbl  ( void *fgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_FlexGMRESSetStopCrit_flt  ( void *fgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_FlexGMRESSetStopCrit_dbl  ( void *fgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_FlexGMRESSetStopCrit_long_dbl  ( void *fgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_FlexGMRESSetTol_flt  ( void *fgmres_vdata, hypre_float tol );
HYPRE_Int hypre_FlexGMRESSetTol_dbl  ( void *fgmres_vdata, hypre_double tol );
HYPRE_Int hypre_FlexGMRESSetTol_long_dbl  ( void *fgmres_vdata, hypre_long_double tol );
HYPRE_Int hypre_FlexGMRESSetup_flt  ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSetup_dbl  ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSetup_long_dbl  ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSolve_flt  ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSolve_dbl  ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_FlexGMRESSolve_long_dbl  ( void *fgmres_vdata, void *A, void *b, void *x );
HYPRE_Int HYPRE_BiCGSTABDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BiCGSTABDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BiCGSTABDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_BiCGSTABGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_BiCGSTABGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_BiCGSTABGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_BiCGSTABGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BiCGSTABGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BiCGSTABGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_BiCGSTABGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_BiCGSTABGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_BiCGSTABGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_BiCGSTABGetResidual_flt  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_BiCGSTABGetResidual_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_BiCGSTABGetResidual_long_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_BiCGSTABSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_BiCGSTABSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_BiCGSTABSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_BiCGSTABSetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float cf_tol );
HYPRE_Int HYPRE_BiCGSTABSetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double cf_tol );
HYPRE_Int HYPRE_BiCGSTABSetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double cf_tol );
HYPRE_Int HYPRE_BiCGSTABSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BiCGSTABSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BiCGSTABSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_BiCGSTABSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BiCGSTABSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BiCGSTABSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_BiCGSTABSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BiCGSTABSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BiCGSTABSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_BiCGSTABSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                     HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_BiCGSTABSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                     HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_BiCGSTABSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                     HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_BiCGSTABSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BiCGSTABSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BiCGSTABSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int print_level );
HYPRE_Int HYPRE_BiCGSTABSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_BiCGSTABSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_BiCGSTABSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_BiCGSTABSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_BiCGSTABSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_BiCGSTABSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_BiCGSTABSetup_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSetup_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSolve_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSolve_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_BiCGSTABSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRDestroy_flt  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CGNRDestroy_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CGNRDestroy_long_dbl  ( HYPRE_Solver solver );
HYPRE_Int HYPRE_CGNRGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_CGNRGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_CGNRGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_CGNRGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CGNRGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CGNRGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_CGNRGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CGNRGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CGNRGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_CGNRSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_CGNRSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_CGNRSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int logging );
HYPRE_Int HYPRE_CGNRSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_CGNRSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_CGNRSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_CGNRSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_CGNRSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_CGNRSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_CGNRSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                 HYPRE_PtrToSolverFcn precondT, HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_CGNRSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                 HYPRE_PtrToSolverFcn precondT, HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_CGNRSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                 HYPRE_PtrToSolverFcn precondT, HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_CGNRSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_CGNRSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_CGNRSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_CGNRSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_CGNRSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_CGNRSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_CGNRSetup_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSetup_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSolve_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSolve_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_CGNRSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESGetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float *a_tol );
HYPRE_Int HYPRE_COGMRESGetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double *a_tol );
HYPRE_Int HYPRE_COGMRESGetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *a_tol );
HYPRE_Int HYPRE_COGMRESGetCGS_flt  ( HYPRE_Solver solver, HYPRE_Int *cgs );
HYPRE_Int HYPRE_COGMRESGetCGS_dbl  ( HYPRE_Solver solver, HYPRE_Int *cgs );
HYPRE_Int HYPRE_COGMRESGetCGS_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *cgs );
HYPRE_Int HYPRE_COGMRESGetConverged_flt  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_COGMRESGetConverged_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_COGMRESGetConverged_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_COGMRESGetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float *cf_tol );
HYPRE_Int HYPRE_COGMRESGetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double *cf_tol );
HYPRE_Int HYPRE_COGMRESGetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *cf_tol );
HYPRE_Int HYPRE_COGMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_COGMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_COGMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_COGMRESGetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_COGMRESGetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_COGMRESGetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_COGMRESGetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_COGMRESGetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_COGMRESGetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_COGMRESGetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_COGMRESGetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_COGMRESGetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_COGMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_COGMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_COGMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_COGMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_COGMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_COGMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_COGMRESGetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_COGMRESGetResidual_flt  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_COGMRESGetResidual_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_COGMRESGetResidual_long_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_COGMRESGetTol_flt  ( HYPRE_Solver solver, hypre_float *tol );
HYPRE_Int HYPRE_COGMRESGetTol_dbl  ( HYPRE_Solver solver, hypre_double *tol );
HYPRE_Int HYPRE_COGMRESGetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *tol );
HYPRE_Int HYPRE_COGMRESGetUnroll_flt  ( HYPRE_Solver solver, HYPRE_Int *unroll );
HYPRE_Int HYPRE_COGMRESGetUnroll_dbl  ( HYPRE_Solver solver, HYPRE_Int *unroll );
HYPRE_Int HYPRE_COGMRESGetUnroll_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *unroll );
HYPRE_Int HYPRE_COGMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_COGMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_COGMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_COGMRESSetCGS_flt  ( HYPRE_Solver solver, HYPRE_Int cgs );
HYPRE_Int HYPRE_COGMRESSetCGS_dbl  ( HYPRE_Solver solver, HYPRE_Int cgs );
HYPRE_Int HYPRE_COGMRESSetCGS_long_dbl  ( HYPRE_Solver solver, HYPRE_Int cgs );
HYPRE_Int HYPRE_COGMRESSetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float cf_tol );
HYPRE_Int HYPRE_COGMRESSetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double cf_tol );
HYPRE_Int HYPRE_COGMRESSetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double cf_tol );
HYPRE_Int HYPRE_COGMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_COGMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_COGMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_COGMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_COGMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_COGMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_COGMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_COGMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_COGMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_COGMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                    HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_COGMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                    HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_COGMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                    HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_COGMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_COGMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_COGMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_COGMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_COGMRESSetUnroll_flt  ( HYPRE_Solver solver, HYPRE_Int unroll );
HYPRE_Int HYPRE_COGMRESSetUnroll_dbl  ( HYPRE_Solver solver, HYPRE_Int unroll );
HYPRE_Int HYPRE_COGMRESSetUnroll_long_dbl  ( HYPRE_Solver solver, HYPRE_Int unroll );
HYPRE_Int HYPRE_COGMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_COGMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                               HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESGetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float *a_tol );
HYPRE_Int HYPRE_FlexGMRESGetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double *a_tol );
HYPRE_Int HYPRE_FlexGMRESGetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *a_tol );
HYPRE_Int HYPRE_FlexGMRESGetConverged_flt  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_FlexGMRESGetConverged_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_FlexGMRESGetConverged_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_FlexGMRESGetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float *cf_tol );
HYPRE_Int HYPRE_FlexGMRESGetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double *cf_tol );
HYPRE_Int HYPRE_FlexGMRESGetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *cf_tol );
HYPRE_Int HYPRE_FlexGMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_FlexGMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_FlexGMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_FlexGMRESGetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_FlexGMRESGetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_FlexGMRESGetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_FlexGMRESGetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_FlexGMRESGetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_FlexGMRESGetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_FlexGMRESGetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_FlexGMRESGetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_FlexGMRESGetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_FlexGMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_FlexGMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_FlexGMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_FlexGMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_FlexGMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_FlexGMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_FlexGMRESGetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_FlexGMRESGetResidual_flt  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_FlexGMRESGetResidual_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_FlexGMRESGetResidual_long_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_FlexGMRESGetTol_flt  ( HYPRE_Solver solver, hypre_float *tol );
HYPRE_Int HYPRE_FlexGMRESGetTol_dbl  ( HYPRE_Solver solver, hypre_double *tol );
HYPRE_Int HYPRE_FlexGMRESGetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *tol );
HYPRE_Int HYPRE_FlexGMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_FlexGMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_FlexGMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_FlexGMRESSetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float cf_tol );
HYPRE_Int HYPRE_FlexGMRESSetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double cf_tol );
HYPRE_Int HYPRE_FlexGMRESSetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double cf_tol );
HYPRE_Int HYPRE_FlexGMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_FlexGMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_FlexGMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_FlexGMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_FlexGMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_FlexGMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_FlexGMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_FlexGMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_FlexGMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_FlexGMRESSetModifyPC_flt  ( HYPRE_Solver solver, HYPRE_Int (*modify_pc )(HYPRE_Solver,
                                                                                    HYPRE_Int, hypre_float ));
HYPRE_Int HYPRE_FlexGMRESSetModifyPC_dbl  ( HYPRE_Solver solver, HYPRE_Int (*modify_pc )(HYPRE_Solver,
                                                                                    HYPRE_Int, hypre_double ));
HYPRE_Int HYPRE_FlexGMRESSetModifyPC_long_dbl  ( HYPRE_Solver solver, HYPRE_Int (*modify_pc )(HYPRE_Solver,
                                                                                    HYPRE_Int, hypre_long_double ));
HYPRE_Int HYPRE_FlexGMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                      HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_FlexGMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                      HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_FlexGMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                      HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_FlexGMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_FlexGMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_FlexGMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_FlexGMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_FlexGMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_FlexGMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                                 HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESGetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float *a_tol );
HYPRE_Int HYPRE_GMRESGetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double *a_tol );
HYPRE_Int HYPRE_GMRESGetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *a_tol );
HYPRE_Int HYPRE_GMRESGetConverged_flt  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_GMRESGetConverged_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_GMRESGetConverged_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_GMRESGetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float *cf_tol );
HYPRE_Int HYPRE_GMRESGetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double *cf_tol );
HYPRE_Int HYPRE_GMRESGetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *cf_tol );
HYPRE_Int HYPRE_GMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_GMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_GMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_GMRESGetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_GMRESGetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_GMRESGetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_GMRESGetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_GMRESGetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_GMRESGetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_GMRESGetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_GMRESGetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_GMRESGetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_GMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_GMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_GMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_GMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_GMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_GMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_GMRESGetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_GMRESGetRelChange_flt  ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_GMRESGetRelChange_dbl  ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_GMRESGetRelChange_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_GMRESGetResidual_flt  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_GMRESGetResidual_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_GMRESGetResidual_long_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_GMRESGetSkipRealResidualCheck_flt  ( HYPRE_Solver solver, HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_GMRESGetSkipRealResidualCheck_dbl  ( HYPRE_Solver solver, HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_GMRESGetSkipRealResidualCheck_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *skip_real_r_check );
HYPRE_Int HYPRE_GMRESGetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_GMRESGetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_GMRESGetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_GMRESGetTol_flt  ( HYPRE_Solver solver, hypre_float *tol );
HYPRE_Int HYPRE_GMRESGetTol_dbl  ( HYPRE_Solver solver, hypre_double *tol );
HYPRE_Int HYPRE_GMRESGetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *tol );
HYPRE_Int HYPRE_GMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_GMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_GMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_GMRESSetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float cf_tol );
HYPRE_Int HYPRE_GMRESSetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double cf_tol );
HYPRE_Int HYPRE_GMRESSetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double cf_tol );
HYPRE_Int HYPRE_GMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_GMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_GMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_GMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_GMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_GMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_GMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_GMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_GMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_GMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                  HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_GMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                  HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_GMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                  HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_GMRESSetPrecondMatrix_flt  ( HYPRE_Solver solver, HYPRE_Matrix precond_matrix);
HYPRE_Int HYPRE_GMRESSetPrecondMatrix_dbl  ( HYPRE_Solver solver, HYPRE_Matrix precond_matrix);
HYPRE_Int HYPRE_GMRESSetPrecondMatrix_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix precond_matrix);
HYPRE_Int HYPRE_GMRESGetPrecondMatrix_flt( HYPRE_Solver  solver, HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_GMRESGetPrecondMatrix_dbl( HYPRE_Solver  solver, HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_GMRESGetPrecondMatrix_long_dbl( HYPRE_Solver  solver, HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_GMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_GMRESSetRelChange_flt  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_GMRESSetRelChange_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_GMRESSetRelChange_long_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_GMRESSetSkipRealResidualCheck_flt  ( HYPRE_Solver solver, HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_GMRESSetSkipRealResidualCheck_dbl  ( HYPRE_Solver solver, HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_GMRESSetSkipRealResidualCheck_long_dbl  ( HYPRE_Solver solver, HYPRE_Int skip_real_r_check );
HYPRE_Int HYPRE_GMRESSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_GMRESSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_GMRESSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_GMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_GMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_GMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_GMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_GMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESGetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float *a_tol );
HYPRE_Int HYPRE_LGMRESGetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double *a_tol );
HYPRE_Int HYPRE_LGMRESGetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *a_tol );
HYPRE_Int HYPRE_LGMRESGetAugDim_flt  ( HYPRE_Solver solver, HYPRE_Int *aug_dim );
HYPRE_Int HYPRE_LGMRESGetAugDim_dbl  ( HYPRE_Solver solver, HYPRE_Int *aug_dim );
HYPRE_Int HYPRE_LGMRESGetAugDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *aug_dim );
HYPRE_Int HYPRE_LGMRESGetConverged_flt  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_LGMRESGetConverged_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_LGMRESGetConverged_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_LGMRESGetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float *cf_tol );
HYPRE_Int HYPRE_LGMRESGetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double *cf_tol );
HYPRE_Int HYPRE_LGMRESGetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *cf_tol );
HYPRE_Int HYPRE_LGMRESGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_LGMRESGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_LGMRESGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_LGMRESGetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_LGMRESGetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_LGMRESGetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *k_dim );
HYPRE_Int HYPRE_LGMRESGetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_LGMRESGetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_LGMRESGetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_LGMRESGetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_LGMRESGetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_LGMRESGetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *min_iter );
HYPRE_Int HYPRE_LGMRESGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_LGMRESGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_LGMRESGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_LGMRESGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_LGMRESGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_LGMRESGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_LGMRESGetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_LGMRESGetResidual_flt  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_LGMRESGetResidual_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_LGMRESGetResidual_long_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_LGMRESGetTol_flt  ( HYPRE_Solver solver, hypre_float *tol );
HYPRE_Int HYPRE_LGMRESGetTol_dbl  ( HYPRE_Solver solver, hypre_double *tol );
HYPRE_Int HYPRE_LGMRESGetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *tol );
HYPRE_Int HYPRE_LGMRESSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_LGMRESSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_LGMRESSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_LGMRESSetAugDim_flt  ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_LGMRESSetAugDim_dbl  ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_LGMRESSetAugDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int aug_dim );
HYPRE_Int HYPRE_LGMRESSetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float cf_tol );
HYPRE_Int HYPRE_LGMRESSetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double cf_tol );
HYPRE_Int HYPRE_LGMRESSetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double cf_tol );
HYPRE_Int HYPRE_LGMRESSetKDim_flt  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_LGMRESSetKDim_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_LGMRESSetKDim_long_dbl  ( HYPRE_Solver solver, HYPRE_Int k_dim );
HYPRE_Int HYPRE_LGMRESSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_LGMRESSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_LGMRESSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_LGMRESSetMinIter_flt  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_LGMRESSetMinIter_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_LGMRESSetMinIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int min_iter );
HYPRE_Int HYPRE_LGMRESSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                   HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_LGMRESSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                   HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_LGMRESSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                   HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_LGMRESSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_LGMRESSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_LGMRESSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_LGMRESSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_LGMRESSetup_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSetup_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSolve_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSolve_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_LGMRESSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGGetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float *a_tol );
HYPRE_Int HYPRE_PCGGetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double *a_tol );
HYPRE_Int HYPRE_PCGGetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *a_tol );
HYPRE_Int HYPRE_PCGGetAbsoluteTolFactor_flt  ( HYPRE_Solver solver, hypre_float *abstolf );
HYPRE_Int HYPRE_PCGGetAbsoluteTolFactor_dbl  ( HYPRE_Solver solver, hypre_double *abstolf );
HYPRE_Int HYPRE_PCGGetAbsoluteTolFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double *abstolf );
HYPRE_Int HYPRE_PCGGetConverged_flt  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_PCGGetConverged_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_PCGGetConverged_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *converged );
HYPRE_Int HYPRE_PCGGetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float *cf_tol );
HYPRE_Int HYPRE_PCGGetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double *cf_tol );
HYPRE_Int HYPRE_PCGGetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *cf_tol );
HYPRE_Int HYPRE_PCGGetFinalRelativeResidualNorm_flt  ( HYPRE_Solver solver, hypre_float *norm );
HYPRE_Int HYPRE_PCGGetFinalRelativeResidualNorm_dbl  ( HYPRE_Solver solver, hypre_double *norm );
HYPRE_Int HYPRE_PCGGetFinalRelativeResidualNorm_long_dbl  ( HYPRE_Solver solver, hypre_long_double *norm );
HYPRE_Int HYPRE_PCGGetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_PCGGetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_PCGGetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *max_iter );
HYPRE_Int HYPRE_PCGGetNumIterations_flt  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_PCGGetNumIterations_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_PCGGetNumIterations_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_PCGGetPrecond_flt  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_PCGGetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_PCGGetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr );
HYPRE_Int HYPRE_PCGGetPrecondMatrix_flt( HYPRE_Solver  solver, HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_PCGGetPrecondMatrix_dbl( HYPRE_Solver  solver, HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_PCGGetPrecondMatrix_long_dbl( HYPRE_Solver  solver, HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int HYPRE_PCGGetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *level );
HYPRE_Int HYPRE_PCGGetRecomputeResidual_flt  ( HYPRE_Solver solver, HYPRE_Int *recompute_residual );
HYPRE_Int HYPRE_PCGGetRecomputeResidual_dbl  ( HYPRE_Solver solver, HYPRE_Int *recompute_residual );
HYPRE_Int HYPRE_PCGGetRecomputeResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *recompute_residual );
HYPRE_Int HYPRE_PCGGetRecomputeResidualP_flt  ( HYPRE_Solver solver, HYPRE_Int *recompute_residual_p );
HYPRE_Int HYPRE_PCGGetRecomputeResidualP_dbl  ( HYPRE_Solver solver, HYPRE_Int *recompute_residual_p );
HYPRE_Int HYPRE_PCGGetRecomputeResidualP_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *recompute_residual_p );
HYPRE_Int HYPRE_PCGGetRelChange_flt  ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_PCGGetRelChange_dbl  ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_PCGGetRelChange_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *rel_change );
HYPRE_Int HYPRE_PCGGetResidual_flt  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_PCGGetResidual_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_PCGGetResidual_long_dbl  ( HYPRE_Solver solver, void *residual );
HYPRE_Int HYPRE_PCGGetResidualTol_flt  ( HYPRE_Solver solver, hypre_float *rtol );
HYPRE_Int HYPRE_PCGGetResidualTol_dbl  ( HYPRE_Solver solver, hypre_double *rtol );
HYPRE_Int HYPRE_PCGGetResidualTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *rtol );
HYPRE_Int HYPRE_PCGGetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_PCGGetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_PCGGetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *stop_crit );
HYPRE_Int HYPRE_PCGGetTol_flt  ( HYPRE_Solver solver, hypre_float *tol );
HYPRE_Int HYPRE_PCGGetTol_dbl  ( HYPRE_Solver solver, hypre_double *tol );
HYPRE_Int HYPRE_PCGGetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double *tol );
HYPRE_Int HYPRE_PCGGetTwoNorm_flt  ( HYPRE_Solver solver, HYPRE_Int *two_norm );
HYPRE_Int HYPRE_PCGGetTwoNorm_dbl  ( HYPRE_Solver solver, HYPRE_Int *two_norm );
HYPRE_Int HYPRE_PCGGetTwoNorm_long_dbl  ( HYPRE_Solver solver, HYPRE_Int *two_norm );
HYPRE_Int HYPRE_PCGSetAbsoluteTol_flt  ( HYPRE_Solver solver, hypre_float a_tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTol_dbl  ( HYPRE_Solver solver, hypre_double a_tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double a_tol );
HYPRE_Int HYPRE_PCGSetAbsoluteTolFactor_flt  ( HYPRE_Solver solver, hypre_float abstolf );
HYPRE_Int HYPRE_PCGSetAbsoluteTolFactor_dbl  ( HYPRE_Solver solver, hypre_double abstolf );
HYPRE_Int HYPRE_PCGSetAbsoluteTolFactor_long_dbl  ( HYPRE_Solver solver, hypre_long_double abstolf );
HYPRE_Int HYPRE_PCGSetConvergenceFactorTol_flt  ( HYPRE_Solver solver, hypre_float cf_tol );
HYPRE_Int HYPRE_PCGSetConvergenceFactorTol_dbl  ( HYPRE_Solver solver, hypre_double cf_tol );
HYPRE_Int HYPRE_PCGSetConvergenceFactorTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double cf_tol );
HYPRE_Int HYPRE_PCGSetLogging_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGSetLogging_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGSetLogging_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGSetMaxIter_flt  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_PCGSetMaxIter_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_PCGSetMaxIter_long_dbl  ( HYPRE_Solver solver, HYPRE_Int max_iter );
HYPRE_Int HYPRE_PCGSetPrecond_flt  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_PCGSetPrecond_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_PCGSetPrecond_long_dbl  ( HYPRE_Solver solver, HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup, HYPRE_Solver precond_solver );
HYPRE_Int HYPRE_PCGSetPrecondMatrix_flt( HYPRE_Solver  solver, HYPRE_Matrix precond_matrix);
HYPRE_Int HYPRE_PCGSetPrecondMatrix_dbl( HYPRE_Solver  solver, HYPRE_Matrix precond_matrix);
HYPRE_Int HYPRE_PCGSetPrecondMatrix_long_dbl( HYPRE_Solver  solver, HYPRE_Matrix precond_matrix);
HYPRE_Int HYPRE_PCGSetPrintLevel_flt  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGSetPrintLevel_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGSetPrintLevel_long_dbl  ( HYPRE_Solver solver, HYPRE_Int level );
HYPRE_Int HYPRE_PCGSetRecomputeResidual_flt  ( HYPRE_Solver solver, HYPRE_Int recompute_residual );
HYPRE_Int HYPRE_PCGSetRecomputeResidual_dbl  ( HYPRE_Solver solver, HYPRE_Int recompute_residual );
HYPRE_Int HYPRE_PCGSetRecomputeResidual_long_dbl  ( HYPRE_Solver solver, HYPRE_Int recompute_residual );
HYPRE_Int HYPRE_PCGSetRecomputeResidualP_flt  ( HYPRE_Solver solver, HYPRE_Int recompute_residual_p );
HYPRE_Int HYPRE_PCGSetRecomputeResidualP_dbl  ( HYPRE_Solver solver, HYPRE_Int recompute_residual_p );
HYPRE_Int HYPRE_PCGSetRecomputeResidualP_long_dbl  ( HYPRE_Solver solver, HYPRE_Int recompute_residual_p );
HYPRE_Int HYPRE_PCGSetRelChange_flt  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_PCGSetRelChange_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_PCGSetRelChange_long_dbl  ( HYPRE_Solver solver, HYPRE_Int rel_change );
HYPRE_Int HYPRE_PCGSetResidualTol_flt  ( HYPRE_Solver solver, hypre_float rtol );
HYPRE_Int HYPRE_PCGSetResidualTol_dbl  ( HYPRE_Solver solver, hypre_double rtol );
HYPRE_Int HYPRE_PCGSetResidualTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double rtol );
HYPRE_Int HYPRE_PCGSetStopCrit_flt  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_PCGSetStopCrit_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_PCGSetStopCrit_long_dbl  ( HYPRE_Solver solver, HYPRE_Int stop_crit );
HYPRE_Int HYPRE_PCGSetTol_flt  ( HYPRE_Solver solver, hypre_float tol );
HYPRE_Int HYPRE_PCGSetTol_dbl  ( HYPRE_Solver solver, hypre_double tol );
HYPRE_Int HYPRE_PCGSetTol_long_dbl  ( HYPRE_Solver solver, hypre_long_double tol );
HYPRE_Int HYPRE_PCGSetTwoNorm_flt  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_PCGSetTwoNorm_dbl  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_PCGSetTwoNorm_long_dbl  ( HYPRE_Solver solver, HYPRE_Int two_norm );
HYPRE_Int HYPRE_PCGSetup_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSetup_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSetup_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSolve_flt  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSolve_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
HYPRE_Int HYPRE_PCGSolve_long_dbl  ( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
void *hypre_LGMRESCreate_flt( hypre_LGMRESFunctions *lgmres_functions );
void *hypre_LGMRESCreate_dbl( hypre_LGMRESFunctions *lgmres_functions );
void *hypre_LGMRESCreate_long_dbl( hypre_LGMRESFunctions *lgmres_functions );
HYPRE_Int hypre_LGMRESDestroy_flt  ( void *lgmres_vdata );
HYPRE_Int hypre_LGMRESDestroy_dbl  ( void *lgmres_vdata );
HYPRE_Int hypre_LGMRESDestroy_long_dbl  ( void *lgmres_vdata );
hypre_LGMRESFunctions *hypre_LGMRESFunctionsCreate_flt(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_float alpha, void *A,
                                      void *x, hypre_float beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_float   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_float alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_float alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_LGMRESFunctions *hypre_LGMRESFunctionsCreate_dbl(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_double alpha, void *A,
                                      void *x, hypre_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_LGMRESFunctions *hypre_LGMRESFunctionsCreate_long_dbl(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_long_double alpha, void *A,
                                      void *x, hypre_long_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_long_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_long_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_long_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
HYPRE_Int hypre_LGMRESGetAbsoluteTol_flt  ( void *lgmres_vdata, hypre_float *a_tol );
HYPRE_Int hypre_LGMRESGetAbsoluteTol_dbl  ( void *lgmres_vdata, hypre_double *a_tol );
HYPRE_Int hypre_LGMRESGetAbsoluteTol_long_dbl  ( void *lgmres_vdata, hypre_long_double *a_tol );
HYPRE_Int hypre_LGMRESGetAugDim_flt  ( void *lgmres_vdata, HYPRE_Int *aug_dim );
HYPRE_Int hypre_LGMRESGetAugDim_dbl  ( void *lgmres_vdata, HYPRE_Int *aug_dim );
HYPRE_Int hypre_LGMRESGetAugDim_long_dbl  ( void *lgmres_vdata, HYPRE_Int *aug_dim );
HYPRE_Int hypre_LGMRESGetConverged_flt  ( void *lgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_LGMRESGetConverged_dbl  ( void *lgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_LGMRESGetConverged_long_dbl  ( void *lgmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_LGMRESGetConvergenceFactorTol_flt  ( void *lgmres_vdata, hypre_float *cf_tol );
HYPRE_Int hypre_LGMRESGetConvergenceFactorTol_dbl  ( void *lgmres_vdata, hypre_double *cf_tol );
HYPRE_Int hypre_LGMRESGetConvergenceFactorTol_long_dbl  ( void *lgmres_vdata, hypre_long_double *cf_tol );
HYPRE_Int hypre_LGMRESGetFinalRelativeResidualNorm_flt  ( void *lgmres_vdata,
                                                     hypre_float *relative_residual_norm );
HYPRE_Int hypre_LGMRESGetFinalRelativeResidualNorm_dbl  ( void *lgmres_vdata,
                                                     hypre_double *relative_residual_norm );
HYPRE_Int hypre_LGMRESGetFinalRelativeResidualNorm_long_dbl  ( void *lgmres_vdata,
                                                     hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_LGMRESGetKDim_flt  ( void *lgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_LGMRESGetKDim_dbl  ( void *lgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_LGMRESGetKDim_long_dbl  ( void *lgmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_LGMRESGetLogging_flt  ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetLogging_dbl  ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetLogging_long_dbl  ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetMaxIter_flt  ( void *lgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_LGMRESGetMaxIter_dbl  ( void *lgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_LGMRESGetMaxIter_long_dbl  ( void *lgmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_LGMRESGetMinIter_flt  ( void *lgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_LGMRESGetMinIter_dbl  ( void *lgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_LGMRESGetMinIter_long_dbl  ( void *lgmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_LGMRESGetNumIterations_flt  ( void *lgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_LGMRESGetNumIterations_dbl  ( void *lgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_LGMRESGetNumIterations_long_dbl  ( void *lgmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_LGMRESGetPrecond_flt  ( void *lgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_LGMRESGetPrecond_dbl  ( void *lgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_LGMRESGetPrecond_long_dbl  ( void *lgmres_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_LGMRESGetPrintLevel_flt  ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetPrintLevel_dbl  ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetPrintLevel_long_dbl  ( void *lgmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_LGMRESGetResidual_flt  ( void *lgmres_vdata, void **residual );
HYPRE_Int hypre_LGMRESGetResidual_dbl  ( void *lgmres_vdata, void **residual );
HYPRE_Int hypre_LGMRESGetResidual_long_dbl  ( void *lgmres_vdata, void **residual );
HYPRE_Int hypre_LGMRESGetStopCrit_flt  ( void *lgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_LGMRESGetStopCrit_dbl  ( void *lgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_LGMRESGetStopCrit_long_dbl  ( void *lgmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_LGMRESGetTol_flt  ( void *lgmres_vdata, hypre_float *tol );
HYPRE_Int hypre_LGMRESGetTol_dbl  ( void *lgmres_vdata, hypre_double *tol );
HYPRE_Int hypre_LGMRESGetTol_long_dbl  ( void *lgmres_vdata, hypre_long_double *tol );
HYPRE_Int hypre_LGMRESSetAbsoluteTol_flt  ( void *lgmres_vdata, hypre_float a_tol );
HYPRE_Int hypre_LGMRESSetAbsoluteTol_dbl  ( void *lgmres_vdata, hypre_double a_tol );
HYPRE_Int hypre_LGMRESSetAbsoluteTol_long_dbl  ( void *lgmres_vdata, hypre_long_double a_tol );
HYPRE_Int hypre_LGMRESSetAugDim_flt  ( void *lgmres_vdata, HYPRE_Int aug_dim );
HYPRE_Int hypre_LGMRESSetAugDim_dbl  ( void *lgmres_vdata, HYPRE_Int aug_dim );
HYPRE_Int hypre_LGMRESSetAugDim_long_dbl  ( void *lgmres_vdata, HYPRE_Int aug_dim );
HYPRE_Int hypre_LGMRESSetConvergenceFactorTol_flt  ( void *lgmres_vdata, hypre_float cf_tol );
HYPRE_Int hypre_LGMRESSetConvergenceFactorTol_dbl  ( void *lgmres_vdata, hypre_double cf_tol );
HYPRE_Int hypre_LGMRESSetConvergenceFactorTol_long_dbl  ( void *lgmres_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_LGMRESSetKDim_flt  ( void *lgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_LGMRESSetKDim_dbl  ( void *lgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_LGMRESSetKDim_long_dbl  ( void *lgmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_LGMRESSetLogging_flt  ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESSetLogging_dbl  ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESSetLogging_long_dbl  ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESSetMaxIter_flt  ( void *lgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_LGMRESSetMaxIter_dbl  ( void *lgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_LGMRESSetMaxIter_long_dbl  ( void *lgmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_LGMRESSetMinIter_flt  ( void *lgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_LGMRESSetMinIter_dbl  ( void *lgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_LGMRESSetMinIter_long_dbl  ( void *lgmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_LGMRESSetPrecond_flt  ( void *lgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_LGMRESSetPrecond_dbl  ( void *lgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_LGMRESSetPrecond_long_dbl  ( void *lgmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                             void*), HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_LGMRESSetPrintLevel_flt  ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESSetPrintLevel_dbl  ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESSetPrintLevel_long_dbl  ( void *lgmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_LGMRESSetStopCrit_flt  ( void *lgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_LGMRESSetStopCrit_dbl  ( void *lgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_LGMRESSetStopCrit_long_dbl  ( void *lgmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_LGMRESSetTol_flt  ( void *lgmres_vdata, hypre_float tol );
HYPRE_Int hypre_LGMRESSetTol_dbl  ( void *lgmres_vdata, hypre_double tol );
HYPRE_Int hypre_LGMRESSetTol_long_dbl  ( void *lgmres_vdata, hypre_long_double tol );
HYPRE_Int hypre_LGMRESSetup_flt  ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSetup_dbl  ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSetup_long_dbl  ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSolve_flt  ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSolve_dbl  ( void *lgmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_LGMRESSolve_long_dbl  ( void *lgmres_vdata, void *A, void *b, void *x );
void* hypre_PCGCreate_flt( hypre_PCGFunctions *pcg_functions );
void* hypre_PCGCreate_dbl( hypre_PCGFunctions *pcg_functions );
void* hypre_PCGCreate_long_dbl( hypre_PCGFunctions *pcg_functions );
HYPRE_Int hypre_PCGDestroy_flt  ( void *pcg_vdata );
HYPRE_Int hypre_PCGDestroy_dbl  ( void *pcg_vdata );
HYPRE_Int hypre_PCGDestroy_long_dbl  ( void *pcg_vdata );
hypre_PCGFunctions *hypre_PCGFunctionsCreate_flt(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_float alpha, void *A,
                                      void *x, hypre_float beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_float   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_float alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_float alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_PCGFunctions *hypre_PCGFunctionsCreate_dbl(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_double alpha, void *A,
                                      void *x, hypre_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_PCGFunctions *hypre_PCGFunctionsCreate_long_dbl(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_long_double alpha, void *A,
                                      void *x, hypre_long_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_long_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_long_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_long_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
HYPRE_Int hypre_PCGGetAbsoluteTol_flt  ( void *pcg_vdata, hypre_float *a_tol );
HYPRE_Int hypre_PCGGetAbsoluteTol_dbl  ( void *pcg_vdata, hypre_double *a_tol );
HYPRE_Int hypre_PCGGetAbsoluteTol_long_dbl  ( void *pcg_vdata, hypre_long_double *a_tol );
HYPRE_Int hypre_PCGGetAbsoluteTolFactor_flt  ( void *pcg_vdata, hypre_float *atolf );
HYPRE_Int hypre_PCGGetAbsoluteTolFactor_dbl  ( void *pcg_vdata, hypre_double *atolf );
HYPRE_Int hypre_PCGGetAbsoluteTolFactor_long_dbl  ( void *pcg_vdata, hypre_long_double *atolf );
HYPRE_Int hypre_PCGGetConverged_flt  ( void *pcg_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_PCGGetConverged_dbl  ( void *pcg_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_PCGGetConverged_long_dbl  ( void *pcg_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_PCGGetConvergenceFactorTol_flt  ( void *pcg_vdata, hypre_float *cf_tol );
HYPRE_Int hypre_PCGGetConvergenceFactorTol_dbl  ( void *pcg_vdata, hypre_double *cf_tol );
HYPRE_Int hypre_PCGGetConvergenceFactorTol_long_dbl  ( void *pcg_vdata, hypre_long_double *cf_tol );
HYPRE_Int hypre_PCGGetFinalRelativeResidualNorm_flt  ( void *pcg_vdata,
                                                  hypre_float *relative_residual_norm );
HYPRE_Int hypre_PCGGetFinalRelativeResidualNorm_dbl  ( void *pcg_vdata,
                                                  hypre_double *relative_residual_norm );
HYPRE_Int hypre_PCGGetFinalRelativeResidualNorm_long_dbl  ( void *pcg_vdata,
                                                  hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_PCGGetLogging_flt  ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGGetLogging_dbl  ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGGetLogging_long_dbl  ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGGetMaxIter_flt  ( void *pcg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PCGGetMaxIter_dbl  ( void *pcg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PCGGetMaxIter_long_dbl  ( void *pcg_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_PCGGetNumIterations_flt  ( void *pcg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PCGGetNumIterations_dbl  ( void *pcg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PCGGetNumIterations_long_dbl  ( void *pcg_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_PCGGetPrecond_flt  ( void *pcg_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_PCGGetPrecond_dbl  ( void *pcg_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_PCGGetPrecond_long_dbl  ( void *pcg_vdata, HYPRE_Solver *precond_data_ptr );
HYPRE_Int hypre_PCGGetPrecondMatrix_flt( void  *pcg_vdata,  HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_PCGGetPrecondMatrix_dbl( void  *pcg_vdata,  HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_PCGGetPrecondMatrix_long_dbl( void  *pcg_vdata,  HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_PCGGetPrintLevel_flt  ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGGetPrintLevel_dbl  ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGGetPrintLevel_long_dbl  ( void *pcg_vdata, HYPRE_Int *level );
HYPRE_Int hypre_PCGGetRecomputeResidual_flt  ( void *pcg_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_PCGGetRecomputeResidual_dbl  ( void *pcg_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_PCGGetRecomputeResidual_long_dbl  ( void *pcg_vdata, HYPRE_Int *recompute_residual );
HYPRE_Int hypre_PCGGetRecomputeResidualP_flt  ( void *pcg_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_PCGGetRecomputeResidualP_dbl  ( void *pcg_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_PCGGetRecomputeResidualP_long_dbl  ( void *pcg_vdata, HYPRE_Int *recompute_residual_p );
HYPRE_Int hypre_PCGGetRelChange_flt  ( void *pcg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PCGGetRelChange_dbl  ( void *pcg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PCGGetRelChange_long_dbl  ( void *pcg_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_PCGGetResidual_flt  ( void *pcg_vdata, void **residual );
HYPRE_Int hypre_PCGGetResidual_dbl  ( void *pcg_vdata, void **residual );
HYPRE_Int hypre_PCGGetResidual_long_dbl  ( void *pcg_vdata, void **residual );
HYPRE_Int hypre_PCGGetResidualTol_flt  ( void *pcg_vdata, hypre_float *rtol );
HYPRE_Int hypre_PCGGetResidualTol_dbl  ( void *pcg_vdata, hypre_double *rtol );
HYPRE_Int hypre_PCGGetResidualTol_long_dbl  ( void *pcg_vdata, hypre_long_double *rtol );
HYPRE_Int hypre_PCGGetStopCrit_flt  ( void *pcg_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_PCGGetStopCrit_dbl  ( void *pcg_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_PCGGetStopCrit_long_dbl  ( void *pcg_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_PCGGetTol_flt  ( void *pcg_vdata, hypre_float *tol );
HYPRE_Int hypre_PCGGetTol_dbl  ( void *pcg_vdata, hypre_double *tol );
HYPRE_Int hypre_PCGGetTol_long_dbl  ( void *pcg_vdata, hypre_long_double *tol );
HYPRE_Int hypre_PCGGetTwoNorm_flt  ( void *pcg_vdata, HYPRE_Int *two_norm );
HYPRE_Int hypre_PCGGetTwoNorm_dbl  ( void *pcg_vdata, HYPRE_Int *two_norm );
HYPRE_Int hypre_PCGGetTwoNorm_long_dbl  ( void *pcg_vdata, HYPRE_Int *two_norm );
HYPRE_Int hypre_PCGPrintLogging_flt  ( void *pcg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PCGPrintLogging_dbl  ( void *pcg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PCGPrintLogging_long_dbl  ( void *pcg_vdata, HYPRE_Int myid );
HYPRE_Int hypre_PCGSetAbsoluteTol_flt  ( void *pcg_vdata, hypre_float a_tol );
HYPRE_Int hypre_PCGSetAbsoluteTol_dbl  ( void *pcg_vdata, hypre_double a_tol );
HYPRE_Int hypre_PCGSetAbsoluteTol_long_dbl  ( void *pcg_vdata, hypre_long_double a_tol );
HYPRE_Int hypre_PCGSetAbsoluteTolFactor_flt  ( void *pcg_vdata, hypre_float atolf );
HYPRE_Int hypre_PCGSetAbsoluteTolFactor_dbl  ( void *pcg_vdata, hypre_double atolf );
HYPRE_Int hypre_PCGSetAbsoluteTolFactor_long_dbl  ( void *pcg_vdata, hypre_long_double atolf );
HYPRE_Int hypre_PCGSetConvergenceFactorTol_flt  ( void *pcg_vdata, hypre_float cf_tol );
HYPRE_Int hypre_PCGSetConvergenceFactorTol_dbl  ( void *pcg_vdata, hypre_double cf_tol );
HYPRE_Int hypre_PCGSetConvergenceFactorTol_long_dbl  ( void *pcg_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_PCGSetHybrid_flt  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetHybrid_dbl  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetHybrid_long_dbl  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetLogging_flt  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetLogging_dbl  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetLogging_long_dbl  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetMaxIter_flt  ( void *pcg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PCGSetMaxIter_dbl  ( void *pcg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PCGSetMaxIter_long_dbl  ( void *pcg_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_PCGSetPrecond_flt  ( void *pcg_vdata, HYPRE_Int (*precond )(void*, void*, void*, void*),
                                HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_PCGSetPrecond_dbl  ( void *pcg_vdata, HYPRE_Int (*precond )(void*, void*, void*, void*),
                                HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_PCGSetPrecond_long_dbl  ( void *pcg_vdata, HYPRE_Int (*precond )(void*, void*, void*, void*),
                                HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_PCGSetPrecondMatrix_flt( void  *pcg_vdata,  void  *precond_matrix );
HYPRE_Int hypre_PCGSetPrecondMatrix_dbl( void  *pcg_vdata,  void  *precond_matrix );
HYPRE_Int hypre_PCGSetPrecondMatrix_long_dbl( void  *pcg_vdata,  void  *precond_matrix );
HYPRE_Int hypre_PCGSetPrintLevel_flt  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetPrintLevel_dbl  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetPrintLevel_long_dbl  ( void *pcg_vdata, HYPRE_Int level );
HYPRE_Int hypre_PCGSetRecomputeResidual_flt  ( void *pcg_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_PCGSetRecomputeResidual_dbl  ( void *pcg_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_PCGSetRecomputeResidual_long_dbl  ( void *pcg_vdata, HYPRE_Int recompute_residual );
HYPRE_Int hypre_PCGSetRecomputeResidualP_flt  ( void *pcg_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_PCGSetRecomputeResidualP_dbl  ( void *pcg_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_PCGSetRecomputeResidualP_long_dbl  ( void *pcg_vdata, HYPRE_Int recompute_residual_p );
HYPRE_Int hypre_PCGSetRelChange_flt  ( void *pcg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PCGSetRelChange_dbl  ( void *pcg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PCGSetRelChange_long_dbl  ( void *pcg_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_PCGSetResidualTol_flt  ( void *pcg_vdata, hypre_float rtol );
HYPRE_Int hypre_PCGSetResidualTol_dbl  ( void *pcg_vdata, hypre_double rtol );
HYPRE_Int hypre_PCGSetResidualTol_long_dbl  ( void *pcg_vdata, hypre_long_double rtol );
HYPRE_Int hypre_PCGSetStopCrit_flt  ( void *pcg_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_PCGSetStopCrit_dbl  ( void *pcg_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_PCGSetStopCrit_long_dbl  ( void *pcg_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_PCGSetTol_flt  ( void *pcg_vdata, hypre_float tol );
HYPRE_Int hypre_PCGSetTol_dbl  ( void *pcg_vdata, hypre_double tol );
HYPRE_Int hypre_PCGSetTol_long_dbl  ( void *pcg_vdata, hypre_long_double tol );
HYPRE_Int hypre_PCGSetTwoNorm_flt  ( void *pcg_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_PCGSetTwoNorm_dbl  ( void *pcg_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_PCGSetTwoNorm_long_dbl  ( void *pcg_vdata, HYPRE_Int two_norm );
HYPRE_Int hypre_PCGSetup_flt  ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSetup_dbl  ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSetup_long_dbl  ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSolve_flt  ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSolve_dbl  ( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_PCGSolve_long_dbl  ( void *pcg_vdata, void *A, void *b, void *x );

void *hypre_GMRESCreate_flt( hypre_GMRESFunctions *gmres_functions );
void *hypre_GMRESCreate_dbl( hypre_GMRESFunctions *gmres_functions );
void *hypre_GMRESCreate_long_dbl( hypre_GMRESFunctions *gmres_functions );
HYPRE_Int hypre_GMRESDestroy_flt  ( void *gmres_vdata );
HYPRE_Int hypre_GMRESDestroy_dbl  ( void *gmres_vdata );
HYPRE_Int hypre_GMRESDestroy_long_dbl  ( void *gmres_vdata );
hypre_FlexGMRESFunctions *hypre_GMRESFunctionsCreate_flt(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_float alpha, void *A,
                                      void *x, hypre_float beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_float   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_float alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_float alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_FlexGMRESFunctions *hypre_GMRESFunctionsCreate_dbl(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_double alpha, void *A,
                                      void *x, hypre_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
hypre_FlexGMRESFunctions *hypre_GMRESFunctionsCreate_long_dbl(
       void *(*CAlloc)        ( size_t count, size_t elt_size, HYPRE_MemoryLocation location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
                                      HYPRE_Int   *num_procs ),
      void *(*CreateVector)  ( void *vector ),
      void *(*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *(*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, hypre_long_double alpha, void *A,
                                      void *x, hypre_long_double beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      hypre_long_double   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( hypre_long_double alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( hypre_long_double alpha, void *x, void *y ),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );
HYPRE_Int hypre_GMRESGetAbsoluteTol_flt  ( void *gmres_vdata, hypre_float *a_tol );
HYPRE_Int hypre_GMRESGetAbsoluteTol_dbl  ( void *gmres_vdata, hypre_double *a_tol );
HYPRE_Int hypre_GMRESGetAbsoluteTol_long_dbl  ( void *gmres_vdata, hypre_long_double *a_tol );
HYPRE_Int hypre_GMRESGetConverged_flt  ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_GMRESGetConverged_dbl  ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_GMRESGetConverged_long_dbl  ( void *gmres_vdata, HYPRE_Int *converged );
HYPRE_Int hypre_GMRESGetConvergenceFactorTol_flt  ( void *gmres_vdata, hypre_float *cf_tol );
HYPRE_Int hypre_GMRESGetConvergenceFactorTol_dbl  ( void *gmres_vdata, hypre_double *cf_tol );
HYPRE_Int hypre_GMRESGetConvergenceFactorTol_long_dbl  ( void *gmres_vdata, hypre_long_double *cf_tol );
HYPRE_Int hypre_GMRESGetFinalRelativeResidualNorm_flt  ( void *gmres_vdata,
                                                    hypre_float *relative_residual_norm );
HYPRE_Int hypre_GMRESGetFinalRelativeResidualNorm_dbl  ( void *gmres_vdata,
                                                    hypre_double *relative_residual_norm );
HYPRE_Int hypre_GMRESGetFinalRelativeResidualNorm_long_dbl  ( void *gmres_vdata,
                                                    hypre_long_double *relative_residual_norm );
HYPRE_Int hypre_GMRESGetKDim_flt  ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_GMRESGetKDim_dbl  ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_GMRESGetKDim_long_dbl  ( void *gmres_vdata, HYPRE_Int *k_dim );
HYPRE_Int hypre_GMRESGetLogging_flt  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESGetLogging_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESGetLogging_long_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESGetMaxIter_flt  ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_GMRESGetMaxIter_dbl  ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_GMRESGetMaxIter_long_dbl  ( void *gmres_vdata, HYPRE_Int *max_iter );
HYPRE_Int hypre_GMRESGetMinIter_flt  ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_GMRESGetMinIter_dbl  ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_GMRESGetMinIter_long_dbl  ( void *gmres_vdata, HYPRE_Int *min_iter );
HYPRE_Int hypre_GMRESGetNumIterations_flt  ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_GMRESGetNumIterations_dbl  ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_GMRESGetNumIterations_long_dbl  ( void *gmres_vdata, HYPRE_Int *num_iterations );
HYPRE_Int hypre_GMRESGetPrintLevel_flt  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESGetPrintLevel_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESGetPrintLevel_long_dbl  ( void *gmres_vdata, HYPRE_Int *level );
HYPRE_Int hypre_GMRESGetRelChange_flt  ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_GMRESGetRelChange_dbl  ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_GMRESGetRelChange_long_dbl  ( void *gmres_vdata, HYPRE_Int *rel_change );
HYPRE_Int hypre_GMRESGetResidual_flt  ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_GMRESGetResidual_dbl  ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_GMRESGetResidual_long_dbl  ( void *gmres_vdata, void **residual );
HYPRE_Int hypre_GMRESGetSkipRealResidualCheck_flt  ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_GMRESGetSkipRealResidualCheck_dbl  ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_GMRESGetSkipRealResidualCheck_long_dbl  ( void *gmres_vdata, HYPRE_Int *skip_real_r_check );
HYPRE_Int hypre_GMRESGetStopCrit_flt  ( void *gmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_GMRESGetStopCrit_dbl  ( void *gmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_GMRESGetStopCrit_long_dbl  ( void *gmres_vdata, HYPRE_Int *stop_crit );
HYPRE_Int hypre_GMRESGetTol_flt  ( void *gmres_vdata, hypre_float *tol );
HYPRE_Int hypre_GMRESGetTol_dbl  ( void *gmres_vdata, hypre_double *tol );
HYPRE_Int hypre_GMRESGetTol_long_dbl  ( void *gmres_vdata, hypre_long_double *tol );
HYPRE_Int hypre_GMRESSetAbsoluteTol_flt  ( void *gmres_vdata, hypre_float a_tol );
HYPRE_Int hypre_GMRESSetAbsoluteTol_dbl  ( void *gmres_vdata, hypre_double a_tol );
HYPRE_Int hypre_GMRESSetAbsoluteTol_long_dbl  ( void *gmres_vdata, hypre_long_double a_tol );
HYPRE_Int hypre_GMRESSetConvergenceFactorTol_flt  ( void *gmres_vdata, hypre_float cf_tol );
HYPRE_Int hypre_GMRESSetConvergenceFactorTol_dbl  ( void *gmres_vdata, hypre_double cf_tol );
HYPRE_Int hypre_GMRESSetConvergenceFactorTol_long_dbl  ( void *gmres_vdata, hypre_long_double cf_tol );
HYPRE_Int hypre_GMRESSetHybrid_flt  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetHybrid_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetHybrid_long_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetKDim_flt  ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_GMRESSetKDim_dbl  ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_GMRESSetKDim_long_dbl  ( void *gmres_vdata, HYPRE_Int k_dim );
HYPRE_Int hypre_GMRESSetLogging_flt  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetLogging_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetLogging_long_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetMaxIter_flt  ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_GMRESSetMaxIter_dbl  ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_GMRESSetMaxIter_long_dbl  ( void *gmres_vdata, HYPRE_Int max_iter );
HYPRE_Int hypre_GMRESSetMinIter_flt  ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_GMRESSetMinIter_dbl  ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_GMRESSetMinIter_long_dbl  ( void *gmres_vdata, HYPRE_Int min_iter );
HYPRE_Int hypre_GMRESSetPrecond_flt  ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                           void*),
                                  HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_GMRESSetPrecond_dbl  ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                           void*),
                                  HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_GMRESSetPrecond_long_dbl  ( void *gmres_vdata, HYPRE_Int (*precond )(void*, void*, void*,
                                                                           void*),
                                  HYPRE_Int (*precond_setup )(void*, void*, void*, void*), void *precond_data );
HYPRE_Int hypre_GMRESSetPrecondMatrix_flt( void  *gmres_vdata,  void  *precond_matrix );
HYPRE_Int hypre_GMRESSetPrecondMatrix_dbl( void  *gmres_vdata,  void  *precond_matrix );
HYPRE_Int hypre_GMRESSetPrecondMatrix_long_dbl( void  *gmres_vdata,  void  *precond_matrix );
HYPRE_Int hypre_GMRESGetPrecondMatrix_flt( void  *gmres_vdata,  HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_GMRESGetPrecondMatrix_dbl( void  *gmres_vdata,  HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_GMRESGetPrecondMatrix_long_dbl( void  *gmres_vdata,  HYPRE_Matrix *precond_matrix_ptr );
HYPRE_Int hypre_GMRESSetPrintLevel_flt  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetPrintLevel_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetPrintLevel_long_dbl  ( void *gmres_vdata, HYPRE_Int level );
HYPRE_Int hypre_GMRESSetRelChange_flt  ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_GMRESSetRelChange_dbl  ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_GMRESSetRelChange_long_dbl  ( void *gmres_vdata, HYPRE_Int rel_change );
HYPRE_Int hypre_GMRESSetSkipRealResidualCheck_flt  ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_GMRESSetSkipRealResidualCheck_dbl  ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_GMRESSetSkipRealResidualCheck_long_dbl  ( void *gmres_vdata, HYPRE_Int skip_real_r_check );
HYPRE_Int hypre_GMRESSetStopCrit_flt  ( void *gmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_GMRESSetStopCrit_dbl  ( void *gmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_GMRESSetStopCrit_long_dbl  ( void *gmres_vdata, HYPRE_Int stop_crit );
HYPRE_Int hypre_GMRESSetTol_flt  ( void *gmres_vdata, hypre_float tol );
HYPRE_Int hypre_GMRESSetTol_dbl  ( void *gmres_vdata, hypre_double tol );
HYPRE_Int hypre_GMRESSetTol_long_dbl  ( void *gmres_vdata, hypre_long_double tol );
HYPRE_Int hypre_GMRESSetup_flt  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSetup_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSetup_long_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSolve_flt  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSolve_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
HYPRE_Int hypre_GMRESSolve_long_dbl  ( void *gmres_vdata, void *A, void *b, void *x );
#endif

#endif
