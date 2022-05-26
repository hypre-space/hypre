/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_FEI__
#define hypre_FEI__

#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* bicgs.c */
void * hypre_BiCGSCreate( );
int hypre_BiCGSDestroy( void *bicgs_vdata );
int hypre_BiCGSSetup( void *bicgs_vdata, void *A, void *b, void *x         );
int hypre_BiCGSSolve(void  *bicgs_vdata, void  *A, void  *b, void  *x);
int hypre_BiCGSSetTol( void *bicgs_vdata, double tol );
int hypre_BiCGSSetMaxIter( void *bicgs_vdata, int max_iter );
int hypre_BiCGSSetStopCrit( void *bicgs_vdata, double stop_crit );
int hypre_BiCGSSetPrecond( void  *bicgs_vdata, int  (*precond)(void*,void*,void*,void*),
						   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data );
int hypre_BiCGSSetLogging( void *bicgs_vdata, int logging);
int hypre_BiCGSGetNumIterations(void *bicgs_vdata,int  *num_iterations);
int hypre_BiCGSGetFinalRelativeResidualNorm( void   *bicgs_vdata,
											 double *relative_residual_norm );

/* bicgstabl.c */
	void * hypre_BiCGSTABLCreate( );
	int hypre_BiCGSTABLDestroy( void *bicgstab_vdata );
	int hypre_BiCGSTABLSetup( void *bicgstab_vdata, void *A, void *b, void *x         );
	int hypre_BiCGSTABLSolve(void  *bicgstab_vdata, void  *A, void  *b, void  *x);
	int hypre_BiCGSTABLSetSize( void *bicgstab_vdata, int size );
	int hypre_BiCGSTABLSetMaxIter( void *bicgstab_vdata, int max_iter );
	int hypre_BiCGSTABLSetStopCrit( void *bicgstab_vdata, double stop_crit );
	int hypre_BiCGSTABLSetPrecond( void  *bicgstab_vdata, int  (*precond)(void*, void*, void*, void*),
								   int  (*precond_setup)(void*, void*, void*, void*), void  *precond_data );
	int hypre_BiCGSTABLSetLogging( void *bicgstab_vdata, int logging);
	int hypre_BiCGSTABLGetNumIterations(void *bicgstab_vdata,int  *num_iterations);
	int hypre_BiCGSTABLGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
													 double *relative_residual_norm );
/* fgmres.c */
void *hypre_FGMRESCreate();
int hypre_FGMRESDestroy( void *fgmres_vdata );
int hypre_FGMRESSetup( void *fgmres_vdata, void *A, void *b, void *x );
int hypre_FGMRESSolve(void  *fgmres_vdata, void  *A, void  *b, void  *x);
int hypre_FGMRESSetKDim( void *fgmres_vdata, int k_dim );
int hypre_FGMRESSetTol( void *fgmres_vdata, double tol );
int hypre_FGMRESSetMaxIter( void *fgmres_vdata, int max_iter );
int hypre_FGMRESSetStopCrit( void *fgmres_vdata, double stop_crit );
int hypre_FGMRESSetPrecond( void *fgmres_vdata, int (*precond)(void*,void*,void*,void*),
								int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data );
int hypre_FGMRESGetPrecond(void *fgmres_vdata, HYPRE_Solver *precond_data_ptr);
int hypre_FGMRESSetLogging( void *fgmres_vdata, int logging );	
int hypre_FGMRESGetNumIterations( void *fgmres_vdata, int *num_iterations );
int hypre_FGMRESGetFinalRelativeResidualNorm(void *fgmres_vdata,
												 double *relative_residual_norm );
int hypre_FGMRESUpdatePrecondTolerance(void *fgmres_vdata, int (*update_tol)(int*, double));

/* TFQmr.c */
void * hypre_TFQmrCreate();
int hypre_TFQmrDestroy( void *tfqmr_vdata );
int hypre_TFQmrSetup( void *tfqmr_vdata, void *A, void *b, void *x         );
int hypre_TFQmrSolve(void  *tfqmr_vdata, void  *A, void  *b, void  *x);
int hypre_TFQmrSetTol( void *tfqmr_vdata, double tol );
int hypre_TFQmrSetMaxIter( void *tfqmr_vdata, int max_iter );
int hypre_TFQmrSetStopCrit( void *tfqmr_vdata, double stop_crit );
int hypre_TFQmrSetPrecond( void  *tfqmr_vdata, int  (*precond)(void*,void*,void*,void*),
						   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data );
int hypre_TFQmrSetLogging( void *tfqmr_vdata, int logging);
int hypre_TFQmrGetNumIterations(void *tfqmr_vdata,int  *num_iterations);
int hypre_TFQmrGetFinalRelativeResidualNorm( void   *tfqmr_vdata,
											 double *relative_residual_norm );

#ifdef __cplusplus
}
#endif

#endif
