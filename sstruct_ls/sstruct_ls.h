
#include <HYPRE_config.h>

#include "HYPRE_sstruct_ls.h"

#ifndef hypre_SSTRUCT_LS_HEADER
#define hypre_SSTRUCT_LS_HEADER

#include "utilities.h"
#include "krylov.h"
#include "sstruct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif


/* HYPRE_sstruct_gmres.c */
int HYPRE_SStructGMRESCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructGMRESDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructGMRESSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructGMRESSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructGMRESSetKDim( HYPRE_SStructSolver solver , int k_dim );
int HYPRE_SStructGMRESSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructGMRESSetMinIter( HYPRE_SStructSolver solver , int min_iter );
int HYPRE_SStructGMRESSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructGMRESSetStopCrit( HYPRE_SStructSolver solver , int stop_crit );
int HYPRE_SStructGMRESSetPrecond( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructGMRESSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructGMRESGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_pcg.c */
int HYPRE_SStructPCGCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructPCGDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructPCGSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructPCGSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructPCGSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructPCGSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructPCGSetTwoNorm( HYPRE_SStructSolver solver , int two_norm );
int HYPRE_SStructPCGSetRelChange( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructPCGSetPrecond( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructPCGSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructPCGGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructPCGGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructDiagScaleSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );
int HYPRE_SStructDiagScale( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );

/* HYPRE_sstruct_split.c */
int HYPRE_SStructSplitCreate( MPI_Comm comm , HYPRE_SStructSolver *solver_ptr );
int HYPRE_SStructSplitDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSplitSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSplitSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructSplitSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructSplitSetZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetNonZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetStructSolver( HYPRE_SStructSolver solver , int ssolver );
int HYPRE_SStructSplitGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructSplitGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_sys_pfmg.c */
int HYPRE_SStructSysPFMGCreate( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructSysPFMGDestroy( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetup( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSysPFMGSolve( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSysPFMGSetTol( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructSysPFMGSetMaxIter( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructSysPFMGSetRelChange( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructSysPFMGSetZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetNonZeroGuess( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetRelaxType( HYPRE_SStructSolver solver , int relax_type );
int HYPRE_SStructSysPFMGSetNumPreRelax( HYPRE_SStructSolver solver , int num_pre_relax );
int HYPRE_SStructSysPFMGSetNumPostRelax( HYPRE_SStructSolver solver , int num_post_relax );
int HYPRE_SStructSysPFMGSetSkipRelax( HYPRE_SStructSolver solver , int skip_relax );
int HYPRE_SStructSysPFMGSetDxyz( HYPRE_SStructSolver solver , double *dxyz );
int HYPRE_SStructSysPFMGSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructSysPFMGGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );

/* krylov.c */
int hypre_SStructKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_SStructKrylovIdentity( void *vdata , void *A , void *b , void *x );

/* krylov_sstruct.c */
char *hypre_SStructKrylovCAlloc( int count , int elt_size );
int hypre_SStructKrylovFree( char *ptr );
void *hypre_SStructKrylovCreateVector( void *vvector );
void *hypre_SStructKrylovCreateVectorArray( int n , void *vvector );
int hypre_SStructKrylovDestroyVector( void *vvector );
void *hypre_SStructKrylovMatvecCreate( void *A , void *x );
int hypre_SStructKrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_SStructKrylovMatvecDestroy( void *matvec_data );
double hypre_SStructKrylovInnerProd( void *x , void *y );
int hypre_SStructKrylovCopyVector( void *x , void *y );
int hypre_SStructKrylovClearVector( void *x );
int hypre_SStructKrylovScaleVector( double alpha , void *x );
int hypre_SStructKrylovAxpy( double alpha , void *x , void *y );
int hypre_SStructKrylovCommInfo( void *A , int *my_id , int *num_procs );

/* node_relax.c */
void *hypre_NodeRelaxCreate( MPI_Comm comm );
int hypre_NodeRelaxDestroy( void *relax_vdata );
int hypre_NodeRelaxSetup( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_NodeRelax( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_NodeRelaxSetTol( void *relax_vdata , double tol );
int hypre_NodeRelaxSetMaxIter( void *relax_vdata , int max_iter );
int hypre_NodeRelaxSetZeroGuess( void *relax_vdata , int zero_guess );
int hypre_NodeRelaxSetWeight( void *relax_vdata , double weight );
int hypre_NodeRelaxSetNumNodesets( void *relax_vdata , int num_nodesets );
int hypre_NodeRelaxSetNodeset( void *relax_vdata , int nodeset , int nodeset_size , hypre_Index nodeset_stride , hypre_Index *nodeset_indices );
int hypre_NodeRelaxSetNodesetRank( void *relax_vdata , int nodeset , int nodeset_rank );
int hypre_NodeRelaxSetTempVec( void *relax_vdata , hypre_SStructPVector *t );

/* sys_pfmg.c */
void *hypre_SysPFMGCreate( MPI_Comm comm );
int hypre_SysPFMGDestroy( void *sys_pfmg_vdata );
int hypre_SysPFMGSetTol( void *sys_pfmg_vdata , double tol );
int hypre_SysPFMGSetMaxIter( void *sys_pfmg_vdata , int max_iter );
int hypre_SysPFMGSetRelChange( void *sys_pfmg_vdata , int rel_change );
int hypre_SysPFMGSetZeroGuess( void *sys_pfmg_vdata , int zero_guess );
int hypre_SysPFMGSetRelaxType( void *sys_pfmg_vdata , int relax_type );
int hypre_SysPFMGSetNumPreRelax( void *sys_pfmg_vdata , int num_pre_relax );
int hypre_SysPFMGSetNumPostRelax( void *sys_pfmg_vdata , int num_post_relax );
int hypre_SysPFMGSetSkipRelax( void *sys_pfmg_vdata , int skip_relax );
int hypre_SysPFMGSetDxyz( void *sys_pfmg_vdata , double *dxyz );
int hypre_SysPFMGSetLogging( void *sys_pfmg_vdata , int logging );
int hypre_SysPFMGGetNumIterations( void *sys_pfmg_vdata , int *num_iterations );
int hypre_SysPFMGPrintLogging( void *sys_pfmg_vdata , int myid );
int hypre_SysPFMGGetFinalRelativeResidualNorm( void *sys_pfmg_vdata , double *relative_residual_norm );

/* sys_pfmg_relax.c */
void *hypre_SysPFMGRelaxCreate( MPI_Comm comm );
int hypre_SysPFMGRelaxDestroy( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelax( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_SysPFMGRelaxSetup( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_SysPFMGRelaxSetType( void *sys_pfmg_relax_vdata , int relax_type );
int hypre_SysPFMGRelaxSetPreRelax( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelaxSetPostRelax( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelaxSetTol( void *sys_pfmg_relax_vdata , double tol );
int hypre_SysPFMGRelaxSetMaxIter( void *sys_pfmg_relax_vdata , int max_iter );
int hypre_SysPFMGRelaxSetZeroGuess( void *sys_pfmg_relax_vdata , int zero_guess );
int hypre_SysPFMGRelaxSetTempVec( void *sys_pfmg_relax_vdata , hypre_SStructPVector *t );

/* sys_pfmg_setup.c */
int hypre_SysPFMGSetup( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );
int hypre_SysStructCoarsen( hypre_SStructPGrid *fgrid , hypre_Index index , hypre_Index stride , int prune , hypre_SStructPGrid **cgrid_ptr );

/* sys_pfmg_setup_interp.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateInterpOp( hypre_SStructPMatrix *A , hypre_SStructPGrid *cgrid , int cdir );
int hypre_SysPFMGSetupInterpOp( hypre_SStructPMatrix *A , int cdir , hypre_Index findex , hypre_Index stride , hypre_SStructPMatrix *P );

/* sys_pfmg_setup_rap.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateRAPOp( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , hypre_SStructPGrid *coarse_grid , int cdir );
int hypre_SysPFMGSetupRAPOp( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_SStructPMatrix *Ac );

/* sys_pfmg_solve.c */
int hypre_SysPFMGSolve( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );

/* sys_semi_interp.c */
int hypre_SysSemiInterpCreate( void **sys_interp_vdata_ptr );
int hypre_SysSemiInterpSetup( void *sys_interp_vdata , hypre_SStructPMatrix *P , int P_stored_as_transpose , hypre_SStructPVector *xc , hypre_SStructPVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SysSemiInterp( void *sys_interp_vdata , hypre_SStructPMatrix *P , hypre_SStructPVector *xc , hypre_SStructPVector *e );
int hypre_SysSemiInterpDestroy( void *sys_interp_vdata );

/* sys_semi_restrict.c */
int hypre_SysSemiRestrictCreate( void **sys_restrict_vdata_ptr );
int hypre_SysSemiRestrictSetup( void *sys_restrict_vdata , hypre_SStructPMatrix *R , int R_stored_as_transpose , hypre_SStructPVector *r , hypre_SStructPVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SysSemiRestrict( void *sys_restrict_vdata , hypre_SStructPMatrix *R , hypre_SStructPVector *r , hypre_SStructPVector *rc );
int hypre_SysSemiRestrictDestroy( void *sys_restrict_vdata );


#ifdef __cplusplus
}
#endif

#endif

