
#include <HYPRE_config.h>

#include "HYPRE_struct_ls.h"

#ifndef hypre_STRUCT_LS_HEADER
#define hypre_STRUCT_LS_HEADER

#include "utilities.h"
#include "struct_mv.h"
#include "krylov.h"

#ifdef __cplusplus
extern "C" {
#endif


/* HYPRE_struct_gmres.c */
int HYPRE_StructGMRESCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructGMRESDestroy( HYPRE_StructSolver solver );
int HYPRE_StructGMRESSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructGMRESSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructGMRESSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructGMRESSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructGMRESSetPrecond( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
int HYPRE_StructGMRESSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructGMRESGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructGMRESGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_hybrid.c */
int HYPRE_StructHybridCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructHybridDestroy( HYPRE_StructSolver solver );
int HYPRE_StructHybridSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructHybridSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructHybridSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructHybridSetConvergenceTol( HYPRE_StructSolver solver , double cf_tol );
int HYPRE_StructHybridSetDSCGMaxIter( HYPRE_StructSolver solver , int dscg_max_its );
int HYPRE_StructHybridSetPCGMaxIter( HYPRE_StructSolver solver , int pcg_max_its );
int HYPRE_StructHybridSetTwoNorm( HYPRE_StructSolver solver , int two_norm );
int HYPRE_StructHybridSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructHybridSetPrecond( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
int HYPRE_StructHybridSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructHybridGetNumIterations( HYPRE_StructSolver solver , int *num_its );
int HYPRE_StructHybridGetDSCGNumIterations( HYPRE_StructSolver solver , int *dscg_num_its );
int HYPRE_StructHybridGetPCGNumIterations( HYPRE_StructSolver solver , int *pcg_num_its );
int HYPRE_StructHybridGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_jacobi.c */
int HYPRE_StructJacobiCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructJacobiDestroy( HYPRE_StructSolver solver );
int HYPRE_StructJacobiSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructJacobiSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructJacobiSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructJacobiSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructJacobiSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructJacobiSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructJacobiGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructJacobiGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_pcg.c */
int HYPRE_StructPCGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructPCGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructPCGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPCGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPCGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructPCGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructPCGSetTwoNorm( HYPRE_StructSolver solver , int two_norm );
int HYPRE_StructPCGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructPCGSetPrecond( HYPRE_StructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
int HYPRE_StructPCGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructPCGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructPCGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );
int HYPRE_StructDiagScaleSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x );
int HYPRE_StructDiagScale( HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx );

/* HYPRE_struct_pfmg.c */
int HYPRE_StructPFMGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructPFMGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructPFMGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPFMGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPFMGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructPFMGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructPFMGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructPFMGSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructPFMGSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructPFMGSetRelaxType( HYPRE_StructSolver solver , int relax_type );
int HYPRE_StructPFMGSetNumPreRelax( HYPRE_StructSolver solver , int num_pre_relax );
int HYPRE_StructPFMGSetNumPostRelax( HYPRE_StructSolver solver , int num_post_relax );
int HYPRE_StructPFMGSetSkipRelax( HYPRE_StructSolver solver , int skip_relax );
int HYPRE_StructPFMGSetDxyz( HYPRE_StructSolver solver , double *dxyz );
int HYPRE_StructPFMGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructPFMGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructPFMGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_smg.c */
int HYPRE_StructSMGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructSMGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructSMGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSMGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSMGSetMemoryUse( HYPRE_StructSolver solver , int memory_use );
int HYPRE_StructSMGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructSMGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructSMGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructSMGSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSMGSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSMGSetNumPreRelax( HYPRE_StructSolver solver , int num_pre_relax );
int HYPRE_StructSMGSetNumPostRelax( HYPRE_StructSolver solver , int num_post_relax );
int HYPRE_StructSMGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructSMGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructSMGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_sparse_msg.c */
int HYPRE_StructSparseMSGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructSparseMSGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructSparseMSGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSparseMSGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSparseMSGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructSparseMSGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructSparseMSGSetJump( HYPRE_StructSolver solver , int jump );
int HYPRE_StructSparseMSGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructSparseMSGSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSparseMSGSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSparseMSGSetRelaxType( HYPRE_StructSolver solver , int relax_type );
int HYPRE_StructSparseMSGSetNumPreRelax( HYPRE_StructSolver solver , int num_pre_relax );
int HYPRE_StructSparseMSGSetNumPostRelax( HYPRE_StructSolver solver , int num_post_relax );
int HYPRE_StructSparseMSGSetNumFineRelax( HYPRE_StructSolver solver , int num_fine_relax );
int HYPRE_StructSparseMSGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructSparseMSGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* coarsen.c */
int hypre_StructMapFineToCoarse( hypre_Index findex , hypre_Index index , hypre_Index stride , hypre_Index cindex );
int hypre_StructMapCoarseToFine( hypre_Index cindex , hypre_Index index , hypre_Index stride , hypre_Index findex );
int hypre_StructCoarsen( hypre_StructGrid *fgrid , hypre_Index index , hypre_Index stride , int prune , hypre_StructGrid **cgrid_ptr );
int hypre_StructCoarsen( hypre_StructGrid *fgrid , hypre_Index index , hypre_Index stride , int prune , hypre_StructGrid **cgrid_ptr );

/* cyclic_reduction.c */
void *hypre_CyclicReductionCreate( MPI_Comm comm );
hypre_StructMatrix *hypre_CycRedCreateCoarseOp( hypre_StructMatrix *A , hypre_StructGrid *coarse_grid , int cdir );
int hypre_CycRedSetupCoarseOp( hypre_StructMatrix *A , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride );
int hypre_CyclicReductionSetup( void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_CyclicReduction( void *cyc_red_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_CyclicReductionSetBase( void *cyc_red_vdata , hypre_Index base_index , hypre_Index base_stride );
int hypre_CyclicReductionDestroy( void *cyc_red_vdata );

/* general.c */
int hypre_Log2( int p );

/* hybrid.c */
void *hypre_HybridCreate( MPI_Comm comm );
int hypre_HybridDestroy( void *hybrid_vdata );
int hypre_HybridSetTol( void *hybrid_vdata , double tol );
int hypre_HybridSetConvergenceTol( void *hybrid_vdata , double cf_tol );
int hypre_HybridSetDSCGMaxIter( void *hybrid_vdata , int dscg_max_its );
int hypre_HybridSetPCGMaxIter( void *hybrid_vdata , int pcg_max_its );
int hypre_HybridSetTwoNorm( void *hybrid_vdata , int two_norm );
int hypre_HybridSetRelChange( void *hybrid_vdata , int rel_change );
int hypre_HybridSetPrecond( void *pcg_vdata , int (*pcg_precond_solve )(), int (*pcg_precond_setup )(), void *pcg_precond );
int hypre_HybridSetLogging( void *hybrid_vdata , int logging );
int hypre_HybridGetNumIterations( void *hybrid_vdata , int *num_its );
int hypre_HybridGetDSCGNumIterations( void *hybrid_vdata , int *dscg_num_its );
int hypre_HybridGetPCGNumIterations( void *hybrid_vdata , int *pcg_num_its );
int hypre_HybridGetFinalRelativeResidualNorm( void *hybrid_vdata , double *final_rel_res_norm );
int hypre_HybridSetup( void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_HybridSolve( void *hybrid_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* jacobi.c */
void *hypre_JacobiCreate( MPI_Comm comm );
int hypre_JacobiDestroy( void *jacobi_vdata );
int hypre_JacobiSetup( void *jacobi_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_JacobiSolve( void *jacobi_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_JacobiSetTol( void *jacobi_vdata , double tol );
int hypre_JacobiSetMaxIter( void *jacobi_vdata , int max_iter );
int hypre_JacobiSetZeroGuess( void *jacobi_vdata , int zero_guess );
int hypre_JacobiSetTempVec( void *jacobi_vdata , hypre_StructVector *t );

/* pcg_struct.c */
char *hypre_StructKrylovCAlloc( int count , int elt_size );
int hypre_StructKrylovFree( char *ptr );
void *hypre_StructKrylovCreateVector( void *vvector );
void *hypre_StructKrylovCreateVectorArray( int n , void *vvector );
int hypre_StructKrylovDestroyVector( void *vvector );
void *hypre_StructKrylovMatvecCreate( void *A , void *x );
int hypre_StructKrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_StructKrylovMatvecDestroy( void *matvec_data );
double hypre_StructKrylovInnerProd( void *x , void *y );
int hypre_StructKrylovCopyVector( void *x , void *y );
int hypre_StructKrylovClearVector( void *x );
int hypre_StructKrylovScaleVector( double alpha , void *x );
int hypre_StructKrylovAxpy( double alpha , void *x , void *y );
int hypre_StructKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_StructKrylovIdentity( void *vdata , void *A , void *b , void *x );
int hypre_StructKrylovCommInfo( void *A , int *my_id , int *num_procs );

/* pfmg.c */
void *hypre_PFMGCreate( MPI_Comm comm );
int hypre_PFMGDestroy( void *pfmg_vdata );
int hypre_PFMGSetTol( void *pfmg_vdata , double tol );
int hypre_PFMGSetMaxIter( void *pfmg_vdata , int max_iter );
int hypre_PFMGSetRelChange( void *pfmg_vdata , int rel_change );
int hypre_PFMGSetZeroGuess( void *pfmg_vdata , int zero_guess );
int hypre_PFMGSetRelaxType( void *pfmg_vdata , int relax_type );
int hypre_PFMGSetNumPreRelax( void *pfmg_vdata , int num_pre_relax );
int hypre_PFMGSetNumPostRelax( void *pfmg_vdata , int num_post_relax );
int hypre_PFMGSetSkipRelax( void *pfmg_vdata , int skip_relax );
int hypre_PFMGSetDxyz( void *pfmg_vdata , double *dxyz );
int hypre_PFMGSetLogging( void *pfmg_vdata , int logging );
int hypre_PFMGGetNumIterations( void *pfmg_vdata , int *num_iterations );
int hypre_PFMGPrintLogging( void *pfmg_vdata , int myid );
int hypre_PFMGGetFinalRelativeResidualNorm( void *pfmg_vdata , double *relative_residual_norm );

/* pfmg2_setup_rap.c */
hypre_StructMatrix *hypre_PFMG2CreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir );
int hypre_PFMG2BuildRAPSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
int hypre_PFMG2BuildRAPNoSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );

/* pfmg3_setup_rap.c */
hypre_StructMatrix *hypre_PFMG3CreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir );
int hypre_PFMG3BuildRAPSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );
int hypre_PFMG3BuildRAPNoSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *RAP );

/* pfmg_relax.c */
void *hypre_PFMGRelaxCreate( MPI_Comm comm );
int hypre_PFMGRelaxDestroy( void *pfmg_relax_vdata );
int hypre_PFMGRelax( void *pfmg_relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_PFMGRelaxSetup( void *pfmg_relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_PFMGRelaxSetType( void *pfmg_relax_vdata , int relax_type );
int hypre_PFMGRelaxSetPreRelax( void *pfmg_relax_vdata );
int hypre_PFMGRelaxSetPostRelax( void *pfmg_relax_vdata );
int hypre_PFMGRelaxSetTol( void *pfmg_relax_vdata , double tol );
int hypre_PFMGRelaxSetMaxIter( void *pfmg_relax_vdata , int max_iter );
int hypre_PFMGRelaxSetZeroGuess( void *pfmg_relax_vdata , int zero_guess );
int hypre_PFMGRelaxSetTempVec( void *pfmg_relax_vdata , hypre_StructVector *t );

/* pfmg_setup.c */
int hypre_PFMGSetup( void *pfmg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_PFMGComputeDxyz( hypre_StructMatrix *A , double *dxyz );

/* pfmg_setup_interp.c */
hypre_StructMatrix *hypre_PFMGCreateInterpOp( hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir );
int hypre_PFMGSetupInterpOp( hypre_StructMatrix *A , int cdir , hypre_Index findex , hypre_Index stride , hypre_StructMatrix *P );

/* pfmg_setup_rap.c */
hypre_StructMatrix *hypre_PFMGCreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir );
int hypre_PFMGSetupRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_StructMatrix *Ac );

/* pfmg_solve.c */
int hypre_PFMGSolve( void *pfmg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* point_relax.c */
void *hypre_PointRelaxCreate( MPI_Comm comm );
int hypre_PointRelaxDestroy( void *relax_vdata );
int hypre_PointRelaxSetup( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_PointRelax( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_PointRelaxSetTol( void *relax_vdata , double tol );
int hypre_PointRelaxSetMaxIter( void *relax_vdata , int max_iter );
int hypre_PointRelaxSetZeroGuess( void *relax_vdata , int zero_guess );
int hypre_PointRelaxSetWeight( void *relax_vdata , double weight );
int hypre_PointRelaxSetNumPointsets( void *relax_vdata , int num_pointsets );
int hypre_PointRelaxSetPointset( void *relax_vdata , int pointset , int pointset_size , hypre_Index pointset_stride , hypre_Index *pointset_indices );
int hypre_PointRelaxSetPointsetRank( void *relax_vdata , int pointset , int pointset_rank );
int hypre_PointRelaxSetTempVec( void *relax_vdata , hypre_StructVector *t );

/* semi_interp.c */
void *hypre_SemiInterpCreate( void );
int hypre_SemiInterpSetup( void *interp_vdata , hypre_StructMatrix *P , int P_stored_as_transpose , hypre_StructVector *xc , hypre_StructVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SemiInterp( void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e );
int hypre_SemiInterpDestroy( void *interp_vdata );

/* semi_restrict.c */
void *hypre_SemiRestrictCreate( void );
int hypre_SemiRestrictSetup( void *restrict_vdata , hypre_StructMatrix *R , int R_stored_as_transpose , hypre_StructVector *r , hypre_StructVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SemiRestrict( void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc );
int hypre_SemiRestrictDestroy( void *restrict_vdata );

/* smg.c */
void *hypre_SMGCreate( MPI_Comm comm );
int hypre_SMGDestroy( void *smg_vdata );
int hypre_SMGSetMemoryUse( void *smg_vdata , int memory_use );
int hypre_SMGSetTol( void *smg_vdata , double tol );
int hypre_SMGSetMaxIter( void *smg_vdata , int max_iter );
int hypre_SMGSetRelChange( void *smg_vdata , int rel_change );
int hypre_SMGSetZeroGuess( void *smg_vdata , int zero_guess );
int hypre_SMGSetNumPreRelax( void *smg_vdata , int num_pre_relax );
int hypre_SMGSetNumPostRelax( void *smg_vdata , int num_post_relax );
int hypre_SMGSetBase( void *smg_vdata , hypre_Index base_index , hypre_Index base_stride );
int hypre_SMGSetLogging( void *smg_vdata , int logging );
int hypre_SMGGetNumIterations( void *smg_vdata , int *num_iterations );
int hypre_SMGPrintLogging( void *smg_vdata , int myid );
int hypre_SMGGetFinalRelativeResidualNorm( void *smg_vdata , double *relative_residual_norm );
int hypre_SMGSetStructVectorConstantValues( hypre_StructVector *vector , double values , hypre_BoxArray *box_array , hypre_Index stride );

/* smg2_setup_rap.c */
hypre_StructMatrix *hypre_SMG2CreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructGrid *coarse_grid );
int hypre_SMG2BuildRAPSym( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
int hypre_SMG2BuildRAPNoSym( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
int hypre_SMG2RAPPeriodicSym( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
int hypre_SMG2RAPPeriodicNoSym( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );

/* smg3_setup_rap.c */
hypre_StructMatrix *hypre_SMG3CreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructGrid *coarse_grid );
int hypre_SMG3BuildRAPSym( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
int hypre_SMG3BuildRAPNoSym( hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *R , hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
int hypre_SMG3RAPPeriodicSym( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );
int hypre_SMG3RAPPeriodicNoSym( hypre_StructMatrix *RAP , hypre_Index cindex , hypre_Index cstride );

/* smg_axpy.c */
int hypre_SMGAxpy( double alpha , hypre_StructVector *x , hypre_StructVector *y , hypre_Index base_index , hypre_Index base_stride );

/* smg_relax.c */
void *hypre_SMGRelaxCreate( MPI_Comm comm );
int hypre_SMGRelaxDestroyTempVec( void *relax_vdata );
int hypre_SMGRelaxDestroyARem( void *relax_vdata );
int hypre_SMGRelaxDestroyASol( void *relax_vdata );
int hypre_SMGRelaxDestroy( void *relax_vdata );
int hypre_SMGRelax( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_SMGRelaxSetup( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_SMGRelaxSetupTempVec( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_SMGRelaxSetupARem( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_SMGRelaxSetupASol( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );
int hypre_SMGRelaxSetTempVec( void *relax_vdata , hypre_StructVector *temp_vec );
int hypre_SMGRelaxSetMemoryUse( void *relax_vdata , int memory_use );
int hypre_SMGRelaxSetTol( void *relax_vdata , double tol );
int hypre_SMGRelaxSetMaxIter( void *relax_vdata , int max_iter );
int hypre_SMGRelaxSetZeroGuess( void *relax_vdata , int zero_guess );
int hypre_SMGRelaxSetNumSpaces( void *relax_vdata , int num_spaces );
int hypre_SMGRelaxSetNumPreSpaces( void *relax_vdata , int num_pre_spaces );
int hypre_SMGRelaxSetNumRegSpaces( void *relax_vdata , int num_reg_spaces );
int hypre_SMGRelaxSetSpace( void *relax_vdata , int i , int space_index , int space_stride );
int hypre_SMGRelaxSetRegSpaceRank( void *relax_vdata , int i , int reg_space_rank );
int hypre_SMGRelaxSetPreSpaceRank( void *relax_vdata , int i , int pre_space_rank );
int hypre_SMGRelaxSetBase( void *relax_vdata , hypre_Index base_index , hypre_Index base_stride );
int hypre_SMGRelaxSetNumPreRelax( void *relax_vdata , int num_pre_relax );
int hypre_SMGRelaxSetNumPostRelax( void *relax_vdata , int num_post_relax );
int hypre_SMGRelaxSetNewMatrixStencil( void *relax_vdata , hypre_StructStencil *diff_stencil );
int hypre_SMGRelaxSetupBaseBoxArray( void *relax_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* smg_residual.c */
void *hypre_SMGResidualCreate( void );
int hypre_SMGResidualSetup( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
int hypre_SMGResidual( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
int hypre_SMGResidualSetBase( void *residual_vdata , hypre_Index base_index , hypre_Index base_stride );
int hypre_SMGResidualDestroy( void *residual_vdata );

/* smg_residual_unrolled.c */
void *hypre_SMGResidualCreate( void );
int hypre_SMGResidualSetup( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
int hypre_SMGResidual( void *residual_vdata , hypre_StructMatrix *A , hypre_StructVector *x , hypre_StructVector *b , hypre_StructVector *r );
int hypre_SMGResidualSetBase( void *residual_vdata , hypre_Index base_index , hypre_Index base_stride );
int hypre_SMGResidualDestroy( void *residual_vdata );

/* smg_setup.c */
int hypre_SMGSetup( void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* smg_setup_interp.c */
hypre_StructMatrix *hypre_SMGCreateInterpOp( hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir );
int hypre_SMGSetupInterpOp( void *relax_data , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x , hypre_StructMatrix *PT , int cdir , hypre_Index cindex , hypre_Index findex , hypre_Index stride );

/* smg_setup_rap.c */
hypre_StructMatrix *hypre_SMGCreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructGrid *coarse_grid );
int hypre_SMGSetupRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *PT , hypre_StructMatrix *Ac , hypre_Index cindex , hypre_Index cstride );

/* smg_setup_restrict.c */
hypre_StructMatrix *hypre_SMGCreateRestrictOp( hypre_StructMatrix *A , hypre_StructGrid *cgrid , int cdir );
int hypre_SMGSetupRestrictOp( hypre_StructMatrix *A , hypre_StructMatrix *R , hypre_StructVector *temp_vec , int cdir , hypre_Index cindex , hypre_Index cstride );

/* smg_solve.c */
int hypre_SMGSolve( void *smg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* sparse_msg.c */
void *hypre_SparseMSGCreate( MPI_Comm comm );
int hypre_SparseMSGDestroy( void *smsg_vdata );
int hypre_SparseMSGSetTol( void *smsg_vdata , double tol );
int hypre_SparseMSGSetMaxIter( void *smsg_vdata , int max_iter );
int hypre_SparseMSGSetJump( void *smsg_vdata , int jump );
int hypre_SparseMSGSetRelChange( void *smsg_vdata , int rel_change );
int hypre_SparseMSGSetZeroGuess( void *smsg_vdata , int zero_guess );
int hypre_SparseMSGSetRelaxType( void *smsg_vdata , int relax_type );
int hypre_SparseMSGSetNumPreRelax( void *smsg_vdata , int num_pre_relax );
int hypre_SparseMSGSetNumPostRelax( void *smsg_vdata , int num_post_relax );
int hypre_SparseMSGSetNumFineRelax( void *smsg_vdata , int num_fine_relax );
int hypre_SparseMSGSetLogging( void *smsg_vdata , int logging );
int hypre_SparseMSGGetNumIterations( void *smsg_vdata , int *num_iterations );
int hypre_SparseMSGPrintLogging( void *smsg_vdata , int myid );
int hypre_SparseMSGGetFinalRelativeResidualNorm( void *smsg_vdata , double *relative_residual_norm );

/* sparse_msg2_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSG2CreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir );
int hypre_SparseMSG2BuildRAPSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );
int hypre_SparseMSG2BuildRAPNoSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );

/* sparse_msg3_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSG3CreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir );
int hypre_SparseMSG3BuildRAPSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );
int hypre_SparseMSG3BuildRAPNoSym( hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructMatrix *R , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *RAP );

/* sparse_msg_filter.c */
int hypre_SparseMSGFilterSetup( hypre_StructMatrix *A , int *num_grids , int lx , int ly , int lz , int jump , hypre_StructVector *visitx , hypre_StructVector *visity , hypre_StructVector *visitz );
int hypre_SparseMSGFilter( hypre_StructVector *visit , hypre_StructVector *e , int lx , int ly , int lz , int jump );
int hypre_SparseMSGFilterSetup( hypre_StructMatrix *A , int *num_grids , int lx , int ly , int lz , int jump , hypre_StructVector *visitx , hypre_StructVector *visity , hypre_StructVector *visitz );
int hypre_SparseMSGFilter( hypre_StructVector *visit , hypre_StructVector *e , int lx , int ly , int lz , int jump );

/* sparse_msg_interp.c */
void *hypre_SparseMSGInterpCreate( void );
int hypre_SparseMSGInterpSetup( void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride , hypre_Index strideP );
int hypre_SparseMSGInterp( void *interp_vdata , hypre_StructMatrix *P , hypre_StructVector *xc , hypre_StructVector *e );
int hypre_SparseMSGInterpDestroy( void *interp_vdata );

/* sparse_msg_restrict.c */
void *hypre_SparseMSGRestrictCreate( void );
int hypre_SparseMSGRestrictSetup( void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride , hypre_Index strideR );
int hypre_SparseMSGRestrict( void *restrict_vdata , hypre_StructMatrix *R , hypre_StructVector *r , hypre_StructVector *rc );
int hypre_SparseMSGRestrictDestroy( void *restrict_vdata );

/* sparse_msg_setup.c */
int hypre_SparseMSGSetup( void *smsg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );

/* sparse_msg_setup_rap.c */
hypre_StructMatrix *hypre_SparseMSGCreateRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , hypre_StructGrid *coarse_grid , int cdir );
int hypre_SparseMSGSetupRAPOp( hypre_StructMatrix *R , hypre_StructMatrix *A , hypre_StructMatrix *P , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_Index stridePR , hypre_StructMatrix *Ac );

/* sparse_msg_solve.c */
int hypre_SparseMSGSolve( void *smsg_vdata , hypre_StructMatrix *A , hypre_StructVector *b , hypre_StructVector *x );


#ifdef __cplusplus
}
#endif

#endif

