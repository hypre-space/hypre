
#include <HYPRE_config.h>

#include "HYPRE_sstruct_ls.h"

#ifndef hypre_SSTRUCT_LS_HEADER
#define hypre_SSTRUCT_LS_HEADER

#include "utilities.h"
#include "krylov.h"
#include "sstruct_matrix_vector.h"

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


#ifdef __cplusplus
}
#endif

#endif

