
#include <HYPRE_config.h>

#include "HYPRE_sstruct_ls.h"

#ifndef hypre_SSTRUCT_LS_HEADER
#define hypre_SSTRUCT_LS_HEADER

#include "utilities.h"
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
int HYPRE_SStructGMRESSetPrecond( HYPRE_SStructSolver solver , HYPRE_PtrToStructSolverFcn precond , HYPRE_PtrToStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructGMRESSetLogging( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructGMRESGetNumIterations( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver solver , double *norm );

/* gmres.c */
void *hypre_GMRESCreate( void );
int hypre_GMRESDestroy( void *gmres_vdata );
int hypre_GMRESSetup( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSolve( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSetKDim( void *gmres_vdata , int k_dim );
int hypre_GMRESSetTol( void *gmres_vdata , double tol );
int hypre_GMRESSetMinIter( void *gmres_vdata , int min_iter );
int hypre_GMRESSetMaxIter( void *gmres_vdata , int max_iter );
int hypre_GMRESSetStopCrit( void *gmres_vdata , double stop_crit );
int hypre_GMRESSetPrecond( void *gmres_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_GMRESSetLogging( void *gmres_vdata , int logging );
int hypre_GMRESGetNumIterations( void *gmres_vdata , int *num_iterations );
int hypre_GMRESGetFinalRelativeResidualNorm( void *gmres_vdata , double *relative_residual_norm );

/* krylov.c */
int hypre_KrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_KrylovIdentity( void *vdata , void *A , void *b , void *x );

/* krylov_sstruct.c */
char *hypre_KrylovCAlloc( int count , int elt_size );
int hypre_KrylovFree( char *ptr );
void *hypre_KrylovCreateVector( void *vvector );
void *hypre_KrylovCreateVectorArray( int n , void *vvector );
int hypre_KrylovDestroyVector( void *vvector );
void *hypre_KrylovMatvecCreate( void *A , void *x );
int hypre_KrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_KrylovMatvecDestroy( void *matvec_data );
double hypre_KrylovInnerProd( void *x , void *y );
int hypre_KrylovCopyVector( void *x , void *y );
int hypre_KrylovClearVector( void *x );
int hypre_KrylovScaleVector( double alpha , void *x );
int hypre_KrylovAxpy( double alpha , void *x , void *y );
int hypre_KrylovCommInfo( void *A , int *my_id , int *num_procs );


#ifdef __cplusplus
}
#endif

#endif

