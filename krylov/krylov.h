
#ifndef hypre_KRYLOV_HEADER
#define hypre_KRYLOV_HEADER

#include "all_krylov.h"
#include "bicgstab.h"
#include "cgnr.h"
#include "gmres.h"
#include "pcg.h"

#ifdef __cplusplus
extern "C" {
#endif


/* HYPRE_pcg.c */
int HYPRE_PCGSetup( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_PCGSolve( HYPRE_Solver solver , HYPRE_Matrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_PCGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_PCGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_PCGSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_PCGSetTwoNorm( HYPRE_Solver solver , int two_norm );
int HYPRE_PCGSetRelChange( HYPRE_Solver solver , int rel_change );
int HYPRE_PCGSetPrecond( HYPRE_Solver solver , HYPRE_PtrToSolverFcn precond , HYPRE_PtrToSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_PCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_PCGSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_PCGGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_PCGGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* bicgstab.c */
hypre_BiCGSTABFunctions *hypre_BiCGSTABFunctionsCreate( void *(*CreateVector )(void *vvector ), int (*DestroyVector )(void *vvector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), int (*precond )(), int (*precond_setup )());
void *hypre_BiCGSTABCreate( hypre_BiCGSTABFunctions *bicgstab_functions );
int hypre_BiCGSTABDestroy( void *bicgstab_vdata );
int hypre_BiCGSTABSetup( void *bicgstab_vdata , void *A , void *b , void *x );
int hypre_BiCGSTABSolve( void *bicgstab_vdata , void *A , void *b , void *x );
int hypre_BiCGSTABSetTol( void *bicgstab_vdata , double tol );
int hypre_BiCGSTABSetMinIter( void *bicgstab_vdata , int min_iter );
int hypre_BiCGSTABSetMaxIter( void *bicgstab_vdata , int max_iter );
int hypre_BiCGSTABSetStopCrit( void *bicgstab_vdata , double stop_crit );
int hypre_BiCGSTABSetPrecond( void *bicgstab_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_BiCGSTABGetPrecond( void *bicgstab_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_BiCGSTABSetLogging( void *bicgstab_vdata , int logging );
int hypre_BiCGSTABGetNumIterations( void *bicgstab_vdata , int *num_iterations );
int hypre_BiCGSTABGetFinalRelativeResidualNorm( void *bicgstab_vdata , double *relative_residual_norm );

/* cgnr.c */
hypre_CGNRFunctions *hypre_CGNRFunctionsCreate( int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecT )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ), int (*PrecondT )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_CGNRCreate( hypre_CGNRFunctions *cgnr_functions );
int hypre_CGNRDestroy( void *cgnr_vdata );
int hypre_CGNRSetup( void *cgnr_vdata , void *A , void *b , void *x );
int hypre_CGNRSolve( void *cgnr_vdata , void *A , void *b , void *x );
int hypre_CGNRSetTol( void *cgnr_vdata , double tol );
int hypre_CGNRSetMinIter( void *cgnr_vdata , int min_iter );
int hypre_CGNRSetMaxIter( void *cgnr_vdata , int max_iter );
int hypre_CGNRSetStopCrit( void *cgnr_vdata , int stop_crit );
int hypre_CGNRSetPrecond( void *cgnr_vdata , int (*precond )(), int (*precondT )(), int (*precond_setup )(), void *precond_data );
int hypre_CGNRGetPrecond( void *cgnr_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_CGNRSetLogging( void *cgnr_vdata , int logging );
int hypre_CGNRGetNumIterations( void *cgnr_vdata , int *num_iterations );
int hypre_CGNRGetFinalRelativeResidualNorm( void *cgnr_vdata , double *relative_residual_norm );

/* gmres.c */
hypre_GMRESFunctions *hypre_GMRESFunctionsCreate( char *(*CAlloc )(int count ,int elt_size ), int (*Free )(char *ptr ), int (*CommInfo )(void *A ,int *my_id ,int *num_procs ), void *(*CreateVector )(void *vector ), void *(*CreateVectorArray )(int size ,void *vectors ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_GMRESCreate( hypre_GMRESFunctions *gmres_functions );
int hypre_GMRESDestroy( void *gmres_vdata );
int hypre_GMRESSetup( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSolve( void *gmres_vdata , void *A , void *b , void *x );
int hypre_GMRESSetKDim( void *gmres_vdata , int k_dim );
int hypre_GMRESSetTol( void *gmres_vdata , double tol );
int hypre_GMRESSetMinIter( void *gmres_vdata , int min_iter );
int hypre_GMRESSetMaxIter( void *gmres_vdata , int max_iter );
int hypre_GMRESSetStopCrit( void *gmres_vdata , double stop_crit );
int hypre_GMRESSetPrecond( void *gmres_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_GMRESGetPrecond( void *gmres_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_GMRESSetLogging( void *gmres_vdata , int logging );
int hypre_GMRESGetNumIterations( void *gmres_vdata , int *num_iterations );
int hypre_GMRESGetFinalRelativeResidualNorm( void *gmres_vdata , double *relative_residual_norm );

/* pcg.c */
hypre_PCGFunctions *hypre_PCGFunctionsCreate( char *(*CAlloc )(int count ,int elt_size ), int (*Free )(char *ptr ), void *(*CreateVector )(void *vector ), int (*DestroyVector )(void *vector ), void *(*MatvecCreate )(void *A ,void *x ), int (*Matvec )(void *matvec_data ,double alpha ,void *A ,void *x ,double beta ,void *y ), int (*MatvecDestroy )(void *matvec_data ), double (*InnerProd )(void *x ,void *y ), int (*CopyVector )(void *x ,void *y ), int (*ClearVector )(void *x ), int (*ScaleVector )(double alpha ,void *x ), int (*Axpy )(double alpha ,void *x ,void *y ), int (*PrecondSetup )(void *vdata ,void *A ,void *b ,void *x ), int (*Precond )(void *vdata ,void *A ,void *b ,void *x ));
void *hypre_PCGCreate( hypre_PCGFunctions *pcg_functions );
int hypre_PCGDestroy( void *pcg_vdata );
int hypre_PCGSetup( void *pcg_vdata , void *A , void *b , void *x );
int hypre_PCGSolve( void *pcg_vdata , void *A , void *b , void *x );
int hypre_PCGSetTol( void *pcg_vdata , double tol );
int hypre_PCGSetConvergenceFactorTol( void *pcg_vdata , double cf_tol );
int hypre_PCGSetMaxIter( void *pcg_vdata , int max_iter );
int hypre_PCGSetTwoNorm( void *pcg_vdata , int two_norm );
int hypre_PCGSetRelChange( void *pcg_vdata , int rel_change );
int hypre_PCGSetStopCrit( void *pcg_vdata , int stop_crit );
int hypre_PCGGetPrecond( void *pcg_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_PCGSetPrecond( void *pcg_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_PCGSetLogging( void *pcg_vdata , int logging );
int hypre_PCGGetNumIterations( void *pcg_vdata , int *num_iterations );
int hypre_PCGPrintLogging( void *pcg_vdata , int myid );
int hypre_PCGGetFinalRelativeResidualNorm( void *pcg_vdata , double *relative_residual_norm );


#ifdef __cplusplus
}
#endif

#endif

