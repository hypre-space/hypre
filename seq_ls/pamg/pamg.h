
#include <HYPRE_config.h>

#include "HYPRE_ls.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "utilities.h"
#include "krylov.h"
#include "seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Atrunc.c */
int hypre_AMGOpTruncation( hypre_CSRMatrix *A , double trunc_factor , int max_elmts );

/* HYPRE_amg.c */
HYPRE_Solver HYPRE_AMGInitialize( void );
int HYPRE_AMGFinalize( HYPRE_Solver solver );
int HYPRE_AMGSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_AMGSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_AMGSetMaxLevels( HYPRE_Solver solver , int max_levels );
int HYPRE_AMGSetStrongThreshold( HYPRE_Solver solver , double strong_threshold );
int HYPRE_AMGSetATruncFactor( HYPRE_Solver solver , double A_trunc_factor );
int HYPRE_AMGSetAMaxElmts( HYPRE_Solver solver , int A_max_elmts );
int HYPRE_AMGSetPTruncFactor( HYPRE_Solver solver , double P_trunc_factor );
int HYPRE_AMGSetPMaxElmts( HYPRE_Solver solver , int P_max_elmts );
int HYPRE_AMGSetCoarsenType( HYPRE_Solver solver , int coarsen_type );
int HYPRE_AMGSetInterpType( HYPRE_Solver solver , int interp_type );
int HYPRE_AMGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_AMGSetCycleType( HYPRE_Solver solver , int cycle_type );
int HYPRE_AMGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_AMGSetNumRelaxSteps( HYPRE_Solver solver , int num_relax_steps );
int HYPRE_AMGSetNumGridSweeps( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_AMGSetGridRelaxType( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_AMGSetGridRelaxPoints( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_AMGSetRelaxWeight( HYPRE_Solver solver , double *relax_weight );
int HYPRE_AMGSetSchwarzOption( HYPRE_Solver solver , int *schwarz_option );
int HYPRE_AMGSetIOutDat( HYPRE_Solver solver , int ioutdat );
int HYPRE_AMGSetLogFileName( HYPRE_Solver solver , char *log_file_name );
int HYPRE_AMGSetLogging( HYPRE_Solver solver , int ioutdat , char *log_file_name );
int HYPRE_AMGSetNumFunctions( HYPRE_Solver solver , int num_functions );
int HYPRE_AMGSetDofFunc( HYPRE_Solver solver , int *dof_func );

/* HYPRE_csr_gmres.c */
int HYPRE_CSRGMRESCreate( HYPRE_Solver *solver );
int HYPRE_CSRGMRESDestroy( HYPRE_Solver solver );
int HYPRE_CSRGMRESSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRGMRESSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRGMRESSetKDim( HYPRE_Solver solver , int k_dim );
int HYPRE_CSRGMRESSetTol( HYPRE_Solver solver , double tol );
int HYPRE_CSRGMRESSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_CSRGMRESSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_CSRGMRESSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_CSRGMRESSetPrecond( HYPRE_Solver solver , int (*precond )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), int (*precond_setup )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), void *precond_data );
int HYPRE_CSRGMRESGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_CSRGMRESSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_CSRGMRESGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_CSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_csr_pcg.c */
int HYPRE_CSRPCGCreate( HYPRE_Solver *solver );
int HYPRE_CSRPCGDestroy( HYPRE_Solver solver );
int HYPRE_CSRPCGSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRPCGSolve( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x );
int HYPRE_CSRPCGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_CSRPCGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_CSRPCGSetTwoNorm( HYPRE_Solver solver , int two_norm );
int HYPRE_CSRPCGSetRelChange( HYPRE_Solver solver , int rel_change );
int HYPRE_CSRPCGSetPrecond( HYPRE_Solver solver , int (*precond )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), int (*precond_setup )(HYPRE_Solver sol ,HYPRE_CSRMatrix matrix ,HYPRE_Vector b ,HYPRE_Vector x ), void *precond_data );
int HYPRE_CSRPCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_CSRPCGSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_CSRPCGGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_CSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );
int HYPRE_CSRDiagScaleSetup( HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector y , HYPRE_Vector x );
int HYPRE_CSRDiagScale( HYPRE_Solver solver , HYPRE_CSRMatrix HA , HYPRE_Vector Hy , HYPRE_Vector Hx );

/* SPamg-pcg.c */
int main( int argc , char *argv []);

/* amg.c */
void *hypre_AMGInitialize( void );
int hypre_AMGFinalize( void *data );
int hypre_AMGSetMaxLevels( void *data , int max_levels );
int hypre_AMGSetStrongThreshold( void *data , double strong_threshold );
int hypre_AMGSetATruncFactor( void *data , double A_trunc_factor );
int hypre_AMGSetPTruncFactor( void *data , double P_trunc_factor );
int hypre_AMGSetAMaxElmts( void *data , int A_max_elmts );
int hypre_AMGSetPMaxElmts( void *data , int P_max_elmts );
int hypre_AMGSetCoarsenType( void *data , int coarsen_type );
int hypre_AMGSetInterpType( void *data , int interp_type );
int hypre_AMGSetMaxIter( void *data , int max_iter );
int hypre_AMGSetCycleType( void *data , int cycle_type );
int hypre_AMGSetTol( void *data , double tol );
int hypre_AMGSetNumRelaxSteps( void *data , int num_relax_steps );
int hypre_AMGSetNumGridSweeps( void *data , int *num_grid_sweeps );
int hypre_AMGSetGridRelaxType( void *data , int *grid_relax_type );
int hypre_AMGSetGridRelaxPoints( void *data , int **grid_relax_points );
int hypre_AMGSetRelaxWeight( void *data , double *relax_weight );
int hypre_AMGSetSchwarzOption( void *data , int *schwarz_option );
int hypre_AMGSetIOutDat( void *data , int ioutdat );
int hypre_AMGSetLogFileName( void *data , char *log_file_name );
int hypre_AMGSetLogging( void *data , int ioutdat , char *log_file_name );
int hypre_AMGSetNumFunctions( void *data , int num_functions );
int hypre_AMGSetNumPoints( void *data , int num_points );
int hypre_AMGSetDofFunc( void *data , int *dof_func );
int hypre_AMGSetDofPoint( void *data , int *dof_point );
int hypre_AMGSetPointDofMap( void *data , int *point_dof_map );

/* amg_setup.c */
int hypre_AMGSetup( void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u );

/* amg_solve.c */
int hypre_AMGSolve( void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u );

/* amgstats.c */
int hypre_AMGSetupStats( void *amg_vdata );
void hypre_WriteSolverParams( void *data );

/* cg_fun.c */
char *hypre_CGCAlloc( int count , int elt_size );
int hypre_CGFree( char *ptr );
void *hypre_CGCreateVector( void *vvector );
void *hypre_CGCreateVectorArray( int n , void *vvector );
int hypre_CGDestroyVector( void *vvector );
void *hypre_CGMatvecCreate( void *A , void *x );
int hypre_CGMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_CGMatvecT( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_CGMatvecDestroy( void *matvec_data );
double hypre_CGInnerProd( void *x , void *y );
int hypre_CGCopyVector( void *x , void *y );
int hypre_CGClearVector( void *x );
int hypre_CGScaleVector( double alpha , void *x );
int hypre_CGAxpy( double alpha , void *x , void *y );
int hypre_CGCommInfo( void *A , int *my_id , int *num_procs );
int hypre_CGIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_CGIdentity( void *vdata , void *A , void *b , void *x );

/* coarsen.c */
int hypre_AMGCoarsen( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_AMGCoarsenRuge( hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_AMGCoarsenRugeLoL( hypre_CSRMatrix *A , double strength_threshold , int *dof_func , hypre_CSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr );

/* coarsenCR.c */
int hypre_AMGCoarsenCR( hypre_CSRMatrix *A , double strength_threshold , double relax_weight , int relax_type , int num_relax_steps , int **CF_marker_ptr , int *coarse_size_ptr );

/* cycle.c */
int hypre_AMGCycle( void *amg_vdata , hypre_Vector **F_array , hypre_Vector **U_array );

/* difconv.c */
hypre_CSRMatrix *hypre_GenerateDifConv( int nx , int ny , int nz , int P , int Q , int R , double *value );

/* driver.c */
int main( int argc , char *argv []);
int BuildFromFile( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildLaplacian( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildLaplacian9pt( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildLaplacian27pt( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildDifConv( int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr );
int BuildRhsFromFile( int argc , char *argv [], int arg_index , hypre_CSRMatrix *A , hypre_Vector **b_ptr );
int BuildFuncsFromFile( int argc , char *argv [], int arg_index , int **dof_func_ptr );

/* indepset.c */
int hypre_InitAMGIndepSet( hypre_CSRMatrix *S , double *measure_array , double cconst );
int hypre_AMGIndepSet( hypre_CSRMatrix *S , double *measure_array , double cconst , int *graph_array , int graph_array_size , int *IS_marker );

/* interp.c */
int hypre_AMGBuildInterp( hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int *dof_func , int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );

/* interpCR.c */
int hypre_AMGBuildCRInterp( hypre_CSRMatrix *A , int *CF_marker , int n_coarse , int num_relax_steps , int relax_type , double relax_weight , hypre_CSRMatrix **P_ptr );

/* interpRBM.c */
int hypre_AMGBuildRBMInterp( hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , int *dof_func , int num_functions , int **coarse_dof_func_ptr , hypre_CSRMatrix **P_ptr );
int row_mat_rectmat_prod( double *a1 , double *a2 , double *a3 , int i_row , int m , int n );
int matinv( double *x , double *a , int k );

/* laplace.c */
hypre_CSRMatrix *hypre_GenerateLaplacian( int nx , int ny , int nz , int P , int Q , int R , double *value );
int map( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* laplace_27pt.c */
hypre_CSRMatrix *hypre_GenerateLaplacian27pt( int nx , int ny , int nz , int P , int Q , int R , double *value );
int map3( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* laplace_9pt.c */
hypre_CSRMatrix *hypre_GenerateLaplacian9pt( int nx , int ny , int P , int Q , double *value );
int map2( int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part );

/* pcg.c */
void PCG( hypre_Vector *x , hypre_Vector *b , double tol , void *data );
void PCGSetup( hypre_CSRMatrix *A , int (*precond )(), void *precond_data , void *data );

/* random.c */
void hypre_SeedRand( int seed );
double hypre_Rand( void );

/* rap.c */
int hypre_AMGBuildCoarseOperator( hypre_CSRMatrix *RT , hypre_CSRMatrix *A , hypre_CSRMatrix *P , hypre_CSRMatrix **RAP_ptr );

/* relax.c */
int hypre_AMGRelax( hypre_CSRMatrix *A , hypre_Vector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_Vector *u , hypre_Vector *Vtemp );
int gselim( double *A , double *x , int n );

/* scaled_matnorm.c */
int hypre_CSRMatrixScaledNorm( hypre_CSRMatrix *A , double *scnorm );

/* schwarz.c */
int hypre_AMGNodalSchwarzSmoother( hypre_CSRMatrix *A , int *dof_func , int num_functions , int option , int **i_domain_dof_pointer , int **j_domain_dof_pointer , double **domain_matrixinverse_pointer , int *num_domains_pointer );
int hypre_SchwarzSolve( hypre_CSRMatrix *A , hypre_Vector *rhs_vector , int num_domains , int *i_domain_dof , int *j_domain_dof , double *domain_matrixinverse , hypre_Vector *x_vector , hypre_Vector *aux_vector );
int transpose_matrix_create( int **i_face_element_pointer , int **j_face_element_pointer , int *i_element_face , int *j_element_face , int num_elements , int num_faces );
int matrix_matrix_product( int **i_element_edge_pointer , int **j_element_edge_pointer , int *i_element_face , int *j_element_face , int *i_face_edge , int *j_face_edge , int num_elements , int num_faces , int num_edges );
int hypre_AMGCreateDomainDof( hypre_CSRMatrix *A , int **i_domain_dof_pointer , int **j_domain_dof_pointer , double **domain_matrixinverse_pointer , int *num_domains_pointer );
int hypre_AMGeAgglomerate( int *i_AE_element , int *j_AE_element , int *i_face_face , int *j_face_face , int *w_face_face , int *i_face_element , int *j_face_element , int *i_element_face , int *j_element_face , int *i_face_to_prefer_weight , int *i_face_weight , int num_faces , int num_elements , int *num_AEs_pointer );
int update_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int remove_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int move_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );

/* transpose.c */
int hypre_CSRMatrixTranspose( hypre_CSRMatrix *A , hypre_CSRMatrix **AT );

/* trunc.c */
int hypre_AMGTruncation( hypre_CSRMatrix *A , double trunc_factor , int max_elmts );
void swap3( int *v , double *w , int i , int j );
void qsort2( int *v , double *w , int left , int right );


#ifdef __cplusplus
}
#endif

#endif

