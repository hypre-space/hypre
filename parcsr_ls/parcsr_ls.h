
#include <HYPRE_config.h>

#include "HYPRE_parcsr_ls.h"

#ifndef hypre_PARCSR_LS_HEADER
#define hypre_PARCSR_LS_HEADER

#include "utilities.h"
#include "seq_matrix_vector.h"
#include "parcsr_matrix_vector.h"

#ifdef __cplusplus
extern "C" {
#endif


/* F90_HYPRE_parcsr_ParaSails.c */
void hypre_F90_IFACE( int hypre_parcsrparasailscreate );
void hypre_F90_IFACE( int hypre_parcsrparasailsdestroy );
void hypre_F90_IFACE( int hypre_parcsrparasailssetup );
void hypre_F90_IFACE( int hypre_parcsrparasailssolve );
void hypre_F90_IFACE( int hypre_parcsrparasailssetparams );
void hypre_F90_IFACE( int hypre_parcsrparasailssetfilter );
void hypre_F90_IFACE( int hypre_parcsrparasailssetsym );

/* F90_HYPRE_parcsr_amg.c */
void hypre_F90_IFACE( int hypre_boomeramgcreate );
void hypre_F90_IFACE( int hypre_boomeramgdestroy );
void hypre_F90_IFACE( int hypre_boomeramgsetup );
void hypre_F90_IFACE( int hypre_boomeramgsolve );
void hypre_F90_IFACE( int hypre_boomeramgsolvet );
void hypre_F90_IFACE( int hypre_boomeramgsetrestriction );
void hypre_F90_IFACE( int hypre_boomeramgsetmaxlevels );
void hypre_F90_IFACE( int hypre_boomeramgsetstrongthrshld );
void hypre_F90_IFACE( int hypre_boomeramgsetmaxrowsum );
void hypre_F90_IFACE( int hypre_boomeramgsettruncfactor );
void hypre_F90_IFACE( int hypre_boomeramgsetinterptype );
void hypre_F90_IFACE( int hypre_boomeramgsetminiter );
void hypre_F90_IFACE( int hypre_boomeramgsetmaxiter );
void hypre_F90_IFACE( int hypre_boomeramgsetcoarsentype );
void hypre_F90_IFACE( int hypre_boomeramgsetmeasuretype );
void hypre_F90_IFACE( int hypre_boomeramgsetsetuptype );
void hypre_F90_IFACE( int hypre_boomeramgsetcycletype );
void hypre_F90_IFACE( int hypre_boomeramgsettol );
void hypre_F90_IFACE( int hypre_boomeramgsetnumgridsweeps );
void hypre_F90_IFACE( int hypre_boomeramginitgridrelaxatn );
void hypre_F90_IFACE( int hypre_boomeramgsetgridrelaxtype );
void hypre_F90_IFACE( int hypre_boomeramgsetgridrelaxpnts );
void hypre_F90_IFACE( int hypre_boomeramgsetrelaxweight );
void hypre_F90_IFACE( int hypre_boomeramgsetioutdat );
void hypre_F90_IFACE( int hypre_boomeramgsetlogfilename );
void hypre_F90_IFACE( int hypre_boomeramgsetlogging );
void hypre_F90_IFACE( int hypre_boomeramgsetdebugflag );
void hypre_F90_IFACE( int hypre_boomeramggetnumiterations );
void hypre_F90_IFACE( int hypre_boomeramggetfinalreltvres );

/* F90_HYPRE_parcsr_cgnr.c */
void hypre_F90_IFACE( int hypre_parcsrcgnrcreate );
void hypre_F90_IFACE( int hypre_parcsrcgnrdestroy );
void hypre_F90_IFACE( int hypre_parcsrcgnrsetup );
void hypre_F90_IFACE( int hypre_parcsrcgnrsolve );
void hypre_F90_IFACE( int hypre_parcsrcgnrsettol );
void hypre_F90_IFACE( int hypre_parcsrcgnrsetmaxiter );
void hypre_F90_IFACE( int hypre_parcsrcgnrsetprecond );
void hypre_F90_IFACE( int hypre_parcsrcgnrgetprecond );
void hypre_F90_IFACE( int hypre_parcsrcgnrsetlogging );
void hypre_F90_IFACE( int hypre_parcsrcgnrgetnumiteration );
void hypre_F90_IFACE( int hypre_parcsrcgnrgetfinalrelativ );

/* F90_HYPRE_parcsr_gmres.c */
void hypre_F90_IFACE( int hypre_parcsrgmrescreate );
void hypre_F90_IFACE( int hypre_parcsrgmresdestroy );
void hypre_F90_IFACE( int hypre_parcsrgmressetup );
void hypre_F90_IFACE( int hypre_parcsrgmressolve );
void hypre_F90_IFACE( int hypre_parcsrgmressetkdim );
void hypre_F90_IFACE( int hypre_parcsrgmressettol );
void hypre_F90_IFACE( int hypre_parcsrgmressetminiter );
void hypre_F90_IFACE( int hypre_parcsrgmressetmaxiter );
void hypre_F90_IFACE( int hypre_parcsrgmressetprecond );
void hypre_F90_IFACE( int hypre_parcsrgmresgetprecond );
void hypre_F90_IFACE( int hypre_parcsrgmressetlogging );
void hypre_F90_IFACE( int hypre_parcsrgmresgetnumiteratio );
void hypre_F90_IFACE( int hypre_parcsrgmresgetfinalrelati );

/* F90_HYPRE_parcsr_pcg.c */
void hypre_F90_IFACE( int hypre_parcsrpcgcreate );
void hypre_F90_IFACE( int hypre_parcsrpcgdestroy );
void hypre_F90_IFACE( int hypre_parcsrpcgsetup );
void hypre_F90_IFACE( int hypre_parcsrpcgsolve );
void hypre_F90_IFACE( int hypre_parcsrpcgsettol );
void hypre_F90_IFACE( int hypre_parcsrpcgsetmaxiter );
void hypre_F90_IFACE( int hypre_parcsrpcgsettwonorm );
void hypre_F90_IFACE( int hypre_parcsrpcgsetrelchange );
void hypre_F90_IFACE( int hypre_parcsrpcgsetprecond );
void hypre_F90_IFACE( int hypre_parcsrpcggetprecond );
void hypre_F90_IFACE( int hypre_parcsrpcgsetlogging );
void hypre_F90_IFACE( int hypre_parcsrpcggetnumiterations );
void hypre_F90_IFACE( int hypre_parcsrpcggetfinalrelative );
void hypre_F90_IFACE( int hypre_parcsrdiagscalesetup );
void hypre_F90_IFACE( int hypre_parcsrdiagscale );

/* F90_HYPRE_parcsr_pilut.c */
void hypre_F90_IFACE( int hypre_parcsrpilutcreate );
void hypre_F90_IFACE( int hypre_parcsrpilutdestroy );
void hypre_F90_IFACE( int hypre_parcsrpilutsetup );
void hypre_F90_IFACE( int hypre_parcsrpilutsolve );
void hypre_F90_IFACE( int hypre_parcsrpilutsetmaxiter );
void hypre_F90_IFACE( int hypre_parcsrpilutsetdroptoleran );
void hypre_F90_IFACE( int hypre_parcsrpilutsetfacrowsize );

/* F90_par_laplace.c */
void hypre_F90_IFACE( int generatelaplacian );

/* HYPRE_parcsr_ParaSails.c */
int HYPRE_ParCSRParaSailsCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRParaSailsDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRParaSailsSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRParaSailsSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRParaSailsSetParams( HYPRE_Solver solver , double thresh , int nlevels );
int HYPRE_ParCSRParaSailsSetFilter( HYPRE_Solver solver , double filter );
int HYPRE_ParCSRParaSailsSetSym( HYPRE_Solver solver , int sym );
int HYPRE_ParCSRParaSailsSetLoadbal( HYPRE_Solver solver , double loadbal );
int HYPRE_ParCSRParaSailsSetReuse( HYPRE_Solver solver , int reuse );
int HYPRE_ParCSRParaSailsSetLogging( HYPRE_Solver solver , int logging );

/* HYPRE_parcsr_amg.c */
int HYPRE_BoomerAMGCreate( HYPRE_Solver *solver );
int HYPRE_BoomerAMGDestroy( HYPRE_Solver solver );
int HYPRE_BoomerAMGSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSolveT( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSetRestriction( HYPRE_Solver solver , int restr_par );
int HYPRE_BoomerAMGSetMaxLevels( HYPRE_Solver solver , int max_levels );
int HYPRE_BoomerAMGSetStrongThreshold( HYPRE_Solver solver , double strong_threshold );
int HYPRE_BoomerAMGSetMaxRowSum( HYPRE_Solver solver , double max_row_sum );
int HYPRE_BoomerAMGSetTruncFactor( HYPRE_Solver solver , double trunc_factor );
int HYPRE_BoomerAMGSetInterpType( HYPRE_Solver solver , int interp_type );
int HYPRE_BoomerAMGSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_BoomerAMGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_BoomerAMGSetCoarsenType( HYPRE_Solver solver , int coarsen_type );
int HYPRE_BoomerAMGSetMeasureType( HYPRE_Solver solver , int measure_type );
int HYPRE_BoomerAMGSetSetupType( HYPRE_Solver solver , int setup_type );
int HYPRE_BoomerAMGSetCycleType( HYPRE_Solver solver , int cycle_type );
int HYPRE_BoomerAMGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_BoomerAMGSetNumGridSweeps( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_BoomerAMGInitGridRelaxation( int **num_grid_sweeps_ptr , int **grid_relax_type_ptr , int ***grid_relax_points_ptr , int coarsen_type , double **relax_weights_ptr , int max_levels );
int HYPRE_BoomerAMGSetGridRelaxType( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_BoomerAMGSetGridRelaxPoints( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_BoomerAMGSetRelaxWeight( HYPRE_Solver solver , double *relax_weight );
int HYPRE_BoomerAMGSetIOutDat( HYPRE_Solver solver , int ioutdat );
int HYPRE_BoomerAMGSetLogFileName( HYPRE_Solver solver , char *log_file_name );
int HYPRE_BoomerAMGSetLogging( HYPRE_Solver solver , int ioutdat , char *log_file_name );
int HYPRE_BoomerAMGSetDebugFlag( HYPRE_Solver solver , int debug_flag );
int HYPRE_BoomerAMGGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_BoomerAMGGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *rel_resid_norm );

/* HYPRE_parcsr_cgnr.c */
int HYPRE_ParCSRCGNRCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRCGNRDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRCGNRSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRCGNRSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRCGNRSetTol( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRCGNRSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRCGNRSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRCGNRSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRCGNRSetPrecond( HYPRE_Solver solver , int (*precond )(HYPRE_Solver sol ,HYPRE_ParCSRMatrix matrix ,HYPRE_ParVector b ,HYPRE_ParVector x ), int (*precondT )(HYPRE_Solver sol ,HYPRE_ParCSRMatrix matrix ,HYPRE_ParVector b ,HYPRE_ParVector x ), int (*precond_setup )(HYPRE_Solver sol ,HYPRE_ParCSRMatrix matrix ,HYPRE_ParVector b ,HYPRE_ParVector x ), void *precond_data );
int HYPRE_ParCSRCGNRGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRCGNRSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRCGNRGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_gmres.c */
int HYPRE_ParCSRGMRESCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRGMRESDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRGMRESSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRGMRESSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRGMRESSetKDim( HYPRE_Solver solver , int k_dim );
int HYPRE_ParCSRGMRESSetTol( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRGMRESSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRGMRESSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRGMRESSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRGMRESSetPrecond( HYPRE_Solver solver , int (*precond )(HYPRE_Solver sol ,HYPRE_ParCSRMatrix matrix ,HYPRE_ParVector b ,HYPRE_ParVector x ), int (*precond_setup )(HYPRE_Solver sol ,HYPRE_ParCSRMatrix matrix ,HYPRE_ParVector b ,HYPRE_ParVector x ), void *precond_data );
int HYPRE_ParCSRGMRESGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRGMRESSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRGMRESGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_pcg.c */
int HYPRE_ParCSRPCGCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRPCGDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRPCGSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRPCGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRPCGSetTwoNorm( HYPRE_Solver solver , int two_norm );
int HYPRE_ParCSRPCGSetRelChange( HYPRE_Solver solver , int rel_change );
int HYPRE_ParCSRPCGSetPrecond( HYPRE_Solver solver , int (*precond )(HYPRE_Solver sol ,HYPRE_ParCSRMatrix matrix ,HYPRE_ParVector b ,HYPRE_ParVector x ), int (*precond_setup )(HYPRE_Solver sol ,HYPRE_ParCSRMatrix matrix ,HYPRE_ParVector b ,HYPRE_ParVector x ), void *precond_data );
int HYPRE_ParCSRPCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRPCGSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRPCGGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );
int HYPRE_ParCSRDiagScaleSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector y , HYPRE_ParVector x );
int HYPRE_ParCSRDiagScale( HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx );

/* HYPRE_parcsr_pilut.c */
int HYPRE_ParCSRPilutCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRPilutDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRPilutSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPilutSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPilutSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRPilutSetDropTolerance( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRPilutSetFactorRowSize( HYPRE_Solver solver , int size );

/* cgnr.c */
void *hypre_CGNRCreate( void );
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

/* driver.c */
int main( int argc , char *argv []);
int BuildParFromFile( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParDifConv( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParFromOneFile( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildRhsParFromOneFile( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix A , HYPRE_ParVector *b_ptr );
int BuildParLaplacian9pt( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian27pt( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );

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
int hypre_GMRESGetPrecond( void *gmres_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_GMRESSetLogging( void *gmres_vdata , int logging );
int hypre_GMRESGetNumIterations( void *gmres_vdata , int *num_iterations );
int hypre_GMRESGetFinalRelativeResidualNorm( void *gmres_vdata , double *relative_residual_norm );

/* par_amg.c */
void *hypre_BoomerAMGCreate( void );
int hypre_BoomerAMGDestroy( void *data );
int hypre_BoomerAMGSetRestriction( void *data , int restr_par );
int hypre_BoomerAMGSetMaxLevels( void *data , int max_levels );
int hypre_BoomerAMGSetStrongThreshold( void *data , double strong_threshold );
int hypre_BoomerAMGSetMaxRowSum( void *data , double max_row_sum );
int hypre_BoomerAMGSetTruncFactor( void *data , double trunc_factor );
int hypre_BoomerAMGSetInterpType( void *data , int interp_type );
int hypre_BoomerAMGSetMinIter( void *data , int min_iter );
int hypre_BoomerAMGSetMaxIter( void *data , int max_iter );
int hypre_BoomerAMGSetCoarsenType( void *data , int coarsen_type );
int hypre_BoomerAMGSetMeasureType( void *data , int measure_type );
int hypre_BoomerAMGSetSetupType( void *data , int setup_type );
int hypre_BoomerAMGSetCycleType( void *data , int cycle_type );
int hypre_BoomerAMGSetTol( void *data , double tol );
int hypre_BoomerAMGSetNumGridSweeps( void *data , int *num_grid_sweeps );
int hypre_BoomerAMGSetGridRelaxType( void *data , int *grid_relax_type );
int hypre_BoomerAMGSetGridRelaxPoints( void *data , int **grid_relax_points );
int hypre_BoomerAMGSetRelaxWeight( void *data , double *relax_weight );
int hypre_BoomerAMGSetIOutDat( void *data , int ioutdat );
int hypre_BoomerAMGSetLogFileName( void *data , char *log_file_name );
int hypre_BoomerAMGSetNumIterations( void *data , int num_iterations );
int hypre_BoomerAMGSetLogging( void *data , int ioutdat , char *log_file_name );
int hypre_BoomerAMGSetDebugFlag( void *data , int debug_flag );
int hypre_BoomerAMGSetNumUnknowns( void *data , int num_unknowns );
int hypre_BoomerAMGSetNumPoints( void *data , int num_points );
int hypre_BoomerAMGSetUnknownMap( void *data , int *unknown_map );
int hypre_BoomerAMGSetPointMap( void *data , int *point_map );
int hypre_BoomerAMGSetVatPoint( void *data , int *v_at_point );
int hypre_BoomerAMGGetNumIterations( void *data , int *num_iterations );
int hypre_BoomerAMGGetRelResidualNorm( void *data , double *rel_resid_norm );

/* par_amg_setup.c */
int hypre_BoomerAMGSetup( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solve.c */
int hypre_BoomerAMGSolve( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solveT.c */
int hypre_BoomerAMGSolveT( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
int hypre_BoomerAMGCycleT( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );
int hypre_BoomerAMGRelaxT( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp );

/* par_coarsen.c */
int hypre_BoomerAMGCoarsen( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int debug_flag , hypre_ParCSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_BoomerAMGCoarsenRuge( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int measure_type , int coarsen_type , int debug_flag , hypre_ParCSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr );
int hypre_BoomerAMGCoarsenFalgout( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int debug_flag , hypre_ParCSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr );

/* par_cycle.c */
int hypre_BoomerAMGCycle( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );

/* par_difconv.c */
HYPRE_ParCSRMatrix GenerateDifConv( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );

/* par_indepset.c */
int hypre_BoomerAMGIndepSetInit( hypre_ParCSRMatrix *S , double *measure_array );
int hypre_BoomerAMGIndepSet( hypre_ParCSRMatrix *S , hypre_CSRMatrix *S_ext , double *measure_array , int *graph_array , int graph_array_size , int *graph_array_offd , int graph_array_offd_size , int *IS_marker , int *IS_marker_offd );

/* par_interp.c */
int hypre_BoomerAMGBuildInterp( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int debug_flag , double trunc_factor , hypre_ParCSRMatrix **P_ptr );

/* par_laplace.c */
HYPRE_ParCSRMatrix GenerateLaplacian( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );
int map( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* par_laplace_27pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian27pt( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );
int map3( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* par_laplace_9pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian9pt( MPI_Comm comm , int nx , int ny , int P , int Q , int p , int q , double *value );
int map2( int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part );

/* par_rap.c */
int hypre_BoomerAMGBuildCoarseOperator( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr );
hypre_CSRMatrix *hypre_ExchangeRAPData( hypre_CSRMatrix *RAP_int , hypre_ParCSRCommPkg *comm_pkg_RT );

/* par_rap_communication.c */
int hypre_GetCommPkgRTFromCommPkgA( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A );
int hypre_GenerateSendMapAndCommPkg( MPI_Comm comm , int num_sends , int num_recvs , int *recv_procs , int *send_procs , int *recv_vec_starts , hypre_ParCSRMatrix *A );
int hypre_GenerateRAPCommPkg( hypre_ParCSRMatrix *RAP , hypre_ParCSRMatrix *A );

/* par_relax.c */
int hypre_BoomerAMGRelax( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp );
int gselim( double *A , double *x , int n );

/* par_scaled_matnorm.c */
int hypre_ParCSRMatrixScaledNorm( hypre_ParCSRMatrix *A , double *scnorm );

/* par_stats.c */
int hypre_BoomerAMGSetupStats( void *amg_vdata , hypre_ParCSRMatrix *A );
int hypre_BoomerAMGWriteSolverParams( void *data );

/* par_strength.c */
int hypre_BoomerAMGCreateS( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , hypre_ParCSRMatrix **S_ptr );

/* pcg.c */
int hypre_KrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_KrylovIdentity( void *vdata , void *A , void *b , void *x );
void *hypre_KrylovCreate( void );
int hypre_KrylovDestroy( void *pcg_vdata );
int hypre_KrylovSetup( void *pcg_vdata , void *A , void *b , void *x );
int hypre_KrylovSolve( void *pcg_vdata , void *A , void *b , void *x );
int hypre_KrylovSetTol( void *pcg_vdata , double tol );
int hypre_KrylovSetMaxIter( void *pcg_vdata , int max_iter );
int hypre_KrylovSetTwoNorm( void *pcg_vdata , int two_norm );
int hypre_KrylovSetRelChange( void *pcg_vdata , int rel_change );
int hypre_KrylovSetPrecond( void *pcg_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data );
int hypre_KrylovGetPrecond( void *pcg_vdata , HYPRE_Solver *precond_data_ptr );
int hypre_KrylovSetLogging( void *pcg_vdata , int logging );
int hypre_KrylovGetNumIterations( void *pcg_vdata , int *num_iterations );
int hypre_KrylovPrintLogging( void *pcg_vdata , int myid );
int hypre_KrylovGetFinalRelativeResidualNorm( void *pcg_vdata , double *relative_residual_norm );

/* pcg_par.c */
char *hypre_KrylovCAlloc( int count , int elt_size );
int hypre_KrylovFree( char *ptr );
void *hypre_KrylovCreateVector( void *vvector );
void *hypre_KrylovCreateVectorArray( int n , void *vvector );
int hypre_KrylovDestroyVector( void *vvector );
void *hypre_KrylovMatvecCreate( void *A , void *x );
int hypre_KrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_KrylovMatvecT( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_KrylovMatvecDestroy( void *matvec_data );
double hypre_KrylovInnerProd( void *x , void *y );
int hypre_KrylovCopyVector( void *x , void *y );
int hypre_KrylovClearVector( void *x );
int hypre_KrylovScaleVector( double alpha , void *x );
int hypre_KrylovAxpy( double alpha , void *x , void *y );
int hypre_KrylovCommInfo( void *A , int *my_id , int *num_procs );

/* transpose.c */
int hypre_CSRMatrixTranspose( hypre_CSRMatrix *A , hypre_CSRMatrix **AT );


#ifdef __cplusplus
}
#endif

#endif

