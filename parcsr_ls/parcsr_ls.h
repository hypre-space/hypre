
#include <HYPRE_config.h>

#include "HYPRE_parcsr_ls.h"

#ifndef hypre_PARCSR_LS_HEADER
#define hypre_PARCSR_LS_HEADER

#include "utilities.h"
#include "krylov.h"
#include "seq_mv.h"
#include "parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif


/* amg_hybrid.c */
void *hypre_AMGHybridCreate( void );
int hypre_AMGHybridDestroy( void *AMGhybrid_vdata );
int hypre_AMGHybridSetTol( void *AMGhybrid_vdata , double tol );
int hypre_AMGHybridSetConvergenceTol( void *AMGhybrid_vdata , double cf_tol );
int hypre_AMGHybridSetDSCGMaxIter( void *AMGhybrid_vdata , int dscg_max_its );
int hypre_AMGHybridSetPCGMaxIter( void *AMGhybrid_vdata , int pcg_max_its );
int hypre_AMGHybridSetTwoNorm( void *AMGhybrid_vdata , int two_norm );
int hypre_AMGHybridSetRelChange( void *AMGhybrid_vdata , int rel_change );
int hypre_AMGHybridSetPrecond( void *pcg_vdata , int (*pcg_precond_solve )(), int (*pcg_precond_setup )(), void *pcg_precond );
int hypre_AMGHybridSetLogging( void *AMGhybrid_vdata , int logging );
int hypre_AMGHybridSetPLogging( void *AMGhybrid_vdata , int plogging );
int hypre_AMGHybridSetStrongThreshold( void *AMGhybrid_vdata , double strong_threshold );
int hypre_AMGHybridSetMaxRowSum( void *AMGhybrid_vdata , double max_row_sum );
int hypre_AMGHybridSetTruncFactor( void *AMGhybrid_vdata , double trunc_factor );
int hypre_AMGHybridSetMaxLevels( void *AMGhybrid_vdata , int max_levels );
int hypre_AMGHybridSetMeasureType( void *AMGhybrid_vdata , int measure_type );
int hypre_AMGHybridSetCoarsenType( void *AMGhybrid_vdata , int coarsen_type );
int hypre_AMGHybridSetCycleType( void *AMGhybrid_vdata , int cycle_type );
int hypre_AMGHybridSetNumGridSweeps( void *AMGhybrid_vdata , int *num_grid_sweeps );
int hypre_AMGHybridSetGridRelaxType( void *AMGhybrid_vdata , int *grid_relax_type );
int hypre_AMGHybridSetGridRelaxPoints( void *AMGhybrid_vdata , int **grid_relax_points );
int hypre_AMGHybridSetRelaxWeight( void *AMGhybrid_vdata , double *relax_weight );
int hypre_AMGHybridSetOmega( void *AMGhybrid_vdata , double *omega );
int hypre_AMGHybridGetNumIterations( void *AMGhybrid_vdata , int *num_its );
int hypre_AMGHybridGetDSCGNumIterations( void *AMGhybrid_vdata , int *dscg_num_its );
int hypre_AMGHybridGetPCGNumIterations( void *AMGhybrid_vdata , int *pcg_num_its );
int hypre_AMGHybridGetFinalRelativeResidualNorm( void *AMGhybrid_vdata , double *final_rel_res_norm );
int hypre_AMGHybridSetup( void *AMGhybrid_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );
int hypre_AMGHybridSolve( void *AMGhybrid_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *b , hypre_ParVector *x );

/* driver.c */
int BuildParFromFile( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParDifConv( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParFromOneFile( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildRhsParFromOneFile( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix A , HYPRE_ParVector *b_ptr );
int BuildParLaplacian9pt( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );
int BuildParLaplacian27pt( int argc , char *argv [], int arg_index , HYPRE_ParCSRMatrix *A_ptr );

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
int HYPRE_BoomerAMGSetOmega( HYPRE_Solver solver , double *omega );
int HYPRE_BoomerAMGSetSmoothOption( HYPRE_Solver solver , int *smooth_option );
int HYPRE_BoomerAMGSetSmoothNumSweep( HYPRE_Solver solver , int smooth_num_sweep );
int HYPRE_BoomerAMGSetLogLevel( HYPRE_Solver solver , int log_level );
int HYPRE_BoomerAMGSetPrintLevel( HYPRE_Solver solver , int print_level );
int HYPRE_BoomerAMGSetPrintFileName( HYPRE_Solver solver , const char *print_file_name );
int HYPRE_BoomerAMGSetDebugFlag( HYPRE_Solver solver , int debug_flag );
int HYPRE_BoomerAMGGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_BoomerAMGGetResidual( HYPRE_Solver solver , HYPRE_ParVector *residual );
int HYPRE_BoomerAMGGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *rel_resid_norm );
int HYPRE_BoomerAMGSetVariant( HYPRE_Solver solver , int variant );
int HYPRE_BoomerAMGSetOverlap( HYPRE_Solver solver , int overlap );
int HYPRE_BoomerAMGSetDomainType( HYPRE_Solver solver , int domain_type );
int HYPRE_BoomerAMGSetSchwarzRlxWeight( HYPRE_Solver solver , double schwarz_rlx_weight );
int HYPRE_BoomerAMGSetSym( HYPRE_Solver solver , int sym );
int HYPRE_BoomerAMGSetLevel( HYPRE_Solver solver , int level );
int HYPRE_BoomerAMGSetThreshold( HYPRE_Solver solver , double threshold );
int HYPRE_BoomerAMGSetFilter( HYPRE_Solver solver , double filter );
int HYPRE_BoomerAMGSetDropTol( HYPRE_Solver solver , double drop_tol );
int HYPRE_BoomerAMGSetMaxNzPerRow( HYPRE_Solver solver , int max_nz_per_row );
int HYPRE_BoomerAMGSetEuclidFile( HYPRE_Solver solver , char *euclidfile );
int HYPRE_BoomerAMGSetNumFunctions( HYPRE_Solver solver , int num_functions );
int HYPRE_BoomerAMGSetDofFunc( HYPRE_Solver solver , int *dof_func );
int HYPRE_BoomerAMGSetGSMG( HYPRE_Solver solver , int gsmg );
int HYPRE_BoomerAMGSetNumSamples( HYPRE_Solver solver , int num_samples );

/* HYPRE_parcsr_bicgstab.c */
int HYPRE_ParCSRBiCGSTABCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRBiCGSTABDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRBiCGSTABSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSTABSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSTABSetTol( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRBiCGSTABSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRBiCGSTABSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRBiCGSTABSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRBiCGSTABSetPrecond( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRBiCGSTABGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRBiCGSTABSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRBiCGSTABGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_cgnr.c */
int HYPRE_ParCSRCGNRCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRCGNRDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRCGNRSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRCGNRSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRCGNRSetTol( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRCGNRSetMinIter( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRCGNRSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRCGNRSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRCGNRSetPrecond( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precondT , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRCGNRGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRCGNRSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRCGNRGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_Euclid.c */
int HYPRE_EuclidCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_EuclidDestroy( HYPRE_Solver solver );
int HYPRE_EuclidSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_EuclidSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector bb , HYPRE_ParVector xx );
int HYPRE_EuclidSetParams( HYPRE_Solver solver , int argc , char *argv []);
int HYPRE_EuclidSetParamsFromFile( HYPRE_Solver solver , char *filename );

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
int HYPRE_ParCSRGMRESSetPrecond( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRGMRESGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRGMRESSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRGMRESGetNumIterations( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

/* HYPRE_parcsr_hybrid.c */
int HYPRE_ParCSRHybridCreate( HYPRE_Solver *solver );
int HYPRE_ParCSRHybridDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRHybridSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRHybridSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRHybridSetTol( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRHybridSetConvergenceTol( HYPRE_Solver solver , double cf_tol );
int HYPRE_ParCSRHybridSetDSCGMaxIter( HYPRE_Solver solver , int dscg_max_its );
int HYPRE_ParCSRHybridSetPCGMaxIter( HYPRE_Solver solver , int pcg_max_its );
int HYPRE_ParCSRHybridSetTwoNorm( HYPRE_Solver solver , int two_norm );
int HYPRE_ParCSRHybridSetRelChange( HYPRE_Solver solver , int rel_change );
int HYPRE_ParCSRHybridSetPrecond( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRHybridSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRHybridSetPLogging( HYPRE_Solver solver , int plogging );
int HYPRE_ParCSRHybridSetStrongThreshold( HYPRE_Solver solver , double strong_threshold );
int HYPRE_ParCSRHybridSetMaxRowSum( HYPRE_Solver solver , double max_row_sum );
int HYPRE_ParCSRHybridSetTruncFactor( HYPRE_Solver solver , double trunc_factor );
int HYPRE_ParCSRHybridSetMaxLevels( HYPRE_Solver solver , int max_levels );
int HYPRE_ParCSRHybridSetMeasureType( HYPRE_Solver solver , int measure_type );
int HYPRE_ParCSRHybridSetCoarsenType( HYPRE_Solver solver , int coarsen_type );
int HYPRE_ParCSRHybridSetCycleType( HYPRE_Solver solver , int cycle_type );
int HYPRE_ParCSRHybridSetNumGridSweeps( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_ParCSRHybridSetGridRelaxType( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_ParCSRHybridSetGridRelaxPoints( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_ParCSRHybridSetRelaxWeight( HYPRE_Solver solver , double *relax_weight );
int HYPRE_ParCSRHybridSetOmega( HYPRE_Solver solver , double *omega );
int HYPRE_ParCSRHybridGetNumIterations( HYPRE_Solver solver , int *num_its );
int HYPRE_ParCSRHybridGetDSCGNumIterations( HYPRE_Solver solver , int *dscg_num_its );
int HYPRE_ParCSRHybridGetPCGNumIterations( HYPRE_Solver solver , int *pcg_num_its );
int HYPRE_ParCSRHybridGetFinalRelativeResidualNorm( HYPRE_Solver solver , double *norm );

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
int HYPRE_ParaSailsCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParaSailsDestroy( HYPRE_Solver solver );
int HYPRE_ParaSailsSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParaSailsSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParaSailsSetParams( HYPRE_Solver solver , double thresh , int nlevels );
int HYPRE_ParaSailsSetFilter( HYPRE_Solver solver , double filter );
int HYPRE_ParaSailsSetSym( HYPRE_Solver solver , int sym );
int HYPRE_ParaSailsSetLoadbal( HYPRE_Solver solver , double loadbal );
int HYPRE_ParaSailsSetReuse( HYPRE_Solver solver , int reuse );
int HYPRE_ParaSailsSetLogging( HYPRE_Solver solver , int logging );
int HYPRE_ParaSailsBuildIJMatrix( HYPRE_Solver solver , HYPRE_IJMatrix *pij_A );

/* HYPRE_parcsr_pcg.c */
int HYPRE_ParCSRPCGCreate( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRPCGDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRPCGSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSetTol( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRPCGSetMaxIter( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRPCGSetStopCrit( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRPCGSetTwoNorm( HYPRE_Solver solver , int two_norm );
int HYPRE_ParCSRPCGSetRelChange( HYPRE_Solver solver , int rel_change );
int HYPRE_ParCSRPCGSetPrecond( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRPCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRPCGSetPrintLevel( HYPRE_Solver solver , int level );
int HYPRE_ParCSRPCGSetLogging( HYPRE_Solver solver , int level );
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

/* HYPRE_parcsr_schwarz.c */
int HYPRE_SchwarzCreate( HYPRE_Solver *solver );
int HYPRE_SchwarzDestroy( HYPRE_Solver solver );
int HYPRE_SchwarzSetup( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_SchwarzSolve( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_SchwarzSetVariant( HYPRE_Solver solver , int variant );
int HYPRE_SchwarzSetOverlap( HYPRE_Solver solver , int overlap );
int HYPRE_SchwarzSetDomainType( HYPRE_Solver solver , int domain_type );
int HYPRE_SchwarzSetDomainStructure( HYPRE_Solver solver , HYPRE_CSRMatrix domain_structure );
int HYPRE_SchwarzSetNumFunctions( HYPRE_Solver solver , int num_functions );
int HYPRE_SchwarzSetRelaxWeight( HYPRE_Solver solver , double relax_weight );
int HYPRE_SchwarzSetDofFunc( HYPRE_Solver solver , int *dof_func );

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
int hypre_BoomerAMGSetOmega( void *data , double *omega );
int hypre_BoomerAMGSetSmoothOption( void *data , int *smooth_option );
int hypre_BoomerAMGSetSmoothNumSweep( void *data , int smooth_num_sweep );
int hypre_BoomerAMGSetLogLevel( void *data , int log_level );
int hypre_BoomerAMGSetPrintLevel( void *data , int print_level );
int hypre_BoomerAMGSetPrintFileName( void *data , const char *print_file_name );
int hypre_BoomerAMGSetNumIterations( void *data , int num_iterations );
int hypre_BoomerAMGSetDebugFlag( void *data , int debug_flag );
int hypre_BoomerAMGSetGSMG(void *data , int par);
int hypre_BoomerAMGSetNumSamples(void *data , int par);
int hypre_BoomerAMGSetNumFunctions( void *data , int num_functions );
int hypre_BoomerAMGSetNumPoints( void *data , int num_points );
int hypre_BoomerAMGSetDofFunc( void *data , int *dof_func );
int hypre_BoomerAMGSetPointDofMap( void *data , int *point_dof_map );
int hypre_BoomerAMGSetDofPoint( void *data , int *dof_point );
int hypre_BoomerAMGGetNumIterations( void *data , int *num_iterations );
int hypre_BoomerAMGGetResidual( void *data , hypre_ParVector **resid );
int hypre_BoomerAMGGetRelResidualNorm( void *data , double *rel_resid_norm );
int hypre_BoomerAMGSetVariant( void *data , int variant );
int hypre_BoomerAMGSetOverlap( void *data , int overlap );
int hypre_BoomerAMGSetDomainType( void *data , int domain_type );
int hypre_BoomerAMGSetSchwarzRlxWeight( void *data , double schwarz_rlx_weight );
int hypre_BoomerAMGSetSym( void *data , int sym );
int hypre_BoomerAMGSetLevel( void *data , int level );
int hypre_BoomerAMGSetThreshold( void *data , double thresh );
int hypre_BoomerAMGSetFilter( void *data , double filter );
int hypre_BoomerAMGSetDropTol( void *data , double drop_tol );
int hypre_BoomerAMGSetMaxNzPerRow( void *data , int max_nz_per_row );
int hypre_BoomerAMGSetEuclidFile( void *data , char *euclidfile );

/* par_amg_setup.c */
int hypre_BoomerAMGSetup( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solve.c */
int hypre_BoomerAMGSolve( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solveT.c */
int hypre_BoomerAMGSolveT( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
int hypre_BoomerAMGCycleT( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );
int hypre_BoomerAMGRelaxT( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp );

/* par_cg_relax_wt.c */
int hypre_BoomerAMGCGRelaxWt( void *amg_vdata , int level , int num_cg_sweeps , double *rlx_wt );
int hypre_Bisection( int n, double *diag, double *offd, double y, double z, double tol, int k, double *ev_ptr);

/* par_coarsen.c */
int hypre_BoomerAMGCoarsen( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int CF_init , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenRuge( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int coarsen_type , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenFalgout( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int debug_flag , int **CF_marker_ptr );

/* par_coarse_parms.c */
int hypre_BoomerAMGCoarseParms( MPI_Comm comm , int local_num_variables , int num_functions , int *dof_func , int *CF_marker , int **coarse_dof_func_ptr , int **coarse_pnts_global_ptr );

/* par_cycle.c */
int hypre_BoomerAMGCycle( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );

/* par_difconv.c */
HYPRE_ParCSRMatrix GenerateDifConv( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );

/* par_gsmg.c */
int hypre_ParCSRMatrixClone(hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **Sp , int copy_data);
int hypre_ParCSRMatrixFillSmooth(int nsamples , double *samples , hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int num_functions , int *dof_func);
double hypre_ParCSRMatrixChooseThresh(hypre_ParCSRMatrix *S);
int hypre_ParCSRMatrixThreshold(hypre_ParCSRMatrix *A , double thresh);
int hypre_BoomerAMGCreateSmoothVecs(void *data , hypre_ParCSRMatrix *A , int num_sweeps , int level , double **SmoothVecs_p);
int hypre_BoomerAMGCreateSmoothDirs(void *data , hypre_ParCSRMatrix *A , double *SmoothVecs , double thresh , int num_functions , int *dof_func , hypre_ParCSRMatrix **S_ptr);
int hypre_BoomerAMGFitVectors(int ip , int n , int num , double *V , int nc , const int *ind , double *val);
int hypre_BoomerAMGBuildInterpLS(hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int num_smooth , double *SmoothVecs , hypre_ParCSRMatrix **P_ptr);
int hypre_BoomerAMGBuildInterpGSMG(hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , hypre_ParCSRMatrix **P_ptr);

/* par_indepset.c */
int hypre_BoomerAMGIndepSetInit( hypre_ParCSRMatrix *S , double *measure_array );
int hypre_BoomerAMGIndepSet( hypre_ParCSRMatrix *S , hypre_CSRMatrix *S_ext , double *measure_array , int *graph_array , int graph_array_size , int *graph_array_offd , int graph_array_offd_size , int *IS_marker , int *IS_marker_offd );

/* par_interp.c */
int hypre_BoomerAMGBuildInterp( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , int *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , hypre_ParCSRMatrix **P_ptr );

/* par_laplace_27pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian27pt( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );
int hypre_map3( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* par_laplace_9pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian9pt( MPI_Comm comm , int nx , int ny , int P , int Q , int p , int q , double *value );
int hypre_map2( int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part );

/* par_laplace.c */
HYPRE_ParCSRMatrix GenerateLaplacian( MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value );
int hypre_map( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part );

/* par_rap.c */
hypre_CSRMatrix *hypre_ExchangeRAPData( hypre_CSRMatrix *RAP_int , hypre_ParCSRCommPkg *comm_pkg_RT );
int hypre_BoomerAMGBuildCoarseOperator( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr );

/* par_rap_communication.c */
int hypre_GetCommPkgRTFromCommPkgA( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A );
int hypre_GenerateSendMapAndCommPkg( MPI_Comm comm , int num_sends , int num_recvs , int *recv_procs , int *send_procs , int *recv_vec_starts , hypre_ParCSRMatrix *A );
int hypre_GenerateRAPCommPkg( hypre_ParCSRMatrix *RAP , hypre_ParCSRMatrix *A );

/* par_relax.c */
int hypre_BoomerAMGRelax( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , double omega , hypre_ParVector *u , hypre_ParVector *Vtemp );
int gselim( double *A , double *x , int n );

/* par_scaled_matnorm.c */
int hypre_ParCSRMatrixScaledNorm( hypre_ParCSRMatrix *A , double *scnorm );

/* par_schwarz.c */
void *hypre_SchwarzCreate( void );
int hypre_SchwarzDestroy( void *data );
int hypre_SchwarzSetup( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
int hypre_SchwarzSolve( void *schwarz_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );
int hypre_SchwarzSetVariant( void *data , int variant );
int hypre_SchwarzSetDomainType( void *data , int domain_type );
int hypre_SchwarzSetOverlap( void *data , int overlap );
int hypre_SchwarzSetNumFunctions( void *data , int num_functions );
int hypre_SchwarzSetRelaxWeight( void *data , double relax_weight );
int hypre_SchwarzSetDomainStructure( void *data , hypre_CSRMatrix *domain_structure );
int hypre_SchwarzSetScale( void *data , double *scale );
int hypre_SchwarzReScale( void *data , int size, double value );
int hypre_SchwarzSetDofFunc( void *data , int *dof_func );

/* par_stats.c */
int hypre_BoomerAMGSetupStats( void *amg_vdata , hypre_ParCSRMatrix *A );
int hypre_BoomerAMGWriteSolverParams( void *data );

/* par_strength.c */
int hypre_BoomerAMGCreateS( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int num_functions , int *dof_func , hypre_ParCSRMatrix **S_ptr );

/* pcg_par.c */
char *hypre_ParKrylovCAlloc( int count , int elt_size );
int hypre_ParKrylovFree( char *ptr );
void *hypre_ParKrylovCreateVector( void *vvector );
void *hypre_ParKrylovCreateVectorArray( int n , void *vvector );
int hypre_ParKrylovDestroyVector( void *vvector );
void *hypre_ParKrylovMatvecCreate( void *A , void *x );
int hypre_ParKrylovMatvec( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovMatvecT( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovMatvecDestroy( void *matvec_data );
double hypre_ParKrylovInnerProd( void *x , void *y );
int hypre_ParKrylovCopyVector( void *x , void *y );
int hypre_ParKrylovClearVector( void *x );
int hypre_ParKrylovScaleVector( double alpha , void *x );
int hypre_ParKrylovAxpy( double alpha , void *x , void *y );
int hypre_ParKrylovCommInfo( void *A , int *my_id , int *num_procs );
int hypre_ParKrylovIdentitySetup( void *vdata , void *A , void *b , void *x );
int hypre_ParKrylovIdentity( void *vdata , void *A , void *b , void *x );

/* schwarz.c */
void hypre_F90_NAME_BLAS( int dpotrf , int DPOTRF );
int hypre_MPSchwarzSolve( hypre_ParCSRMatrix *par_A , hypre_Vector *rhs_vector , hypre_CSRMatrix *domain_structure , hypre_ParVector *par_x , hypre_Vector *aux_vector );
int transpose_matrix_create( int **i_face_element_pointer , int **j_face_element_pointer , int *i_element_face , int *j_element_face , int num_elements , int num_faces );
int matrix_matrix_product( int **i_element_edge_pointer , int **j_element_edge_pointer , int *i_element_face , int *j_element_face , int *i_face_edge , int *j_face_edge , int num_elements , int num_faces , int num_edges );
int hypre_AMGCreateDomainDof( hypre_CSRMatrix *A , int domain_type , int overlap , int num_functions , int *dof_func , hypre_CSRMatrix **domain_structure_pointer );
int hypre_AMGeAgglomerate( int *i_AE_element , int *j_AE_element , int *i_face_face , int *j_face_face , int *w_face_face , int *i_face_element , int *j_face_element , int *i_element_face , int *j_element_face , int *i_face_to_prefer_weight , int *i_face_weight , int num_faces , int num_elements , int *num_AEs_pointer );
int update_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int remove_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int move_entry( int weight , int *weight_max , int *previous , int *next , int *first , int *last , int head , int tail , int i );
int matinv( double *x , double *a , int k );
int hypre_parCorrRes( hypre_ParCSRMatrix *A , hypre_ParVector *x , hypre_Vector *rhs , double **tmp_ptr );
int hypre_AdSchwarzSolve( hypre_ParCSRMatrix *par_A , hypre_ParVector *par_rhs , hypre_CSRMatrix *domain_structure , double *scale , hypre_ParVector *par_x , hypre_ParVector *par_aux );
int hypre_GenerateScale( hypre_CSRMatrix *domain_structure , int num_variables , double relaxation_weight , double **scale_pointer );
int hypre_ParAdSchwarzSolve( hypre_ParCSRMatrix *A , hypre_ParVector *F , hypre_CSRMatrix *domain_structure , double *scale , hypre_ParVector *X , hypre_ParVector *Vtemp );
int hypre_ParAMGCreateDomainDof( hypre_ParCSRMatrix *A , int domain_type , int overlap , int num_functions , int *dof_func , hypre_CSRMatrix **domain_structure_pointer );
int hypre_ParGenerateScale( hypre_ParCSRMatrix *A , hypre_CSRMatrix *domain_structure , double relaxation_weight , double **scale_pointer );


#ifdef __cplusplus
}
#endif

#endif

