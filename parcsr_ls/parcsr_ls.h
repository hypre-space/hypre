
#include "HYPRE_parcsr_ls.h"

#ifndef hypre_PARCSR_LS_HEADER
#define hypre_PARCSR_LS_HEADER

#include "utilities.h"
#include "seq_matrix_vector.h"
#include "parcsr_matrix_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_parcsr_amg.c */
HYPRE_Solver HYPRE_ParAMGInitialize P((void ));
int HYPRE_ParAMGFinalize P((HYPRE_Solver solver ));
int HYPRE_ParAMGSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSolveT P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParAMGSetRestriction P((HYPRE_Solver solver , int restr_par ));
int HYPRE_ParAMGSetMaxLevels P((HYPRE_Solver solver , int max_levels ));
int HYPRE_ParAMGSetStrongThreshold P((HYPRE_Solver solver , double strong_threshold ));
int HYPRE_ParAMGSetInterpType P((HYPRE_Solver solver , int interp_type ));
int HYPRE_ParAMGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParAMGSetCoarsenType P((HYPRE_Solver solver , int coarsen_type ));
int HYPRE_ParAMGSetMeasureType P((HYPRE_Solver solver , int measure_type ));
int HYPRE_ParAMGSetCycleType P((HYPRE_Solver solver , int cycle_type ));
int HYPRE_ParAMGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParAMGSetNumGridSweeps P((HYPRE_Solver solver , int *num_grid_sweeps ));
int HYPRE_ParAMGSetGridRelaxType P((HYPRE_Solver solver , int *grid_relax_type ));
int HYPRE_ParAMGSetGridRelaxPoints P((HYPRE_Solver solver , int **grid_relax_points ));
int HYPRE_ParAMGSetRelaxWeight P((HYPRE_Solver solver , double relax_weight ));
int HYPRE_ParAMGSetIOutDat P((HYPRE_Solver solver , int ioutdat ));
int HYPRE_ParAMGSetLogFileName P((HYPRE_Solver solver , char *log_file_name ));
int HYPRE_ParAMGSetLogging P((HYPRE_Solver solver , int ioutdat , char *log_file_name ));
int HYPRE_ParAMGSetDebugFlag P((HYPRE_Solver solver , int debug_flag ));

/* HYPRE_parcsr_cgnr.c */
int HYPRE_ParCSRCGNRInitialize P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRCGNRFinalize P((HYPRE_Solver solver ));
int HYPRE_ParCSRCGNRSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRCGNRSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRCGNRSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRCGNRSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRCGNRSetPrecond P((HYPRE_Solver solver , int (*precond )(), int (*precondT )(), int (*precond_setup )(), void *precond_data ));
int HYPRE_ParCSRCGNRSetLogging P((HYPRE_Solver solver , int logging ));
int HYPRE_ParCSRCGNRGetNumIterations P((HYPRE_Solver solver , int *num_iterations ));
int HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *norm ));

/* HYPRE_parcsr_gmres.c */
int HYPRE_ParCSRGMRESInitialize P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRGMRESFinalize P((HYPRE_Solver solver ));
int HYPRE_ParCSRGMRESSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRGMRESSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRGMRESSetKDim P((HYPRE_Solver solver , int k_dim ));
int HYPRE_ParCSRGMRESSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRGMRESSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRGMRESSetPrecond P((HYPRE_Solver solver , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int HYPRE_ParCSRGMRESSetLogging P((HYPRE_Solver solver , int logging ));
int HYPRE_ParCSRGMRESGetNumIterations P((HYPRE_Solver solver , int *num_iterations ));
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *norm ));

/* HYPRE_parcsr_pcg.c */
int HYPRE_ParCSRPCGInitialize P((MPI_Comm comm , HYPRE_Solver *solver ));
int HYPRE_ParCSRPCGFinalize P((HYPRE_Solver solver ));
int HYPRE_ParCSRPCGSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPCGSolve P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x ));
int HYPRE_ParCSRPCGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParCSRPCGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParCSRPCGSetTwoNorm P((HYPRE_Solver solver , int two_norm ));
int HYPRE_ParCSRPCGSetRelChange P((HYPRE_Solver solver , int rel_change ));
int HYPRE_ParCSRPCGSetPrecond P((HYPRE_Solver solver , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int HYPRE_ParCSRPCGSetLogging P((HYPRE_Solver solver , int logging ));
int HYPRE_ParCSRPCGGetNumIterations P((HYPRE_Solver solver , int *num_iterations ));
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm P((HYPRE_Solver solver , double *norm ));
int HYPRE_ParCSRDiagScaleSetup P((HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector y , HYPRE_ParVector x ));
int HYPRE_ParCSRDiagScale P((HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx ));

/* cgnr.c */
void *hypre_CGNRInitialize P((void ));
int hypre_CGNRFinalize P((void *cgnr_vdata ));
int hypre_CGNRSetup P((void *cgnr_vdata , void *A , void *b , void *x ));
int hypre_CGNRSolve P((void *cgnr_vdata , void *A , void *b , void *x ));
int hypre_CGNRSetTol P((void *cgnr_vdata , double tol ));
int hypre_CGNRSetMaxIter P((void *cgnr_vdata , int max_iter ));
int hypre_CGNRSetPrecond P((void *cgnr_vdata , int (*precond )(), int (*precondT )(), int (*precond_setup )(), void *precond_data ));
int hypre_CGNRSetLogging P((void *cgnr_vdata , int logging ));
int hypre_CGNRGetNumIterations P((void *cgnr_vdata , int *num_iterations ));
int hypre_CGNRGetFinalRelativeResidualNorm P((void *cgnr_vdata , double *relative_residual_norm ));

/* driver.c */
int main P((int argc , char *argv []));
int BuildParFromFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParLaplacian P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParFromOneFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildRhsParFromOneFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix *A , hypre_ParVector **b_ptr ));
int BuildParLaplacian9pt P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParLaplacian27pt P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));

/* driver_interp.c */
int main P((int argc , char *argv []));

/* driver_rap.c */
int main P((int argc , char *argv []));

/* gmres.c */
void *hypre_GMRESInitialize P((void ));
int hypre_GMRESFinalize P((void *gmres_vdata ));
int hypre_GMRESSetup P((void *gmres_vdata , void *A , void *b , void *x ));
int hypre_GMRESSolve P((void *gmres_vdata , void *A , void *b , void *x ));
int hypre_GMRESSetKDim P((void *gmres_vdata , int k_dim ));
int hypre_GMRESSetTol P((void *gmres_vdata , double tol ));
int hypre_GMRESSetMaxIter P((void *gmres_vdata , int max_iter ));
int hypre_GMRESSetPrecond P((void *gmres_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int hypre_GMRESSetLogging P((void *gmres_vdata , int logging ));
int hypre_GMRESGetNumIterations P((void *gmres_vdata , int *num_iterations ));
int hypre_GMRESGetFinalRelativeResidualNorm P((void *gmres_vdata , double *relative_residual_norm ));

/* par_amg.c */
void *hypre_ParAMGInitialize P((void ));
int hypre_ParAMGFinalize P((void *data ));
int hypre_ParAMGSetRestriction P((void *data , int restr_par ));
int hypre_ParAMGSetMaxLevels P((void *data , int max_levels ));
int hypre_ParAMGSetStrongThreshold P((void *data , double strong_threshold ));
int hypre_ParAMGSetInterpType P((void *data , int interp_type ));
int hypre_ParAMGSetMaxIter P((void *data , int max_iter ));
int hypre_ParAMGSetCoarsenType P((void *data , int coarsen_type ));
int hypre_ParAMGSetMeasureType P((void *data , int measure_type ));
int hypre_ParAMGSetCycleType P((void *data , int cycle_type ));
int hypre_ParAMGSetTol P((void *data , double tol ));
int hypre_ParAMGSetNumGridSweeps P((void *data , int *num_grid_sweeps ));
int hypre_ParAMGSetGridRelaxType P((void *data , int *grid_relax_type ));
int hypre_ParAMGSetGridRelaxPoints P((void *data , int **grid_relax_points ));
int hypre_ParAMGSetRelaxWeight P((void *data , double relax_weight ));
int hypre_ParAMGSetIOutDat P((void *data , int ioutdat ));
int hypre_ParAMGSetLogFileName P((void *data , char *log_file_name ));
int hypre_ParAMGSetLogging P((void *data , int ioutdat , char *log_file_name ));
int hypre_ParAMGSetDebugFlag P((void *data , int debug_flag ));
int hypre_ParAMGSetNumUnknowns P((void *data , int num_unknowns ));
int hypre_ParAMGSetNumPoints P((void *data , int num_points ));
int hypre_ParAMGSetUnknownMap P((void *data , int *unknown_map ));
int hypre_ParAMGSetPointMap P((void *data , int *point_map ));
int hypre_ParAMGSetVatPoint P((void *data , int *v_at_point ));

/* par_amg_setup.c */
int hypre_ParAMGSetup P((void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u ));

/* par_amg_solve.c */
int hypre_ParAMGSolve P((void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u ));

/* par_amg_solveT.c */
int hypre_ParAMGSolveT P((void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u ));
int hypre_ParAMGCycleT P((void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array ));
int hypre_ParAMGRelaxT P((hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp ));

/* par_coarsen.c */
int hypre_ParAMGCoarsen P((hypre_ParCSRMatrix *A , double strength_threshold , hypre_ParCSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr ));
int hypre_ParAMGCoarsenRuge P((hypre_ParCSRMatrix *A , double strength_threshold , int measure_type , int coarsen_type , hypre_ParCSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr ));

/* par_cycle.c */
int hypre_ParAMGCycle P((void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array ));

/* par_indepset.c */
int hypre_InitParAMGIndepSet P((hypre_ParCSRMatrix *S , double *measure_array ));
int hypre_ParAMGIndepSet P((hypre_ParCSRMatrix *S , hypre_CSRMatrix *S_ext , double *measure_array , int *graph_array , int graph_array_size , int *IS_marker , int *IS_marker_offd ));

/* par_interp.c */
int hypre_ParAMGBuildInterp P((hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , hypre_ParCSRMatrix **P_ptr ));

/* par_laplace.c */
hypre_ParCSRMatrix *hypre_GenerateLaplacian P((MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value ));
int map P((int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part ));
void qsort0 P((int *v , int left , int right ));
void swap P((int *v , int i , int j ));

/* par_laplace_27pt.c */
hypre_ParCSRMatrix *hypre_GenerateLaplacian27pt P((MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value ));
int map3 P((int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part ));

/* par_laplace_9pt.c */
hypre_ParCSRMatrix *hypre_GenerateLaplacian9pt P((MPI_Comm comm , int nx , int ny , int P , int Q , int p , int q , double *value ));
int map2 P((int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part ));

/* par_rap.c */
int hypre_ParAMGBuildCoarseOperator P((hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr ));
hypre_CSRMatrix *hypre_ExchangeRAPData P((hypre_CSRMatrix *RAP_int , hypre_CommPkg *comm_pkg_RT ));

/* par_rap_communication.c */
int hypre_GetCommPkgRTFromCommPkgA P((hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A ));
hypre_CommPkg *hypre_GenerateSendMapAndCommPkg P((MPI_Comm comm , int num_sends , int num_recvs , int *recv_procs , int *send_procs , int *recv_vec_starts , hypre_ParCSRMatrix *A ));
int hypre_GenerateRAPCommPkg P((hypre_ParCSRMatrix *RAP , hypre_ParCSRMatrix *A ));

/* par_relax.c */
int hypre_ParAMGRelax P((hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , hypre_ParVector *u , hypre_ParVector *Vtemp ));
int gselim P((double *A , double *x , int n ));

/* par_stats.c */
int hypre_ParAMGSetupStats P((void *amg_vdata , hypre_ParCSRMatrix *A ));
void hypre_WriteParAMGSolverParams P((void *data ));

/* pcg.c */
int hypre_PCGIdentitySetup P((void *vdata , void *A , void *b , void *x ));
int hypre_PCGIdentity P((void *vdata , void *A , void *b , void *x ));
void *hypre_PCGInitialize P((void ));
int hypre_PCGFinalize P((void *pcg_vdata ));
int hypre_PCGSetup P((void *pcg_vdata , void *A , void *b , void *x ));
int hypre_PCGSolve P((void *pcg_vdata , void *A , void *b , void *x ));
int hypre_PCGSetTol P((void *pcg_vdata , double tol ));
int hypre_PCGSetMaxIter P((void *pcg_vdata , int max_iter ));
int hypre_PCGSetTwoNorm P((void *pcg_vdata , int two_norm ));
int hypre_PCGSetRelChange P((void *pcg_vdata , int rel_change ));
int hypre_PCGSetPrecond P((void *pcg_vdata , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int hypre_PCGSetLogging P((void *pcg_vdata , int logging ));
int hypre_PCGGetNumIterations P((void *pcg_vdata , int *num_iterations ));
int hypre_PCGPrintLogging P((void *pcg_vdata , int myid ));
int hypre_PCGGetFinalRelativeResidualNorm P((void *pcg_vdata , double *relative_residual_norm ));

/* pcg_par.c */
char *hypre_PCGCAlloc P((int count , int elt_size ));
int hypre_PCGFree P((char *ptr ));
void *hypre_PCGNewVector P((void *vvector ));
void *hypre_PCGNewVectorArray P((int n , void *vvector ));
int hypre_PCGFreeVector P((void *vvector ));
void *hypre_PCGMatvecInitialize P((void *A , void *x ));
int hypre_PCGMatvec P((void *matvec_data , double alpha , void *A , void *x , double beta , void *y ));
int hypre_PCGMatvecT P((void *matvec_data , double alpha , void *A , void *x , double beta , void *y ));
int hypre_PCGMatvecFinalize P((void *matvec_data ));
double hypre_PCGInnerProd P((void *x , void *y ));
int hypre_PCGCopyVector P((void *x , void *y ));
int hypre_PCGClearVector P((void *x ));
int hypre_PCGScaleVector P((double alpha , void *x ));
int hypre_PCGAxpy P((double alpha , void *x , void *y ));

/* transpose.c */
int hypre_CSRMatrixTranspose P((hypre_CSRMatrix *A , hypre_CSRMatrix **AT ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

