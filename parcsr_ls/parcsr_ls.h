
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
int HYPRE_ParAMGSetMaxLevels P((HYPRE_Solver solver , int max_levels ));
int HYPRE_ParAMGSetStrongThreshold P((HYPRE_Solver solver , double strong_threshold ));
int HYPRE_ParAMGSetInterpType P((HYPRE_Solver solver , int interp_type ));
int HYPRE_ParAMGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_ParAMGSetCycleType P((HYPRE_Solver solver , int cycle_type ));
int HYPRE_ParAMGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_ParAMGSetNumGridSweeps P((HYPRE_Solver solver , int *num_grid_sweeps ));
int HYPRE_ParAMGSetGridRelaxType P((HYPRE_Solver solver , int *grid_relax_type ));
int HYPRE_ParAMGSetGridRelaxPoints P((HYPRE_Solver solver , int **grid_relax_points ));
int HYPRE_ParAMGSetRelaxWeight P((HYPRE_Solver solver , double relax_weight ));
int HYPRE_ParAMGSetIOutDat P((HYPRE_Solver solver , int ioutdat ));
int HYPRE_ParAMGSetLogFileName P((HYPRE_Solver solver , char *log_file_name ));
int HYPRE_ParAMGSetLogging P((HYPRE_Solver solver , int ioutdat , char *log_file_name ));

/* driver.c */
int main P((int argc , char *argv []));
int BuildParFromFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParLaplacian P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParFromOneFile P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));
int BuildParLaplacian9pt P((int argc , char *argv [], int arg_index , hypre_ParCSRMatrix **A_ptr ));

/* driver_interp.c */
int main P((int argc , char *argv []));

/* driver_rap.c */
int main P((int argc , char *argv []));

/* par_amg.c */
void *hypre_ParAMGInitialize P((void ));
int hypre_ParAMGFinalize P((void *data ));
int hypre_ParAMGSetMaxLevels P((void *data , int max_levels ));
int hypre_ParAMGSetStrongThreshold P((void *data , double strong_threshold ));
int hypre_ParAMGSetInterpType P((void *data , int interp_type ));
int hypre_ParAMGSetMaxIter P((void *data , int max_iter ));
int hypre_ParAMGSetCycleType P((void *data , int cycle_type ));
int hypre_ParAMGSetTol P((void *data , double tol ));
int hypre_ParAMGSetNumGridSweeps P((void *data , int *num_grid_sweeps ));
int hypre_ParAMGSetGridRelaxType P((void *data , int *grid_relax_type ));
int hypre_ParAMGSetGridRelaxPoints P((void *data , int **grid_relax_points ));
int hypre_ParAMGSetRelaxWeight P((void *data , double relax_weight ));
int hypre_ParAMGSetIOutDat P((void *data , int ioutdat ));
int hypre_ParAMGSetLogFileName P((void *data , char *log_file_name ));
int hypre_ParAMGSetLogging P((void *data , int ioutdat , char *log_file_name ));
int hypre_ParAMGSetNumUnknowns P((void *data , int num_unknowns ));
int hypre_ParAMGSetNumPoints P((void *data , int num_points ));
int hypre_ParAMGSetUnknownMap P((void *data , int *unknown_map ));
int hypre_ParAMGSetPointMap P((void *data , int *point_map ));
int hypre_ParAMGSetVatPoint P((void *data , int *v_at_point ));

/* par_amg_setup.c */
int hypre_ParAMGSetup P((void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u ));

/* par_amg_solve.c */
int hypre_ParAMGSolve P((void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u ));

/* par_coarsen.c */
int hypre_ParAMGCoarsen P((hypre_ParCSRMatrix *A , double strength_threshold , hypre_ParCSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr ));

/* par_cycle.c */
int hypre_ParAMGCycle P((void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array ));

/* par_indepset.c */
int hypre_InitParAMGIndepSet P((hypre_ParCSRMatrix *S , double *measure_array ));
int hypre_ParAMGIndepSet P((hypre_ParCSRMatrix *S , hypre_CSRMatrix *S_ext , double *measure_array , int *graph_array , int graph_array_size , int *IS_marker ));

/* par_interp.c */
int hypre_ParAMGBuildInterp P((hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , hypre_ParCSRMatrix **P_ptr ));

/* par_laplace.c */
hypre_ParCSRMatrix *hypre_GenerateLaplacian P((MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value ));
int map P((int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part ));
void qsort0 P((int *v , int left , int right ));
void swap P((int *v , int i , int j ));

/* par_laplace_9pt.c */
hypre_ParCSRMatrix *hypre_GenerateLaplacian9pt P((MPI_Comm comm , int nx , int ny , int P , int Q , int p , int q , double *value ));
int map2 P((int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part ));

/* par_rap.c */
int hypre_ParAMGBuildCoarseOperator P((hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr ));
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

/* transpose.c */
int hypre_CSRMatrixTranspose P((hypre_CSRMatrix *A , hypre_CSRMatrix **AT ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

