
#include "HYPRE_ls.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "utilities.h"
#include "mv.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_amg.c */
HYPRE_Solver HYPRE_AMGInitialize P((void ));
int HYPRE_AMGFinalize P((HYPRE_Solver solver ));
int HYPRE_AMGSetup P((HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x ));
int HYPRE_AMGSolve P((HYPRE_Solver solver , HYPRE_CSRMatrix A , HYPRE_Vector b , HYPRE_Vector x ));
int HYPRE_AMGSetMaxLevels P((HYPRE_Solver solver , int max_levels ));
int HYPRE_AMGSetStrongThreshold P((HYPRE_Solver solver , double strong_threshold ));
int HYPRE_AMGSetInterpType P((HYPRE_Solver solver , int interp_type ));
int HYPRE_AMGSetMaxIter P((HYPRE_Solver solver , int max_iter ));
int HYPRE_AMGSetCycleType P((HYPRE_Solver solver , int cycle_type ));
int HYPRE_AMGSetTol P((HYPRE_Solver solver , double tol ));
int HYPRE_AMGSetNumGridSweeps P((HYPRE_Solver solver , int *num_grid_sweeps ));
int HYPRE_AMGSetGridRelaxType P((HYPRE_Solver solver , int *grid_relax_type ));
int HYPRE_AMGSetGridRelaxPoints P((HYPRE_Solver solver , int **grid_relax_points ));
int HYPRE_AMGSetIOutDat P((HYPRE_Solver solver , int ioutdat ));
int HYPRE_AMGSetLogFileName P((HYPRE_Solver solver , char *log_file_name ));
int HYPRE_AMGSetLogging P((HYPRE_Solver solver , int ioutdat , char *log_file_name ));

/* SPamg-pcg.c */
int main P((int argc , char *argv []));

/* amg.c */
void *hypre_AMGInitialize P((void ));
int hypre_AMGFinalize P((void *data ));
int hypre_AMGSetMaxLevels P((void *data , int max_levels ));
int hypre_AMGSetStrongThreshold P((void *data , double strong_threshold ));
int hypre_AMGSetInterpType P((void *data , int interp_type ));
int hypre_AMGSetMaxIter P((void *data , int max_iter ));
int hypre_AMGSetCycleType P((void *data , int cycle_type ));
int hypre_AMGSetTol P((void *data , double tol ));
int hypre_AMGSetNumGridSweeps P((void *data , int *num_grid_sweeps ));
int hypre_AMGSetGridRelaxType P((void *data , int *grid_relax_type ));
int hypre_AMGSetGridRelaxPoints P((void *data , int **grid_relax_points ));
int hypre_AMGSetIOutDat P((void *data , int ioutdat ));
int hypre_AMGSetLogFileName P((void *data , char *log_file_name ));
int hypre_AMGSetLogging P((void *data , int ioutdat , char *log_file_name ));
int hypre_AMGSetNumUnknowns P((void *data , int num_unknowns ));
int hypre_AMGSetNumPoints P((void *data , int num_points ));
int hypre_AMGSetUnknownMap P((void *data , int *unknown_map ));
int hypre_AMGSetPointMap P((void *data , int *point_map ));
int hypre_AMGSetVatPoint P((void *data , int *v_at_point ));

/* amg_setup.c */
int hypre_AMGSetup P((void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u ));

/* amg_solve.c */
int hypre_AMGSolve P((void *amg_vdata , hypre_CSRMatrix *A , hypre_Vector *f , hypre_Vector *u ));

/* amgstats.c */
int hypre_AMGSetupStats P((void *amg_vdata ));
void hypre_WriteSolverParams P((void *data ));

/* coarsen.c */
int hypre_AMGCoarsen P((hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr ));

/* cycle.c */
int hypre_AMGCycle P((void *amg_vdata , hypre_Vector **F_array , hypre_Vector **U_array ));

/* driver.c */
int main P((int argc , char *argv []));
int BuildFromFile P((int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr ));
int BuildLaplacian P((int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr ));
int BuildLaplacian9pt P((int argc , char *argv [], int arg_index , hypre_CSRMatrix **A_ptr ));

/* indepset.c */
int hypre_InitAMGIndepSet P((hypre_CSRMatrix *S , double *measure_array ));
int hypre_AMGIndepSet P((hypre_CSRMatrix *S , double *measure_array , int *graph_array , int graph_array_size , int *IS_marker ));

/* interp.c */
int hypre_AMGBuildInterp P((hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , hypre_CSRMatrix **P_ptr ));

/* laplace.c */
hypre_CSRMatrix *hypre_GenerateLaplacian P((int nx , int ny , int nz , int P , int Q , int R , double *value ));
int map P((int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part ));

/* laplace_9pt.c */
hypre_CSRMatrix *hypre_GenerateLaplacian9pt P((int nx , int ny , int P , int Q , double *value ));
int map2 P((int ix , int iy , int p , int q , int P , int Q , int *nx_part , int *ny_part , int *global_part ));

/* pcg.c */
void PCG P((hypre_Vector *x , hypre_Vector *b , double tol , void *data ));
void PCGSetup P((hypre_CSRMatrix *A , int (*precond )(), void *precond_data , void *data ));

/* random.c */
void hypre_SeedRand P((int seed ));
double hypre_Rand P((void ));

/* rap.c */
int hypre_AMGBuildCoarseOperator P((hypre_CSRMatrix *RT , hypre_CSRMatrix *A , hypre_CSRMatrix *P , hypre_CSRMatrix **RAP_ptr ));

/* relax.c */
int hypre_AMGRelax P((hypre_CSRMatrix *A , hypre_Vector *f , int *cf_marker , int relax_type , int relax_points , hypre_Vector *u , hypre_Vector *Vtemp ));
int gselim P((double *A , double *x , int n ));

/* transpose.c */
int hypre_CSRMatrixTranspose P((hypre_CSRMatrix *A , hypre_CSRMatrix **AT ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

