#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* SPamg.c */
int main P((int argc , char *argv []));

/* amg.c */
void *hypre_AMGInitialize P((void ));
hypre_AMGData *hypre_AMGNewData P((int max_levels , double strong_threshold , int interp_type , int max_iter , int cycle_type , double tol , int *num_grid_sweeps , int *grid_relax_type , int *grid_relax_points [4 ], int ioutdat , char log_file_name [256 ]));
void hypre_AMGFinalize P((void *data ));
void hypre_AMGSetMaxLevels P((int max_levels , void *data ));
void hypre_AMGSetStrongThreshold P((double strong_threshold , void *data ));
void hypre_AMGSetInterpType P((int interp_type , void *data ));
void hypre_AMGSetMaxIter P((int max_iter , void *data ));
void hypre_AMGSetCycleType P((int cycle_type , void *data ));
void hypre_AMGSetTol P((double tol , void *data ));
void hypre_AMGSetNumGridSweeps P((int *num_grid_sweeps , void *data ));
void hypre_AMGSetGridRelaxType P((int *grid_relax_type , void *data ));
void hypre_AMGSetGridRelaxPoints P((int **grid_relax_points , void *data ));
void hypre_AMGSetIOutDat P((int ioutdat , void *data ));
void hypre_AMGSetLogFileName P((char *log_file_name , void *data ));
void hypre_AMGSetLogging P((int ioutdat , char *log_file_name , void *data ));
void hypre_AMGSetNumUnknowns P((int num_unknowns , void *data ));
void hypre_AMGSetNumPoints P((int num_points , void *data ));
void hypre_AMGSetUnknownMap P((int *unknown_map , void *data ));
void hypre_AMGSetPointMap P((int *point_map , void *data ));
void hypre_AMGSetVatPoint P((int *v_at_point , void *data ));

/* amg_setup.c */
int hypre_AMGSetup P((hypre_AMGData *amg_data , hypre_CSRMatrix *A ));

/* amg_solve.c */
int hypre_AMGSolve P((hypre_AMGData *amg_data , hypre_Vector *f , hypre_Vector *u ));

/* amgstats.c */
int hypre_AMGSetupStats P((hypre_AMGData *amg_data ));
void hypre_WriteSolverParams P((void *data ));

/* coarsen.c */
int hypre_AMGCoarsen P((hypre_CSRMatrix *A , double strong_threshold , int **CF_marker_ptr , hypre_CSRMatrix **S_ptr ));
void debug_out P((int *ST_data , int *ST_i , int *ST_j , int num_variables ));

/* cycle.c */
int hypre_AMGCycle P((hypre_AMGData *amg_data , hypre_Vector **F_array , hypre_Vector **U_array ));

/* indepset.c */
int hypre_AMGIndepSet P((int *ST_i , int *ST_j , int *S_i , int *S_j , int num_variables , double *measure_array , int *IS_array , int *IS_size ));

/* interp.c */
int hypre_AMGBuildInterp P((hypre_CSRMatrix *A , int *CF_marker , hypre_CSRMatrix *S , hypre_CSRMatrix **P_ptr ));

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
