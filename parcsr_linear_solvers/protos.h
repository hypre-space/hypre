#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* coarsen.c */
int hypre_AMGCoarsen P((hypre_CSRMatrix *A , double strength_threshold , hypre_CSRMatrix **S_ptr , int **CF_marker_ptr , int *coarse_size_ptr ));

/* driver.c */
int main P((int argc , char *argv []));

/* driver_rap.c */
int main P((int argc , char *argv []));

/* indepset.c */
int hypre_InitAMGIndepSet P((hypre_CSRMatrix *S , double *measure_array ));
int hypre_AMGIndepSet P((hypre_CSRMatrix *S , double *measure_array , int *graph_array , int graph_array_size , int *IS_marker ));

/* par_amg.c */
void *hypre_ParAMGInitialize P((void ));
void *hypre_ParAMGNewData P((int max_levels , double strong_threshold , int interp_type , int max_iter , int cycle_type , double tol , int *num_grid_sweeps , int *grid_relax_type , int *grid_relax_points [4 ], int ioutdat , char log_file_name [256 ]));
void hypre_ParAMGFinalize P((void *data ));
void hypre_ParAMGSetMaxLevels P((int max_levels , void *data ));
void hypre_ParAMGSetStrongThreshold P((double strong_threshold , void *data ));
void hypre_ParAMGSetInterpType P((int interp_type , void *data ));
void hypre_ParAMGSetMaxIter P((int max_iter , void *data ));
void hypre_ParAMGSetCycleType P((int cycle_type , void *data ));
void hypre_ParAMGSetTol P((double tol , void *data ));
void hypre_ParAMGSetNumGridSweeps P((int *num_grid_sweeps , void *data ));
void hypre_ParAMGSetGridRelaxType P((int *grid_relax_type , void *data ));
void hypre_ParAMGSetGridRelaxPoints P((int **grid_relax_points , void *data ));
void hypre_ParAMGSetIOutDat P((int ioutdat , void *data ));
void hypre_ParAMGSetLogFileName P((char *log_file_name , void *data ));
void hypre_ParAMGSetLogging P((int ioutdat , char *log_file_name , void *data ));
void hypre_ParAMGSetNumUnknowns P((int num_unknowns , void *data ));
void hypre_ParAMGSetNumPoints P((int num_points , void *data ));
void hypre_ParAMGSetUnknownMap P((int *unknown_map , void *data ));
void hypre_ParAMGSetPointMap P((int *point_map , void *data ));
void hypre_ParAMGSetVatPoint P((int *v_at_point , void *data ));

/* par_amgsetup.c */
int hypre_ParAMGSetup P((void *vamg_data , hypre_ParCSRMatrix *A ));

/* par_interp.c */
int hypre_ParAMGBuildInterp P((hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , hypre_ParCSRMatrix **P_ptr , int **coarse_partitioning_ptr ));

/* par_laplace.c */
hypre_ParCSRMatrix *hypre_GenerateLaplacian P((MPI_Comm comm , int nx , int ny , int nz , int P , int Q , int R , int p , int q , int r , double *value , int **global_part_ptr ));
int map P((int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , int *nx_part , int *ny_part , int *nz_part , int *global_part ));
void qsort0 P((int *v , int left , int right ));
void swap P((int *v , int i , int j ));

/* par_rap.c */
int hypre_ParAMGBuildCoarseOperator P((hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr , int *coarse_partitioning ));
hypre_CSRMatrix *hypre_GeneratePExt P((hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *A ));
hypre_CSRMatrix *hypre_ExchangeRAPData P((hypre_CSRMatrix *RAP_int , hypre_CommPkg *comm_pkg_RT ));

/* par_rap_communication.c */
int hypre_GetCommPkgRTFromCommPkgA P((hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , int *partitioning ));
hypre_CommPkg *hypre_GenerateSendMapAndCommPkg P((MPI_Comm comm , int num_sends , int num_recvs , int *recv_procs , int *send_procs , int *recv_vec_starts , hypre_ParCSRMatrix *A ));
int hypre_GenerateRAPCommPkg P((hypre_ParCSRMatrix *RAP , hypre_ParCSRMatrix *A , int *partitioning ));

/* random.c */
void hypre_SeedRand P((int seed ));
double hypre_Rand P((void ));

/* transpose.c */
int hypre_CSRMatrixTranspose P((hypre_CSRMatrix *A , hypre_CSRMatrix **AT ));

#undef P
