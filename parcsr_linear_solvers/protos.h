#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* coarsen.c */
int hypre_AMGCoarsen P((hypre_CSRMatrix *A , double strength_threshold , int **CF_marker_ptr , hypre_CSRMatrix **S_ptr ));
void debug_out P((int *ST_data , int *ST_i , int *ST_j , int num_variables ));

/* driver_rap.c */
int main P((int argc , char *argv []));

/* indepset.c */
int hypre_AMGIndepSet P((hypre_CSRMatrix *S , double *measure_array , double *work_array , int *graph_array , int graph_array_size , int *IS_size_ptr ));

/* par_rap.c */
int hypre_ParAMGBuildCoarseOperator P((hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr , int *coarse_partitioning ));
hypre_CSRMatrix *hypre_GeneratePExt P((hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *A ));
int BuildCSRJDataType P((int num_nonzeros , double *a_data , int *a_j , MPI_Datatype *csr_jdata_datatype ));
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

/* par_laplace.c */
hypre_ParCSRMatrix *hypre_GenerateLaplacian P((MPI_Comm comm, int nx, int ny,
	int nz, int P, int Q, int R, int p, int q, int r, double *value,
	int **global_part_ptr));
int map P((int ix, int iy, int iz, int p, int q, int r, int P, int Q, int R,
	int *nx_part, int *ny_part, int *nz_part, int *global_part));
void qsort0 P((int *v, int left, int right));
void swap P((int *v, int i, int j));

#undef P
