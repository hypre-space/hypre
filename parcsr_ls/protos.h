#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* par_rap.c */
int hypre_ParAMGBuildCoarseOperator P(( hypre_ParCSRMatrix *RT,
	hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P, 
	hypre_ParCSRMatrix **RAP_ptr, int *coarse_partitioning));
hypre_CSRMatrix *hypre_GeneratePExt P(( hypre_ParCSRMatrix *P,
	hypre_ParCSRMatrix *A ));
int BuildCSRJDataType P((int num_nonzeros, double *a_data, int *a_j,
	MPI_Datatype *csr_jdata_datatype));
hypre_CSRMatrix *hypre_ExchangeRAPData P((hypre_CSRMatrix *RAP_int,
	hypre_CommPkg *comm_pkg_RT));
int hypre_GenerateRAPCommPkg P((hypre_ParCSRMatrix *RAP, 
	hypre_ParCSRMatrix *A, int *partitioning));

/* transpose.c */
int hypre_CSRMatrixTranspose P((hypre_CSRMatrix *A, hypre_CSRMatrix **AT));

#undef P
