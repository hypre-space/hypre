#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 

/* par_vector.c */
hypre_ParVector *hypre_CreateParVector P((MPI_Comm comm, int global_size, 
	int first_index, int local_size ));
int hypre_DestroyParVector P((hypre_ParVector *vector ));
int hypre_InitializeParVector P((hypre_ParVector *vector ));
int hypre_SetParVectorDataOwner P((hypre_ParVector *vector , int owns_data ));
int hypre_PrintParVector P((hypre_ParVector *vector, char *file_name ));
int hypre_SetParVectorConstantValues P((hypre_ParVector *v , double value ));
int hypre_CopyParVector P((hypre_ParVector *x , hypre_ParVector *y ));
int hypre_ScaleParVector P((double alpha , hypre_ParVector *y ));
int hypre_ParAxpy P((double alpha , hypre_ParVector *x , hypre_ParVector *y ));
double hypre_ParInnerProd P((MPI_Comm comm, hypre_ParVector *x , 
   	hypre_ParVector *y ));
hypre_ParVector *hypre_VectorToParVector P((MPI_Comm comm, hypre_Vector *v,
	int *vec_starts));
int hypre_BuildParVectorMPITypes P((MPI_Comm comm, int vec_len, 
	int *vec_starts, MPI_Datatype *vector_mpi_types));

/* communication.c */
hypre_CommHandle *hypre_InitializeCommunication P((int job,
	hypre_CommPkg *comm_pkg, void *send_data, void *recv_data));
int hypre_FinalizeCommunication P((hypre_CommHandle *comm_handle));
int hypre_GenerateMatvecCommunicationInfo P((hypre_ParCSRMatrix *A, 
	int *row_part_starts, int *col_part_starts));
hypre_VectorCommPkg *hypre_InitializeVectorCommPkg P((MPI_Comm comm, 
	int vec_len, int *vec_starts));
int hypre_DestroyVectorCommPkg P(( hypre_VectorCommPkg *vector_comm_pkg));
int hypre_DestroyMatvecCommPkg P(( hypre_CommPkg *comm_pkg));
int BuildCSRMatrixMPIDataType P((int num_nonzeros, int num_rows,
	double *a_data, int *a_i, int *a_j, MPI_Datatype *csr_matrix_datatype));

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_CreateParCSRMatrix P((MPI_Comm comm,
		int global_num_rows, int global_num_cols,
		int first_row_index, int first_col_diag,
		int local_num_rows, int local_num_cols, int num_cols_offd,
		int num_nonzeros_diag, int num_nonzeros_offd));
int hypre_DestroyParCSRMatrix P(( hypre_ParCSRMatrix *matrix));
int hypre_InitializeParCSRMatrix P(( hypre_ParCSRMatrix *matrix));
hypre_ParCSRMatrix *hypre_ReadParCSRMatrix P(( MPI_Comm comm, char *file_name));
int hypre_PrintParCSRMatrix P(( hypre_ParCSRMatrix *matrix, char *file_name));
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix P((MPI_Comm comm,
	hypre_CSRMatrix *A, int **row_starts_ptr, int **col_starts_ptr));
int GenerateDiagAndOffd P((hypre_CSRMatrix *A, hypre_ParCSRMatrix *matrix,
	int first_col_diag, int last_col_diag));

/* par_csr_matvec.c */
int hypre_ParMatvec P((double alpha, hypre_ParCSRMatrix *A, 
	hypre_ParVector *x, double beta, hypre_ParVector *y));
int hypre_ParMatvecT P((double alpha, hypre_ParCSRMatrix *A, 
	hypre_ParVector *x, double beta, hypre_ParVector *y));


#undef P
