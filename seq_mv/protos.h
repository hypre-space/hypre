#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CreateCSRMatrix P((int num_rows , int num_cols , int *row_sizes ));
int HYPRE_DestroyCSRMatrix P((HYPRE_CSRMatrix matrix ));
int HYPRE_InitializeCSRMatrix P((HYPRE_CSRMatrix matrix ));
void HYPRE_PrintCSRMatrix P((HYPRE_CSRMatrix matrix , char *file_name ));

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_NewMappedMatrix P((void ));
int HYPRE_FreeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_LimitedFreeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_InitializeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_AssembleMappedMatrix P((HYPRE_MappedMatrix matrix ));
void HYPRE_PrintMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_GetMappedMatrixColIndex P((HYPRE_MappedMatrix matrix , int j ));
void *HYPRE_GetMappedMatrixMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_SetMappedMatrixMatrix P((HYPRE_MappedMatrix matrix , void *matrix_data ));
int HYPRE_SetMappedMatrixColMap P((HYPRE_MappedMatrix matrix , int (*ColMap )(int ,void *)));
int HYPRE_SetMappedMatrixMapData P((HYPRE_MappedMatrix matrix , void *MapData ));

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_NewMultiblockMatrix P((void ));
int HYPRE_FreeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_LimitedFreeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_InitializeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_AssembleMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
void HYPRE_PrintMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_SetMultiblockMatrixNumSubmatrices P((HYPRE_MultiblockMatrix matrix , int n ));
int HYPRE_SetMultiblockMatrixSubmatrixType P((HYPRE_MultiblockMatrix matrix , int j , int type ));

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_CreateVector P((int size ));
int HYPRE_DestroyVector P((HYPRE_Vector vector ));
int HYPRE_InitializeVector P((HYPRE_Vector vector ));
int HYPRE_PrintVector P((HYPRE_Vector vector , char *file_name ));

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CreateCSRMatrix P((int num_rows , int num_cols , int num_nonzeros ));
int hypre_DestroyCSRMatrix P((hypre_CSRMatrix *matrix ));
int hypre_InitializeCSRMatrix P((hypre_CSRMatrix *matrix ));
int hypre_SetCSRMatrixDataOwner P((hypre_CSRMatrix *matrix , int owns_data ));
hypre_CSRMatrix *hypre_ReadCSRMatrix P((char *file_name ));
int hypre_PrintCSRMatrix P((hypre_CSRMatrix *matrix , char *file_name ));

/* csr_matvec.c */
int hypre_Matvec P((double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));
int hypre_MatvecT P((double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));

/* mapped_matrix.c */
hypre_MappedMatrix *hypre_NewMappedMatrix P((void ));
int hypre_FreeMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_LimitedFreeMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_InitializeMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_AssembleMappedMatrix P((hypre_MappedMatrix *matrix ));
void hypre_PrintMappedMatrix P((hypre_MappedMatrix *matrix ));
int hypre_GetMappedMatrixColIndex P((hypre_MappedMatrix *matrix , int j ));
void *hypre_GetMappedMatrixMatrix P((hypre_MappedMatrix *matrix ));
int hypre_SetMappedMatrixMatrix P((hypre_MappedMatrix *matrix , void *matrix_data ));
int hypre_SetMappedMatrixColMap P((hypre_MappedMatrix *matrix , int (*ColMap )(int ,void *)));
int hypre_SetMappedMatrixMapData P((hypre_MappedMatrix *matrix , void *map_data ));

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_NewMultiblockMatrix P((void ));
int hypre_FreeMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_LimitedFreeMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_InitializeMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_AssembleMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
void hypre_PrintMultiblockMatrix P((hypre_MultiblockMatrix *matrix ));
int hypre_SetMultiblockMatrixNumSubmatrices P((hypre_MultiblockMatrix *matrix , int n ));
int hypre_SetMultiblockMatrixSubmatrixType P((hypre_MultiblockMatrix *matrix , int j , int type ));
int hypre_SetMultiblockMatrixSubmatrix P((hypre_MultiblockMatrix *matrix , int j , void *submatrix ));

/* vector.c */
hypre_Vector *hypre_CreateVector P((int size ));
int hypre_DestroyVector P((hypre_Vector *vector ));
int hypre_InitializeVector P((hypre_Vector *vector ));
int hypre_SetVectorDataOwner P((hypre_Vector *vector , int owns_data ));
hypre_Vector *hypre_ReadVector P((char *file_name ));
int hypre_PrintVector P((hypre_Vector *vector , char *file_name ));
int hypre_SetVectorConstantValues P((hypre_Vector *v , double value ));
int hypre_CopyVector P((hypre_Vector *x , hypre_Vector *y ));
int hypre_ScaleVector P((double alpha , hypre_Vector *y ));
int hypre_Axpy P((double alpha , hypre_Vector *x , hypre_Vector *y ));
double hypre_InnerProd P((hypre_Vector *x , hypre_Vector *y ));

/* csr_matop.c */
hypre_CSRMatrix *hypre_Matadd P((hypre_CSRMatrix *A, hypre_CSRMatrix *B));
hypre_CSRMatrix *hypre_Matmul P((hypre_CSRMatrix *A, hypre_CSRMatrix *B));
hypre_CSRMatrix *hypre_DeleteZerosInMatrix P((hypre_CSRMatrix *A, double tol));

#undef P
