# define	P(s) s

/* HYPRE_distributed_matrix.c */
int HYPRE_DistributedMatrixCreate P((MPI_Comm context, HYPRE_DistributedMatrix *matrix ));
int HYPRE_DistributedMatrixDestroy P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixLimitedDestroy P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixInitialize P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixAssemble P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixSetLocalStorageType P((HYPRE_DistributedMatrix matrix , int type ));
int HYPRE_DistributedMatrixGetLocalStorageType P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixSetLocalStorage P((HYPRE_DistributedMatrix matrix , void *LocalStorage ));
void *HYPRE_DistributedMatrixGetLocalStorage P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixSetTranslator P((HYPRE_DistributedMatrix matrix , void *Translator ));
void *HYPRE_DistributedMatrixGetTranslator P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixSetAuxiliaryData P((HYPRE_DistributedMatrix matrix , void *AuxiliaryData ));
void *HYPRE_DistributedMatrixGetAuxiliaryData P((HYPRE_DistributedMatrix matrix ));
MPI_Comm HYPRE_DistributedMatrixGetContext P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixGetDims P((HYPRE_DistributedMatrix matrix , int *M , int *N ));
int HYPRE_DistributedMatrixSetDims P((HYPRE_DistributedMatrix matrix , int M , int N ));
int HYPRE_DistributedMatrixPrint P((HYPRE_DistributedMatrix matrix ));
int HYPRE_DistributedMatrixGetLocalRange P((HYPRE_DistributedMatrix matrix , int *row_start , int *row_end, int *col_start, int *col_end ));
int HYPRE_DistributedMatrixGetRow P((HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values ));
int HYPRE_DistributedMatrixRestoreRow P((HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values ));

/* distributed_matrix.c */
hypre_DistributedMatrix *hypre_DistributedMatrixCreate P((MPI_Comm context ));
int hypre_DistributedMatrixDestroy P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixLimitedDestroy P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixInitialize P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixAssemble P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixSetLocalStorageType P((hypre_DistributedMatrix *matrix , int type ));
int hypre_DistributedMatrixGetLocalStorageType P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixSetLocalStorage P((hypre_DistributedMatrix *matrix , void *local_storage ));
void *hypre_DistributedMatrixGetLocalStorage P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixSetTranslator P((hypre_DistributedMatrix *matrix , void *translator ));
void *hypre_DistributedMatrixGetTranslator P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixSetAuxiliaryData P((hypre_DistributedMatrix *matrix , void *auxiliary_data ));
void *hypre_DistributedMatrixGetAuxiliaryData P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixPrint P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixGetLocalRange P((hypre_DistributedMatrix *matrix , int *row_start , int *row_end, int *col_start, int *col_end ));
int hypre_DistributedMatrixGetRow P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));
int hypre_DistributedMatrixRestoreRow P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));

/* distributed_matrix_PETSc.c */
int hypre_DistributedMatrixDestroyPETSc P((hypre_DistributedMatrix *distributed_matrix ));
int hypre_DistributedMatrixPrintPETSc P((hypre_DistributedMatrix *matrix ));
int hypre_DistributedMatrixGetLocalRangePETSc P((hypre_DistributedMatrix *matrix , int *start , int *end ));
int hypre_DistributedMatrixGetRowPETSc P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));
int hypre_DistributedMatrixRestoreRowPETSc P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));

/* hypre.c */

#undef P
