# define	P(s) s

/* HYPRE_distributed_matrix.c */
HYPRE_DistributedMatrix HYPRE_NewDistributedMatrix P((MPI_Comm context ));
int HYPRE_FreeDistributedMatrix P((HYPRE_DistributedMatrix matrix ));
int HYPRE_LimitedFreeDistributedMatrix P((HYPRE_DistributedMatrix matrix ));
int HYPRE_InitializeDistributedMatrix P((HYPRE_DistributedMatrix matrix ));
int HYPRE_AssembleDistributedMatrix P((HYPRE_DistributedMatrix matrix ));
int HYPRE_SetDistributedMatrixLocalStorageType P((HYPRE_DistributedMatrix matrix , int type ));
int HYPRE_GetDistributedMatrixLocalStorageType P((HYPRE_DistributedMatrix matrix ));
int HYPRE_SetDistributedMatrixLocalStorage P((HYPRE_DistributedMatrix matrix , void *LocalStorage ));
void *HYPRE_GetDistributedMatrixLocalStorage P((HYPRE_DistributedMatrix matrix ));
int HYPRE_SetDistributedMatrixTranslator P((HYPRE_DistributedMatrix matrix , void *Translator ));
void *HYPRE_GetDistributedMatrixTranslator P((HYPRE_DistributedMatrix matrix ));
int HYPRE_SetDistributedMatrixAuxiliaryData P((HYPRE_DistributedMatrix matrix , void *AuxiliaryData ));
void *HYPRE_GetDistributedMatrixAuxiliaryData P((HYPRE_DistributedMatrix matrix ));
MPI_Comm HYPRE_GetDistributedMatrixContext P((HYPRE_DistributedMatrix matrix ));
int HYPRE_GetDistributedMatrixDims P((HYPRE_DistributedMatrix matrix , int *M , int *N ));
int HYPRE_SetDistributedMatrixDims P((HYPRE_DistributedMatrix matrix , int M , int N ));
int HYPRE_PrintDistributedMatrix P((HYPRE_DistributedMatrix matrix ));
int HYPRE_GetDistributedMatrixLocalRange P((HYPRE_DistributedMatrix matrix , int *row_start , int *row_end, int *col_start, int *col_end ));
int HYPRE_GetDistributedMatrixRow P((HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values ));
int HYPRE_RestoreDistributedMatrixRow P((HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values ));

/* distributed_matrix.c */
hypre_DistributedMatrix *hypre_NewDistributedMatrix P((MPI_Comm context ));
int hypre_FreeDistributedMatrix P((hypre_DistributedMatrix *matrix ));
int hypre_LimitedFreeDistributedMatrix P((hypre_DistributedMatrix *matrix ));
int hypre_InitializeDistributedMatrix P((hypre_DistributedMatrix *matrix ));
int hypre_AssembleDistributedMatrix P((hypre_DistributedMatrix *matrix ));
int hypre_SetDistributedMatrixLocalStorageType P((hypre_DistributedMatrix *matrix , int type ));
int hypre_GetDistributedMatrixLocalStorageType P((hypre_DistributedMatrix *matrix ));
int hypre_SetDistributedMatrixLocalStorage P((hypre_DistributedMatrix *matrix , void *local_storage ));
void *hypre_GetDistributedMatrixLocalStorage P((hypre_DistributedMatrix *matrix ));
int hypre_SetDistributedMatrixTranslator P((hypre_DistributedMatrix *matrix , void *translator ));
void *hypre_GetDistributedMatrixTranslator P((hypre_DistributedMatrix *matrix ));
int hypre_SetDistributedMatrixAuxiliaryData P((hypre_DistributedMatrix *matrix , void *auxiliary_data ));
void *hypre_GetDistributedMatrixAuxiliaryData P((hypre_DistributedMatrix *matrix ));
int hypre_PrintDistributedMatrix P((hypre_DistributedMatrix *matrix ));
int hypre_GetDistributedMatrixLocalRange P((hypre_DistributedMatrix *matrix , int *row_start , int *row_end, int *col_start, int *col_end ));
int hypre_GetDistributedMatrixRow P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));
int hypre_RestoreDistributedMatrixRow P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));

/* distributed_matrix_PETSc.c */
int hypre_FreeDistributedMatrixPETSc P((hypre_DistributedMatrix *distributed_matrix ));
int hypre_PrintDistributedMatrixPETSc P((hypre_DistributedMatrix *matrix ));
int hypre_GetDistributedMatrixLocalRangePETSc P((hypre_DistributedMatrix *matrix , int *start , int *end ));
int hypre_GetDistributedMatrixRowPETSc P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));
int hypre_RestoreDistributedMatrixRowPETSc P((hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values ));

/* hypre.c */

#undef P
