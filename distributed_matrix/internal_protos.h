
/* HYPRE_distributed_matrix.c */
int HYPRE_DistributedMatrixCreate (MPI_Comm context, HYPRE_DistributedMatrix *matrix );
int HYPRE_DistributedMatrixDestroy (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixLimitedDestroy (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixInitialize (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixAssemble (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetLocalStorageType (HYPRE_DistributedMatrix matrix , int type );
int HYPRE_DistributedMatrixGetLocalStorageType (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetLocalStorage (HYPRE_DistributedMatrix matrix , void *LocalStorage );
void *HYPRE_DistributedMatrixGetLocalStorage (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetTranslator (HYPRE_DistributedMatrix matrix , void *Translator );
void *HYPRE_DistributedMatrixGetTranslator (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetAuxiliaryData (HYPRE_DistributedMatrix matrix , void *AuxiliaryData );
void *HYPRE_DistributedMatrixGetAuxiliaryData (HYPRE_DistributedMatrix matrix );
MPI_Comm HYPRE_DistributedMatrixGetContext (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixGetDims (HYPRE_DistributedMatrix matrix , int *M , int *N );
int HYPRE_DistributedMatrixSetDims (HYPRE_DistributedMatrix matrix , int M , int N );
int HYPRE_DistributedMatrixPrint (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixGetLocalRange (HYPRE_DistributedMatrix matrix , int *row_start , int *row_end, int *col_start, int *col_end );
int HYPRE_DistributedMatrixGetRow (HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values );
int HYPRE_DistributedMatrixRestoreRow (HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values );

/* distributed_matrix.c */
hypre_DistributedMatrix *hypre_DistributedMatrixCreate (MPI_Comm context );
int hypre_DistributedMatrixDestroy (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixLimitedDestroy (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixInitialize (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixAssemble (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetLocalStorageType (hypre_DistributedMatrix *matrix , int type );
int hypre_DistributedMatrixGetLocalStorageType (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetLocalStorage (hypre_DistributedMatrix *matrix , void *local_storage );
void *hypre_DistributedMatrixGetLocalStorage (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetTranslator (hypre_DistributedMatrix *matrix , void *translator );
void *hypre_DistributedMatrixGetTranslator (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetAuxiliaryData (hypre_DistributedMatrix *matrix , void *auxiliary_data );
void *hypre_DistributedMatrixGetAuxiliaryData (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixPrint (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixGetLocalRange (hypre_DistributedMatrix *matrix , int *row_start , int *row_end, int *col_start, int *col_end );
int hypre_DistributedMatrixGetRow (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );
int hypre_DistributedMatrixRestoreRow (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );

/* distributed_matrix_PETSc.c */
int hypre_DistributedMatrixDestroyPETSc (hypre_DistributedMatrix *distributed_matrix );
int hypre_DistributedMatrixPrintPETSc (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixGetLocalRangePETSc (hypre_DistributedMatrix *matrix , int *start , int *end );
int hypre_DistributedMatrixGetRowPETSc (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );
int hypre_DistributedMatrixRestoreRowPETSc (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );

