#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


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
int HYPRE_GetDistributedMatrixLocalRange P((HYPRE_DistributedMatrix matrix , int *start , int *end ));
int HYPRE_GetDistributedMatrixRow P((HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values ));
int HYPRE_RestoreDistributedMatrixRow P((HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values ));

#undef P
