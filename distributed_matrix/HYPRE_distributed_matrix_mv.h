
/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the hypre_DistributedMatrix structures
 *
 *****************************************************************************/

#ifndef HYPRE_DISTRIBUTED_MATRIX_MV_HEADER
#define HYPRE_DISTRIBUTED_MATRIX_MV_HEADER


typedef void *HYPRE_DistributedMatrix;

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

#undef P
#endif
