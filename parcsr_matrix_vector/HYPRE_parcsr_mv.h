/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for HYPRE_parcsr_mv library
 *
 *****************************************************************************/

#ifndef HYPRE_PARCSR_MV_HEADER
#define HYPRE_PARCSR_MV_HEADER

#include "HYPRE_utilities.h"
#include "HYPRE_seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

/* this needs to be fixed */
typedef struct {int opaque;} *HYPRE_ParCSRMatrix;
typedef struct {int opaque;} *HYPRE_ParVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

# define        P(s) s

/* HYPRE_parcsr_matrix.c */
int HYPRE_ParCSRMatrixCreate P((MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd , HYPRE_ParCSRMatrix *matrix ));
int HYPRE_ParCSRMatrixDestroy P((HYPRE_ParCSRMatrix matrix ));
int HYPRE_ParCSRMatrixInitialize P((HYPRE_ParCSRMatrix matrix ));
int HYPRE_ParCSRMatrixRead P((MPI_Comm comm , char *file_name , HYPRE_ParCSRMatrix *matrix ));
int HYPRE_ParCSRMatrixPrint P((HYPRE_ParCSRMatrix matrix , char *file_name ));
int HYPRE_ParCSRMatrixGetComm P((HYPRE_ParCSRMatrix matrix , MPI_Comm *comm ));
int HYPRE_ParCSRMatrixGetDims P((HYPRE_ParCSRMatrix matrix , int *M , int *N ));
int HYPRE_ParCSRMatrixGetRowPartitioning P((HYPRE_ParCSRMatrix matrix , int **row_partitioning_ptr ));
int HYPRE_ParCSRMatrixGetColPartitioning P((HYPRE_ParCSRMatrix matrix , int **col_partitioning_ptr ));
int HYPRE_ParCSRMatrixGetLocalRange P((HYPRE_ParCSRMatrix matrix , int *row_start , int *row_end , int *col_start , int *col_end ));
int HYPRE_ParCSRMatrixGetRow P((HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values ));
int HYPRE_ParCSRMatrixRestoreRow P((HYPRE_ParCSRMatrix matrix , int row , int *size , int **col_ind , double **values ));
int HYPRE_CSRMatrixToParCSRMatrix P((MPI_Comm comm , HYPRE_CSRMatrix A_CSR , int *row_partitioning , int *col_partitioning , HYPRE_ParCSRMatrix *matrix ));
int HYPRE_ParCSRMatrixMatvec P((double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y ));

/* HYPRE_parcsr_vector.c */
int HYPRE_ParVectorCreate P((MPI_Comm comm , int global_size , int *partitioning , HYPRE_ParVector *vector ));
int HYPRE_ParVectorDestroy P((HYPRE_ParVector vector ));
int HYPRE_ParVectorInitialize P((HYPRE_ParVector vector ));
int HYPRE_ParVectorRead P((MPI_Comm comm , char *file_name, HYPRE_ParVector *vector ));
int HYPRE_ParVectorPrint P((HYPRE_ParVector vector , char *file_name ));
int HYPRE_ParVectorSetConstantValues P((HYPRE_ParVector vector , double value ));
int HYPRE_ParVectorSetRandomValues P((HYPRE_ParVector vector , int seed ));
int HYPRE_ParVectorCopy P((HYPRE_ParVector x , HYPRE_ParVector y ));
int HYPRE_ParVectorScale P((double value , HYPRE_ParVector x ));
int HYPRE_ParVectorInnerProd P((HYPRE_ParVector x , HYPRE_ParVector y , double *prod ));
int HYPRE_VectorToParVector P((MPI_Comm comm , HYPRE_Vector b , int *partitioning , HYPRE_ParVector *vector ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

