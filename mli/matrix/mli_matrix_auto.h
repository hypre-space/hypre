
#include <HYPRE_config.h>

#ifndef MLI_MATRIX_AUTO_HEADER
#define MLI_MATRIX_AUTO_HEADER

#include "utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Parallel CSR Boolean Matrix data structures
 *
 *****************************************************************************/

#ifndef __MLI_PARCSR_BOOLEAN_Matrix__
#define __MLI_PARCSR_BOOLEAN_Matrix__

/*#include <mpi.h>*/ /* jfp */
#include "communication.h"
#include "../../seq_mv/csr_matrix.h"
#include "../../parcsr_mv/par_csr_matrix.h"

/*--------------------------------------------------------------------------
 * CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    *i;
   int    *j;
   int     num_rows;
   int     num_cols;
   int     num_nonzeros;
   int     owns_data;

} MLI_CSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define MLI_CSRBooleanMatrix_Get_I(matrix)        ((matrix)->i)
#define MLI_CSRBooleanMatrix_Get_J(matrix)        ((matrix)->j)
#define MLI_CSRBooleanMatrix_Get_NRows(matrix)    ((matrix)->num_rows)
#define MLI_CSRBooleanMatrix_Get_NCols(matrix)    ((matrix)->num_cols)
#define MLI_CSRBooleanMatrix_Get_NNZ(matrix)      ((matrix)->num_nonzeros)
#define MLI_CSRBooleanMatrix_Get_OwnsData(matrix) ((matrix)->owns_data)

/*--------------------------------------------------------------------------
 * Parallel CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
   int                   global_num_rows;
   int                   global_num_cols;
   int                   first_row_index;
   int                   first_col_diag;
   MLI_CSRBooleanMatrix *diag;
   MLI_CSRBooleanMatrix *offd;
   int	                *col_map_offd; 
   int 	                *row_starts; 
   int 	                *col_starts;
   hypre_ParCSRCommPkg  *comm_pkg;
   int                   owns_data;
   int                   owns_row_starts;
   int                   owns_col_starts;
   int                   num_nonzeros;
   int                  *rowindices;
   int                   getrowactive;

} MLI_ParCSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define MLI_ParCSRBooleanMatrix_Get_Comm(matrix)          ((matrix)->comm)
#define MLI_ParCSRBooleanMatrix_Get_GlobalNRows(matrix)   ((matrix)->global_num_rows)
#define MLI_ParCSRBooleanMatrix_Get_GlobalNCols(matrix)   ((matrix)->global_num_cols)
#define MLI_ParCSRBooleanMatrix_Get_StartRow(matrix)      ((matrix)->first_row_index)
#define MLI_ParCSRBooleanMatrix_Get_FirstRowIndex(matrix) ((matrix)->first_row_index)
#define MLI_ParCSRBooleanMatrix_Get_FirstColDiag(matrix)  ((matrix)->first_col_diag)
#define MLI_ParCSRBooleanMatrix_Get_Diag(matrix)          ((matrix)->diag)
#define MLI_ParCSRBooleanMatrix_Get_Offd(matrix)          ((matrix)->offd)
#define MLI_ParCSRBooleanMatrix_Get_ColMapOffd(matrix)    ((matrix)->col_map_offd)
#define MLI_ParCSRBooleanMatrix_Get_RowStarts(matrix)     ((matrix)->row_starts)
#define MLI_ParCSRBooleanMatrix_Get_ColStarts(matrix)     ((matrix)->col_starts)
#define MLI_ParCSRBooleanMatrix_Get_CommPkg(matrix)       ((matrix)->comm_pkg)
#define MLI_ParCSRBooleanMatrix_Get_OwnsData(matrix)      ((matrix)->owns_data)
#define MLI_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) ((matrix)->owns_row_starts)
#define MLI_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) ((matrix)->owns_col_starts)
#define MLI_ParCSRBooleanMatrix_Get_NRows(matrix)         ((matrix->diag->num_rows))
#define MLI_ParCSRBooleanMatrix_Get_NCols(matrix)         ((matrix->diag->num_cols))
#define MLI_ParCSRBooleanMatrix_Get_NNZ(matrix)           ((matrix)->num_nonzeros)
#define MLI_ParCSRBooleanMatrix_Get_Rowindices(matrix)    ((matrix)->rowindices)
#define MLI_ParCSRBooleanMatrix_Get_Getrowactive(matrix)  ((matrix)->getrowactive)

#endif

/* driver_matmul.c */
int main( int argc , char *argv []);

/* mli_pcsr_bool_mat.c */
MLI_CSRBooleanMatrix *MLI_CSRBooleanMatrixCreate( int num_rows , int num_cols , int num_nonzeros );
int MLI_CSRBooleanMatrixDestroy( MLI_CSRBooleanMatrix *matrix );
int MLI_CSRBooleanMatrixInitialize( MLI_CSRBooleanMatrix *matrix );
int MLI_CSRBooleanMatrixSetDataOwner( MLI_CSRBooleanMatrix *matrix , int owns_data );
MLI_CSRBooleanMatrix *MLI_CSRBooleanMatrixRead( char *file_name );
int MLI_CSRBooleanMatrixPrint( MLI_CSRBooleanMatrix *matrix , char *file_name );
MLI_ParCSRBooleanMatrix *MLI_ParCSRBooleanMatrixCreate( MPI_Comm comm , int global_num_rows , int global_num_cols , int *row_starts , int *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd );
int MLI_ParCSRBooleanMatrixDestroy( MLI_ParCSRBooleanMatrix *matrix );
int MLI_ParCSRBooleanMatrixInitialize( MLI_ParCSRBooleanMatrix *matrix );
int MLI_ParCSRBooleanMatrixSetNNZ( MLI_ParCSRBooleanMatrix *matrix );
int MLI_ParCSRBooleanMatrixSetDataOwner( MLI_ParCSRBooleanMatrix *matrix , int owns_data );
int MLI_ParCSRBooleanMatrixSetRowStartsOwner( MLI_ParCSRBooleanMatrix *matrix , int owns_row_starts );
int MLI_ParCSRBooleanMatrixSetColStartsOwner( MLI_ParCSRBooleanMatrix *matrix , int owns_col_starts );
MLI_ParCSRBooleanMatrix *MLI_ParCSRBooleanMatrixRead( MPI_Comm comm , char *file_name );
int MLI_ParCSRBooleanMatrixPrint( MLI_ParCSRBooleanMatrix *matrix , char *file_name );
int MLI_ParCSRBooleanMatrixPrintIJ( MLI_ParCSRBooleanMatrix *matrix , char *filename );
int MLI_ParCSRBooleanMatrixGetLocalRange( MLI_ParCSRBooleanMatrix *matrix , int *row_start , int *row_end , int *col_start , int *col_end );
int MLI_ParCSRBooleanMatrixGetRow( MLI_ParCSRBooleanMatrix *mat , int row , int *size , int **col_ind );
int MLI_ParCSRBooleanMatrixRestoreRow( MLI_ParCSRBooleanMatrix *matrix , int row , int *size , int **col_ind );
int MLI_BuildCSRBooleanMatrixMPIDataType( int num_nonzeros , int num_rows , int *a_i , int *a_j , MPI_Datatype *csr_matrix_datatype );
MLI_ParCSRBooleanMatrix *MLI_CSRBooleanMatrixToParCSRBooleanMatrix( MPI_Comm comm , MLI_CSRBooleanMatrix *A , int *row_starts , int *col_starts );
int BooleanGenerateDiagAndOffd( MLI_CSRBooleanMatrix *A , MLI_ParCSRBooleanMatrix *matrix , int first_col_diag , int last_col_diag );

/* mli_pcsr_bool_matop.c */
MLI_ParCSRBooleanMatrix *MLI_ParBooleanMatmul( MLI_ParCSRBooleanMatrix *A , MLI_ParCSRBooleanMatrix *B );
MLI_CSRBooleanMatrix *MLI_ParCSRBooleanMatrixExtractBExt( MLI_ParCSRBooleanMatrix *B , MLI_ParCSRBooleanMatrix *A );


#ifdef __cplusplus
}
#endif

#endif

