/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

struct hypre_ParCSRMatrix_struct;
typedef struct hypre_ParCSRMatrix_struct *HYPRE_ParCSRMatrix;
struct hypre_ParVector_struct;
typedef struct hypre_ParVector_struct *HYPRE_ParVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* HYPRE_parcsr_matrix.c */
HYPRE_Int HYPRE_ParCSRMatrixCreate( MPI_Comm comm , HYPRE_BigInt global_num_rows , HYPRE_BigInt global_num_cols , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixInitialize( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixRead( MPI_Comm comm , const char *file_name , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixPrint( HYPRE_ParCSRMatrix matrix , const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixGetComm( HYPRE_ParCSRMatrix matrix , MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetDims( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt *M , HYPRE_BigInt *N );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end , HYPRE_BigInt *col_start , HYPRE_BigInt *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetRow( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Complex **values );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Complex **values );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , HYPRE_BigInt *row_partitioning , HYPRE_BigInt *col_partitioning , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixMatvec( HYPRE_Complex alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , HYPRE_Complex beta , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT( HYPRE_Complex alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , HYPRE_Complex beta , HYPRE_ParVector y );

/* HYPRE_parcsr_vector.c */
HYPRE_Int HYPRE_ParVectorCreate( MPI_Comm comm , HYPRE_BigInt global_size , HYPRE_BigInt *partitioning , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorDestroy( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInitialize( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorRead( MPI_Comm comm , const char *file_name , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorPrint( HYPRE_ParVector vector , const char *file_name );
HYPRE_Int HYPRE_ParVectorSetConstantValues( HYPRE_ParVector vector , HYPRE_Complex value );
HYPRE_Int HYPRE_ParVectorSetRandomValues( HYPRE_ParVector vector , HYPRE_Int seed );
HYPRE_Int HYPRE_ParVectorCopy( HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorScale( HYPRE_Complex value , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorInnerProd( HYPRE_ParVector x , HYPRE_ParVector y , HYPRE_Real *prod );
HYPRE_Int HYPRE_VectorToParVector( MPI_Comm comm , HYPRE_Vector b , HYPRE_BigInt *partitioning , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorGetValues( HYPRE_ParVector vector , HYPRE_Int num_values , HYPRE_BigInt *indices, HYPRE_Complex *values );

#ifdef __cplusplus
}
#endif

#endif

