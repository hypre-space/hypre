/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




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
HYPRE_Int HYPRE_ParCSRMatrixCreate( MPI_Comm comm , HYPRE_Int global_num_rows , HYPRE_Int global_num_cols , HYPRE_Int *row_starts , HYPRE_Int *col_starts , HYPRE_Int num_cols_offd , HYPRE_Int num_nonzeros_diag , HYPRE_Int num_nonzeros_offd , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixDestroy( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixInitialize( HYPRE_ParCSRMatrix matrix );
HYPRE_Int HYPRE_ParCSRMatrixRead( MPI_Comm comm , const char *file_name , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixPrint( HYPRE_ParCSRMatrix matrix , const char *file_name );
HYPRE_Int HYPRE_ParCSRMatrixGetComm( HYPRE_ParCSRMatrix matrix , MPI_Comm *comm );
HYPRE_Int HYPRE_ParCSRMatrixGetDims( HYPRE_ParCSRMatrix matrix , HYPRE_Int *M , HYPRE_Int *N );
HYPRE_Int HYPRE_ParCSRMatrixGetRowPartitioning( HYPRE_ParCSRMatrix matrix , HYPRE_Int **row_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetColPartitioning( HYPRE_ParCSRMatrix matrix , HYPRE_Int **col_partitioning_ptr );
HYPRE_Int HYPRE_ParCSRMatrixGetLocalRange( HYPRE_ParCSRMatrix matrix , HYPRE_Int *row_start , HYPRE_Int *row_end , HYPRE_Int *col_start , HYPRE_Int *col_end );
HYPRE_Int HYPRE_ParCSRMatrixGetRow( HYPRE_ParCSRMatrix matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );
HYPRE_Int HYPRE_ParCSRMatrixRestoreRow( HYPRE_ParCSRMatrix matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );
HYPRE_Int HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , HYPRE_Int *row_partitioning , HYPRE_Int *col_partitioning , HYPRE_ParCSRMatrix *matrix );
HYPRE_Int HYPRE_ParCSRMatrixMatvec( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParCSRMatrixMatvecT( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );

/* HYPRE_parcsr_vector.c */
HYPRE_Int HYPRE_ParVectorCreate( MPI_Comm comm , HYPRE_Int global_size , HYPRE_Int *partitioning , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorDestroy( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorInitialize( HYPRE_ParVector vector );
HYPRE_Int HYPRE_ParVectorRead( MPI_Comm comm , const char *file_name , HYPRE_ParVector *vector );
HYPRE_Int HYPRE_ParVectorPrint( HYPRE_ParVector vector , const char *file_name );
HYPRE_Int HYPRE_ParVectorSetConstantValues( HYPRE_ParVector vector , double value );
HYPRE_Int HYPRE_ParVectorSetRandomValues( HYPRE_ParVector vector , HYPRE_Int seed );
HYPRE_Int HYPRE_ParVectorCopy( HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_Int HYPRE_ParVectorScale( double value , HYPRE_ParVector x );
HYPRE_Int HYPRE_ParVectorInnerProd( HYPRE_ParVector x , HYPRE_ParVector y , double *prod );
HYPRE_Int HYPRE_VectorToParVector( MPI_Comm comm , HYPRE_Vector b , HYPRE_Int *partitioning , HYPRE_ParVector *vector );

#ifdef __cplusplus
}
#endif

#endif

