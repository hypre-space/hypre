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
 * Header file for HYPRE_IJMatrix and HYPRE_IJVector libraries
 *
 *****************************************************************************/

#ifndef _HYPRE_IJ_MV_H
#define _HYPRE_IJ_MV_H

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

struct hypre_IJMatrix_struct;
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;
struct hypre_IJVector_struct;
typedef struct hypre_IJVector_struct *HYPRE_IJVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/


/* HYPRE_IJMatrix.c */
int HYPRE_IJMatrixCreate( MPI_Comm comm , HYPRE_IJMatrix *in_matrix_ptr , int global_m , int global_n );
int HYPRE_IJMatrixDestroy( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixInitialize( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixAssemble( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixDistribute( HYPRE_IJMatrix IJmatrix , const int *row_starts , const int *col_starts );
int HYPRE_IJMatrixSetLocalStorageType( HYPRE_IJMatrix IJmatrix , int type );
int HYPRE_IJMatrixSetLocalSize( HYPRE_IJMatrix IJmatrix , int local_m , int local_n );
int HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );
int HYPRE_IJMatrixSetDiagRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );
int HYPRE_IJMatrixSetOffDiagRowSizes( HYPRE_IJMatrix IJmatrix , const int *sizes );
int HYPRE_IJMatrixQueryInsertionSemantics( HYPRE_IJMatrix IJmatrix , int *level );
int HYPRE_IJMatrixInsertBlock( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixAddToBlock( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixInsertRow( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixAddToRow( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixAddToRowAfter( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixSetValues( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values );
int HYPRE_IJMatrixSetBlockValues( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int HYPRE_IJMatrixAddToBlockValues( HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values );
int hypre_RefIJMatrix( HYPRE_IJMatrix IJmatrix , HYPRE_IJMatrix *reference );
void *HYPRE_IJMatrixGetLocalStorage( HYPRE_IJMatrix IJmatrix );
int HYPRE_IJMatrixGetRowPartitioning( HYPRE_IJMatrix IJmatrix , const int **row_partitioning );
int HYPRE_IJMatrixGetColPartitioning( HYPRE_IJMatrix IJmatrix , const int **col_partitioning );

/* HYPRE_IJVector.c */
int HYPRE_IJVectorCreate( MPI_Comm comm , HYPRE_IJVector *in_vector_ptr , int global_n );
int HYPRE_IJVectorDestroy( HYPRE_IJVector IJvector );
int HYPRE_IJVectorSetPartitioning( HYPRE_IJVector IJvector , const int *partitioning );
int HYPRE_IJVectorSetLocalPartitioning( HYPRE_IJVector IJvector , int vec_start_this_proc , int vec_start_next_proc );
int HYPRE_IJVectorInitialize( HYPRE_IJVector IJvector );
int HYPRE_IJVectorDistribute( HYPRE_IJVector IJvector , const int *vec_starts );
int HYPRE_IJVectorSetLocalStorageType( HYPRE_IJVector IJvector , int type );
int HYPRE_IJVectorZeroLocalComponents( HYPRE_IJVector IJvector );
int HYPRE_IJVectorSetLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );
int HYPRE_IJVectorSetLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );
int HYPRE_IJVectorAddToLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values );
int HYPRE_IJVectorAddToLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values );
int HYPRE_IJVectorAssemble( HYPRE_IJVector IJvector );
int HYPRE_IJVectorGetLocalComponents( HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , double *values );
int HYPRE_IJVectorGetLocalComponentsInBlock( HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , double *values );
int HYPRE_IJVectorGetLocalStorageType( HYPRE_IJVector IJvector , int *type );
void *HYPRE_IJVectorGetLocalStorage( HYPRE_IJVector IJvector );
int hypre_RefIJVector( HYPRE_IJVector IJvector , HYPRE_IJVector *reference );

#ifdef __cplusplus
}
#endif

#endif
