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

typedef struct {int opaque;} *HYPRE_IJMatrix;
typedef struct {int opaque;} *HYPRE_IJVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

# define        P(s) s

/* HYPRE_IJMatrix.c */
int HYPRE_IJMatrixCreate P((MPI_Comm comm , HYPRE_IJMatrix *in_matrix_ptr , int global_m , int global_n ));
int HYPRE_IJMatrixDestroy P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixInitialize P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixAssemble P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixDistribute P((HYPRE_IJMatrix IJmatrix , const int *row_starts , const int *col_starts ));
int HYPRE_IJMatrixSetLocalStorageType P((HYPRE_IJMatrix IJmatrix , int type ));
int HYPRE_IJMatrixSetLocalSize P((HYPRE_IJMatrix IJmatrix , int local_m , int local_n ));
int HYPRE_IJMatrixSetRowSizes P((HYPRE_IJMatrix IJmatrix , const int *sizes ));
int HYPRE_IJMatrixSetDiagRowSizes P((HYPRE_IJMatrix IJmatrix , const int *sizes ));
int HYPRE_IJMatrixSetOffDiagRowSizes P((HYPRE_IJMatrix IJmatrix , const int *sizes ));
int HYPRE_IJMatrixQueryInsertionSemantics P((HYPRE_IJMatrix IJmatrix , int *level ));
int HYPRE_IJMatrixInsertBlock P((HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values ));
int HYPRE_IJMatrixAddBlock P((HYPRE_IJMatrix IJmatrix , int m , int n , const int *rows , const int *cols , const double *values ));
int HYPRE_IJMatrixInsertRow P((HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values ));
int HYPRE_IJMatrixAddRow P((HYPRE_IJMatrix IJmatrix , int n , int row , const int *cols , const double *values ));
int hypre_RefIJMatrix P((HYPRE_IJMatrix IJmatrix , HYPRE_IJMatrix *reference ));
void *HYPRE_IJMatrixGetLocalStorage P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_IJMatrixGetRowPartitioning P((HYPRE_IJMatrix IJmatrix , const int **row_partitioning ));
int HYPRE_IJMatrixGetColPartitioning P((HYPRE_IJMatrix IJmatrix , const int **col_partitioning ));

/* HYPRE_IJVector.c */
int HYPRE_IJVectorCreate P((MPI_Comm comm, HYPRE_IJVector *in_vector_ptr , int global_n ));
int HYPRE_IJVectorDestroy P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorSetPartitioning P((HYPRE_IJVector IJvector , const int *partitioning ));
int HYPRE_IJVectorSetLocalPartitioning P((HYPRE_IJVector IJvector , int vec_start_this_proc, int vec_start_next_proc ));
int HYPRE_IJVectorInitialize P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorDistribute P((HYPRE_IJVector IJvector , const int *vec_starts ));
int HYPRE_IJVectorSetLocalStorageType P((HYPRE_IJVector IJvector , int type ));

int HYPRE_IJVectorZeroLocalComponents P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorSetLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int HYPRE_IJVectorSetLocalComponentsInBlock P((HYPRE_IJVector IJvector , int globvec_index_start , int glob_vec_index_stop , const int *glob_vec_indices , const double *values ));

int HYPRE_IJVectorAddLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices , const double *values ));
int HYPRE_IJVectorAddLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , const int *value_indices , const double *values ));

int HYPRE_IJVectorAssemble P((HYPRE_IJVector IJvector ));
int HYPRE_IJVectorGetLocalComponents P((HYPRE_IJVector IJvector , int num_values , const int *glob_vec_indices , const int *value_indices, double *values )); 
int HYPRE_IJVectorGetLocalComponentsInBlock P((HYPRE_IJVector IJvector , int global_vec_index_start, int glob_vec_index_stop , const int *value_indices , double *values )); 

int hypre_RefIJVector P((HYPRE_IJVector IJvector , HYPRE_IJVector *reference ));
void *HYPRE_IJVectorGetLocalStorage P((HYPRE_IJVector IJvector ));

#undef P

#ifdef __cplusplus
}
#endif

#endif
