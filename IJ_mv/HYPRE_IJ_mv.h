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
 * Macro parameters
 *--------------------------------------------------------------------------*/

#define HYPRE_PARCSR  1
#define HYPRE_PETSC   2
#define HYPRE_ISIS    3

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

# define        P(s) s

/* HYPRE_IJMatrix.c */
int HYPRE_NewIJMatrix P((MPI_Comm comm, HYPRE_IJMatrix *in_matrix_ptr , int global_m , int global_n ));
int HYPRE_FreeIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_InitializeIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_AssembleIJMatrix P((HYPRE_IJMatrix IJmatrix ));
int HYPRE_DistributeIJMatrix P((HYPRE_IJMatrix IJmatrix , int *row_starts , int *col_starts ));
int HYPRE_SetIJMatrixLocalStorageType P((HYPRE_IJMatrix IJmatrix , int type ));
int HYPRE_SetIJMatrixLocalSize P((HYPRE_IJMatrix IJmatrix , int local_m , int local_n ));
int HYPRE_SetIJMatrixDiagRowSizes P((HYPRE_IJMatrix IJmatrix , int *sizes ));
int HYPRE_SetIJMatrixOffDiagRowSizes P((HYPRE_IJMatrix IJmatrix , int *sizes ));
int HYPRE_SetIJMatrixTotalSize P((HYPRE_IJMatrix IJmatrix , int size ));
int HYPRE_QueryIJMatrixInsertionSemantics P((HYPRE_IJMatrix IJmatrix , int *level ));
int HYPRE_InsertIJMatrixBlock P((HYPRE_IJMatrix IJmatrix , int m , int n , int *rows , int *cols , double *values ));
int HYPRE_AddBlockToIJMatrix P((HYPRE_IJMatrix IJmatrix , int m , int n , int *rows , int *cols , double *values ));
int HYPRE_InsertIJMatrixRow P((HYPRE_IJMatrix IJmatrix , int n , int row , int *cols , double *values ));
int hypre_RefIJMatrix P((HYPRE_IJMatrix IJmatrix , HYPRE_IJMatrix *reference ));
void *HYPRE_GetIJMatrixLocalStorage P((HYPRE_IJMatrix IJmatrix ));

/* HYPRE_IJVector.c */
int HYPRE_NewIJVector P((MPI_Comm comm, HYPRE_IJVector *in_vector_ptr , int global_n ));
int HYPRE_FreeIJVector P((HYPRE_IJVector IJvector ));
int HYPRE_InitializeIJVector P((HYPRE_IJVector IJvector ));
int HYPRE_DistributeIJVector P((HYPRE_IJVector IJvector , int *vec_starts ));
int HYPRE_SetIJVectorLocalStorageType P((HYPRE_IJVector IJvector , int type ));

int HYPRE_ZeroIJVectorLocalComponents P((HYPRE_IJVector IJvector ));
int HYPRE_SetIJVectorLocalComponents P((HYPRE_IJVector IJvector , int num_values, int *glob_vec_indices , int *value_indices, double *values ));
int HYPRE_SetIJVectorLocalComponentsInBlock P((HYPRE_IJVector IJvector , int globvec_index_start , int glob_vec_index_stop , int *glob_vec_indices, double *values ));

int HYPRE_AddToIJVectorLocalComponents P((HYPRE_IJVector IJvector , int num_values , int *glob_vec_indices, int *value_indices, double *values ));
int HYPRE_AddToIJVectorLocalComponentsInBlock P((HYPRE_IJVector IJvector , int glob_vec_index_start , int glob_vec_index_stop , int *value_indices, double *values ));

int HYPRE_GetIJVectorLocalComponents P((HYPRE_IJVector IJvector , int num_values, int *glob_vec_indices, int *value_indices, double *values )); 
int HYPRE_GetIJVectorLocalComponentsInBlock P((HYPRE_IJVector IJvector , int global_vec_index_start, int glob_vec_index_stop, int *value_indices, double *values )); 

int hypre_RefIJVector P((HYPRE_IJVector IJvector , HYPRE_IJVector *reference ));
void *HYPRE_GetIJVectorLocalStorage P((HYPRE_IJVector IJvector ));

#undef P

#ifdef __cplusplus
}
#endif

#endif
