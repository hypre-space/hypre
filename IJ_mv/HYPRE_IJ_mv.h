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
 * Header file for HYPRE_IJMatrix library
 *
 *****************************************************************************/

#ifndef HYPRE_IJ_MV_HEADER
#define HYPRE_IJ_MV_HEADER

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
void * hypre_GetIJMatrixLocalStorage P((HYPRE_IJMatrix IJMatrix ));

#undef P

#ifdef __cplusplus
}
#endif

#endif
