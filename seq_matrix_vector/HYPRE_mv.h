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
 * Header file for HYPRE_mv library
 *
 *****************************************************************************/

#ifndef HYPRE_MV_HEADER
#define HYPRE_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *HYPRE_CSRMatrix;
typedef void *HYPRE_MappedMatrix;
typedef void *HYPRE_MultiblockMatrix;
typedef void *HYPRE_Vector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CreateCSRMatrix P((int num_rows , int num_cols , int *row_sizes ));
int HYPRE_DestroyCSRMatrix P((HYPRE_CSRMatrix matrix ));
int HYPRE_InitializeCSRMatrix P((HYPRE_CSRMatrix matrix ));
void HYPRE_PrintCSRMatrix P((HYPRE_CSRMatrix matrix , char *file_name ));
 
/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_NewMappedMatrix P((void ));
int HYPRE_FreeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_LimitedFreeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_InitializeMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_AssembleMappedMatrix P((HYPRE_MappedMatrix matrix ));
void HYPRE_PrintMappedMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_GetMappedMatrixColIndex P((HYPRE_MappedMatrix matrix , int j ));
void *HYPRE_GetMappedMatrixMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_SetMappedMatrixMatrix P((HYPRE_MappedMatrix matrix , void *matrix_data ));
int HYPRE_SetMappedMatrixColMap P((HYPRE_MappedMatrix matrix , int (*ColMap )(int ,void *)));
int HYPRE_SetMappedMatrixMapData P((HYPRE_MappedMatrix matrix , void *MapData ));
 
/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_NewMultiblockMatrix P((void ));
int HYPRE_FreeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_LimitedFreeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_InitializeMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_AssembleMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
void HYPRE_PrintMultiblockMatrix P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_SetMultiblockMatrixNumSubmatrices P((HYPRE_MultiblockMatrix matrix , int n ));
int HYPRE_SetMultiblockMatrixSubmatrixType P((HYPRE_MultiblockMatrix matrix , int j , int type ));
 
/* HYPRE_vector.c */
HYPRE_Vector HYPRE_CreateVector P((int size ));
int HYPRE_DestroyVector P((HYPRE_Vector vector ));
int HYPRE_InitializeVector P((HYPRE_Vector vector ));
int HYPRE_PrintVector P((HYPRE_Vector vector , char *file_name ));
 
#undef P

#ifdef __cplusplus
}
#endif

#endif
