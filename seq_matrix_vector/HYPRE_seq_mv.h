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

typedef struct {int opaque;} *HYPRE_CSRMatrix;
typedef struct {int opaque;} *HYPRE_MappedMatrix;
typedef struct {int opaque;} *HYPRE_MultiblockMatrix;
typedef struct {int opaque;} *HYPRE_Vector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

# define        P(s) s
 
/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate P((int num_rows , int num_cols , int *row_sizes ));
int HYPRE_CSRMatrixDestroy P((HYPRE_CSRMatrix matrix ));
int HYPRE_CSRMatrixInitialize P((HYPRE_CSRMatrix matrix ));
HYPRE_CSRMatrix HYPRE_CSRMatrixRead P((char *file_name ));
void HYPRE_CSRMatrixPrint P((HYPRE_CSRMatrix matrix , char *file_name ));
 
/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate P((void ));
int HYPRE_MappedMatrixDestroy P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixLimitedDestroy P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixInitialize P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixAssemble P((HYPRE_MappedMatrix matrix ));
void HYPRE_MappedMatrixPrint P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixGetColIndex P((HYPRE_MappedMatrix matrix , int j ));
void *HYPRE_MappedMatrixGetMatrix P((HYPRE_MappedMatrix matrix ));
int HYPRE_MappedMatrixSetMatrix P((HYPRE_MappedMatrix matrix , void *matrix_data ));
int HYPRE_MappedMatrixSetColMap P((HYPRE_MappedMatrix matrix , int (*ColMap )(int ,void *)));
int HYPRE_MappedMatrixSetMapData P((HYPRE_MappedMatrix matrix , void *MapData ));
 
/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate P((void ));
int HYPRE_MultiblockMatrixDestroy P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixLimitedDestroy P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixInitialize P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixAssemble P((HYPRE_MultiblockMatrix matrix ));
void HYPRE_MultiblockMatrixPrint P((HYPRE_MultiblockMatrix matrix ));
int HYPRE_MultiblockMatrixSetNumSubmatrices P((HYPRE_MultiblockMatrix matrix , int n ));
int HYPRE_MultiblockMatrixSetSubmatrixType P((HYPRE_MultiblockMatrix matrix , int j , int type ));
 
/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate P((int size ));
int HYPRE_VectorDestroy P((HYPRE_Vector vector ));
int HYPRE_VectorInitialize P((HYPRE_Vector vector ));
int HYPRE_VectorPrint P((HYPRE_Vector vector , char *file_name ));
HYPRE_Vector HYPRE_VectorRead P((char *file_name ));
 
#undef P

#ifdef __cplusplus
}

#endif

#endif
