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

#ifndef HYPRE_SEQ_MV_HEADER
#define HYPRE_SEQ_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

struct hypre_CSRMatrix_struct;
typedef struct hypre_CSRMatrix_struct *HYPRE_CSRMatrix;
struct hypre_MappedMatrix_struct;
typedef struct hypre_MappedMatrix_struct *HYPRE_MappedMatrix;
struct hypre_MultiblockMatrix_struct;
typedef struct hypre_MultiblockMatrix_struct *HYPRE_MultiblockMatrix;
#ifndef HYPRE_VECTOR_STRUCT
#define HYPRE_VECTOR_STRUCT
struct hypre_Vector_struct;
typedef struct hypre_Vector_struct *HYPRE_Vector;
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate( int num_rows , int num_cols , int *row_sizes );
int HYPRE_CSRMatrixDestroy( HYPRE_CSRMatrix matrix );
int HYPRE_CSRMatrixInitialize( HYPRE_CSRMatrix matrix );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead( char *file_name );
void HYPRE_CSRMatrixPrint( HYPRE_CSRMatrix matrix , char *file_name );

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate( void );
int HYPRE_MappedMatrixDestroy( HYPRE_MappedMatrix matrix );
int HYPRE_MappedMatrixLimitedDestroy( HYPRE_MappedMatrix matrix );
int HYPRE_MappedMatrixInitialize( HYPRE_MappedMatrix matrix );
int HYPRE_MappedMatrixAssemble( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint( HYPRE_MappedMatrix matrix );
int HYPRE_MappedMatrixGetColIndex( HYPRE_MappedMatrix matrix , int j );
void *HYPRE_MappedMatrixGetMatrix( HYPRE_MappedMatrix matrix );
int HYPRE_MappedMatrixSetMatrix( HYPRE_MappedMatrix matrix , void *matrix_data );
int HYPRE_MappedMatrixSetColMap( HYPRE_MappedMatrix matrix , int (*ColMap )(int ,void *));
int HYPRE_MappedMatrixSetMapData( HYPRE_MappedMatrix matrix , void *MapData );

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate( void );
int HYPRE_MultiblockMatrixDestroy( HYPRE_MultiblockMatrix matrix );
int HYPRE_MultiblockMatrixLimitedDestroy( HYPRE_MultiblockMatrix matrix );
int HYPRE_MultiblockMatrixInitialize( HYPRE_MultiblockMatrix matrix );
int HYPRE_MultiblockMatrixAssemble( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint( HYPRE_MultiblockMatrix matrix );
int HYPRE_MultiblockMatrixSetNumSubmatrices( HYPRE_MultiblockMatrix matrix , int n );
int HYPRE_MultiblockMatrixSetSubmatrixType( HYPRE_MultiblockMatrix matrix , int j , int type );

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate( int size );
int HYPRE_VectorDestroy( HYPRE_Vector vector );
int HYPRE_VectorInitialize( HYPRE_Vector vector );
int HYPRE_VectorPrint( HYPRE_Vector vector , char *file_name );
HYPRE_Vector HYPRE_VectorRead( char *file_name );

#ifdef __cplusplus
}

#endif

#endif
