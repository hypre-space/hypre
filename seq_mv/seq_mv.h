
#include <HYPRE_config.h>

#include "HYPRE_seq_mv.h"

#ifndef hypre_MV_HEADER
#define hypre_MV_HEADER

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
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int     *i;
   int     *j;
   int      num_rows;
   int      num_cols;
   int      num_nonzeros;

   /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   int      owns_data;

} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)         ((matrix) -> data)
#define hypre_CSRMatrixI(matrix)            ((matrix) -> i)
#define hypre_CSRMatrixJ(matrix)            ((matrix) -> j)
#define hypre_CSRMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define hypre_CSRMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define hypre_CSRMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define hypre_CSRMatrixOwnsData(matrix)     ((matrix) -> owns_data)



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

} hypre_CSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBooleanMatrix_Get_I(matrix)        ((matrix)->i)
#define hypre_CSRBooleanMatrix_Get_J(matrix)        ((matrix)->j)
#define hypre_CSRBooleanMatrix_Get_NRows(matrix)    ((matrix)->num_rows)
#define hypre_CSRBooleanMatrix_Get_NCols(matrix)    ((matrix)->num_cols)
#define hypre_CSRBooleanMatrix_Get_NNZ(matrix)      ((matrix)->num_nonzeros)
#define hypre_CSRBooleanMatrix_Get_OwnsData(matrix) ((matrix)->owns_data)

#endif

/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Mapped Matrix data structures
 *
 *****************************************************************************/

#ifndef hypre_MAPPED_MATRIX_HEADER
#define hypre_MAPPED_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Mapped Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   void               *matrix;
   int               (*ColMap)(int, void *);
   void               *MapData;

} hypre_MappedMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Mapped Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_MappedMatrixMatrix(matrix)           ((matrix) -> matrix)
#define hypre_MappedMatrixColMap(matrix)           ((matrix) -> ColMap)
#define hypre_MappedMatrixMapData(matrix)           ((matrix) -> MapData)

#define hypre_MappedMatrixColIndex(matrix,j) \
         (hypre_MappedMatrixColMap(matrix)(j,hypre_MappedMatrixMapData(matrix)))

#endif
/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for Multiblock Matrix data structures
 *
 *****************************************************************************/

#ifndef hypre_MULTIBLOCK_MATRIX_HEADER
#define hypre_MULTIBLOCK_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Multiblock Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   int                  num_submatrices;
   int                 *submatrix_types;
   void                **submatrices;

} hypre_MultiblockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Multiblock Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_MultiblockMatrixSubmatrices(matrix)        ((matrix) -> submatrices)
#define hypre_MultiblockMatrixNumSubmatrices(matrix)     ((matrix) -> num_submatrices)
#define hypre_MultiblockMatrixSubmatrixTypes(matrix)     ((matrix) -> submatrix_types)

#define hypre_MultiblockMatrixSubmatrix(matrix,j) (hypre_MultiblockMatrixSubmatrices\
(matrix)[j])
#define hypre_MultiblockMatrixSubmatrixType(matrix,j) (hypre_MultiblockMatrixSubmatrixTypes\
(matrix)[j])

#endif
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
 * Header info for Vector data structure
 *
 *****************************************************************************/

#ifndef hypre_VECTOR_HEADER
#define hypre_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * hypre_Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int      size;

   /* Does the Vector create/destroy `data'? */
   int      owns_data;

} hypre_Vector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorData(vector)      ((vector) -> data)
#define hypre_VectorSize(vector)      ((vector) -> size)
#define hypre_VectorOwnsData(vector)  ((vector) -> owns_data)

#endif

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

/* csr_matop.c */
hypre_CSRMatrix *hypre_CSRMatrixAdd( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiply( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros( hypre_CSRMatrix *A , double tol );

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CSRMatrixCreate( int num_rows , int num_cols , int num_nonzeros );
int hypre_CSRMatrixDestroy( hypre_CSRMatrix *matrix );
int hypre_CSRMatrixInitialize( hypre_CSRMatrix *matrix );
int hypre_CSRMatrixSetDataOwner( hypre_CSRMatrix *matrix , int owns_data );
hypre_CSRMatrix *hypre_CSRMatrixRead( char *file_name );
int hypre_CSRMatrixPrint( hypre_CSRMatrix *matrix , char *file_name );
int hypre_CSRMatrixCopy( hypre_CSRMatrix *A , hypre_CSRMatrix *B , int copy_data );

/* csr_matvec.c */
int hypre_CSRMatrixMatvec( double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y );
int hypre_CSRMatrixMatvecT( double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y );

/* genpart.c */
int hypre_GeneratePartitioning( int length , int num_procs , int **part_ptr );

/* mapped_matrix.c */
hypre_MappedMatrix *hypre_MappedMatrixCreate( void );
int hypre_MappedMatrixDestroy( hypre_MappedMatrix *matrix );
int hypre_MappedMatrixLimitedDestroy( hypre_MappedMatrix *matrix );
int hypre_MappedMatrixInitialize( hypre_MappedMatrix *matrix );
int hypre_MappedMatrixAssemble( hypre_MappedMatrix *matrix );
void hypre_MappedMatrixPrint( hypre_MappedMatrix *matrix );
int hypre_MappedMatrixGetColIndex( hypre_MappedMatrix *matrix , int j );
void *hypre_MappedMatrixGetMatrix( hypre_MappedMatrix *matrix );
int hypre_MappedMatrixSetMatrix( hypre_MappedMatrix *matrix , void *matrix_data );
int hypre_MappedMatrixSetColMap( hypre_MappedMatrix *matrix , int (*ColMap )(int ,void *));
int hypre_MappedMatrixSetMapData( hypre_MappedMatrix *matrix , void *map_data );

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate( void );
int hypre_MultiblockMatrixDestroy( hypre_MultiblockMatrix *matrix );
int hypre_MultiblockMatrixLimitedDestroy( hypre_MultiblockMatrix *matrix );
int hypre_MultiblockMatrixInitialize( hypre_MultiblockMatrix *matrix );
int hypre_MultiblockMatrixAssemble( hypre_MultiblockMatrix *matrix );
void hypre_MultiblockMatrixPrint( hypre_MultiblockMatrix *matrix );
int hypre_MultiblockMatrixSetNumSubmatrices( hypre_MultiblockMatrix *matrix , int n );
int hypre_MultiblockMatrixSetSubmatrixType( hypre_MultiblockMatrix *matrix , int j , int type );
int hypre_MultiblockMatrixSetSubmatrix( hypre_MultiblockMatrix *matrix , int j , void *submatrix );

/* vector.c */
hypre_Vector *hypre_SeqVectorCreate( int size );
int hypre_SeqVectorDestroy( hypre_Vector *vector );
int hypre_SeqVectorInitialize( hypre_Vector *vector );
int hypre_SeqVectorSetDataOwner( hypre_Vector *vector , int owns_data );
hypre_Vector *hypre_SeqVectorRead( char *file_name );
int hypre_SeqVectorPrint( hypre_Vector *vector , char *file_name );
int hypre_SeqVectorSetConstantValues( hypre_Vector *v , double value );
int hypre_SeqVectorSetRandomValues( hypre_Vector *v , int seed );
int hypre_SeqVectorCopy( hypre_Vector *x , hypre_Vector *y );
int hypre_SeqVectorScale( double alpha , hypre_Vector *y );
int hypre_SeqVectorAxpy( double alpha , hypre_Vector *x , hypre_Vector *y );
double hypre_SeqVectorInnerProd( hypre_Vector *x , hypre_Vector *y );


#ifdef __cplusplus
}
#endif

#endif

