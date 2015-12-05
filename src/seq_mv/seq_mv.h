/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.19 $
 ***********************************************************************EHEADER*/





#include <HYPRE_config.h>

#include "HYPRE_seq_mv.h"

#ifndef hypre_MV_HEADER
#define hypre_MV_HEADER

#include "_hypre_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif


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
   HYPRE_Int     *i;
   HYPRE_Int     *j;
   HYPRE_Int      num_rows;
   HYPRE_Int      num_cols;
   HYPRE_Int      num_nonzeros;

   /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   HYPRE_Int      owns_data;

   double  *data;

   /* for compressing rows in matrix multiplication  */
   HYPRE_Int     *rownnz;
   HYPRE_Int      num_rownnz;

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
#define hypre_CSRMatrixRownnz(matrix)       ((matrix) -> rownnz)
#define hypre_CSRMatrixNumRownnz(matrix)    ((matrix) -> num_rownnz)
#define hypre_CSRMatrixOwnsData(matrix)     ((matrix) -> owns_data)



/*--------------------------------------------------------------------------
 * CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int    *i;
   HYPRE_Int    *j;
   HYPRE_Int     num_rows;
   HYPRE_Int     num_cols;
   HYPRE_Int     num_nonzeros;
   HYPRE_Int     owns_data;

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
   HYPRE_Int               (*ColMap)(HYPRE_Int, void *);
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
   HYPRE_Int                  num_submatrices;
   HYPRE_Int                 *submatrix_types;
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
   HYPRE_Int      size;

   /* Does the Vector create/destroy `data'? */
   HYPRE_Int      owns_data;

   /* For multivectors...*/
   HYPRE_Int   num_vectors;  /* the above "size" is size of one vector */
   HYPRE_Int   multivec_storage_method;
   /* ...if 0, store colwise v0[0], v0[1], ..., v1[0], v1[1], ... v2[0]... */
   /* ...if 1, store rowwise v0[0], v1[0], ..., v0[1], v1[1], ... */
   /* With colwise storage, vj[i] = data[ j*size + i]
      With rowwise storage, vj[i] = data[ j + num_vectors*i] */
   HYPRE_Int  vecstride, idxstride;
   /* ... so vj[i] = data[ j*vecstride + i*idxstride ] regardless of row_storage.*/

} hypre_Vector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_VectorData(vector)      ((vector) -> data)
#define hypre_VectorSize(vector)      ((vector) -> size)
#define hypre_VectorOwnsData(vector)  ((vector) -> owns_data)
#define hypre_VectorNumVectors(vector) ((vector) -> num_vectors)
#define hypre_VectorMultiVecStorageMethod(vector) ((vector) -> multivec_storage_method)
#define hypre_VectorVectorStride(vector) ((vector) -> vecstride )
#define hypre_VectorIndexStride(vector) ((vector) -> idxstride )

#endif

/* csr_matop.c */
hypre_CSRMatrix *hypre_CSRMatrixAdd ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixMultiply ( hypre_CSRMatrix *A , hypre_CSRMatrix *B );
hypre_CSRMatrix *hypre_CSRMatrixDeleteZeros ( hypre_CSRMatrix *A , double tol );
HYPRE_Int hypre_CSRMatrixTranspose ( hypre_CSRMatrix *A , hypre_CSRMatrix **AT , HYPRE_Int data );
HYPRE_Int hypre_CSRMatrixReorder ( hypre_CSRMatrix *A );
double hypre_CSRMatrixSumElts ( hypre_CSRMatrix *A );

/* csr_matrix.c */
hypre_CSRMatrix *hypre_CSRMatrixCreate ( HYPRE_Int num_rows , HYPRE_Int num_cols , HYPRE_Int num_nonzeros );
HYPRE_Int hypre_CSRMatrixDestroy ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixInitialize ( hypre_CSRMatrix *matrix );
HYPRE_Int hypre_CSRMatrixSetDataOwner ( hypre_CSRMatrix *matrix , HYPRE_Int owns_data );
HYPRE_Int hypre_CSRMatrixSetRownnz ( hypre_CSRMatrix *matrix );
hypre_CSRMatrix *hypre_CSRMatrixRead ( char *file_name );
HYPRE_Int hypre_CSRMatrixPrint ( hypre_CSRMatrix *matrix , char *file_name );
HYPRE_Int hypre_CSRMatrixCopy ( hypre_CSRMatrix *A , hypre_CSRMatrix *B , HYPRE_Int copy_data );
hypre_CSRMatrix *hypre_CSRMatrixClone ( hypre_CSRMatrix *A );
hypre_CSRMatrix *hypre_CSRMatrixUnion ( hypre_CSRMatrix *A , hypre_CSRMatrix *B , HYPRE_Int *col_map_offd_A , HYPRE_Int *col_map_offd_B , HYPRE_Int **col_map_offd_C );

/* csr_matvec.c */
HYPRE_Int hypre_CSRMatrixMatvec ( double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvecT ( double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y );
HYPRE_Int hypre_CSRMatrixMatvec_FF ( double alpha , hypre_CSRMatrix *A , hypre_Vector *x , double beta , hypre_Vector *y , HYPRE_Int *CF_marker_x , HYPRE_Int *CF_marker_y , HYPRE_Int fpt );

/* genpart.c */
HYPRE_Int hypre_GeneratePartitioning ( HYPRE_Int length , HYPRE_Int num_procs , HYPRE_Int **part_ptr );
HYPRE_Int hypre_GenerateLocalPartitioning ( HYPRE_Int length , HYPRE_Int num_procs , HYPRE_Int myid , HYPRE_Int **part_ptr );

/* HYPRE_csr_matrix.c */
HYPRE_CSRMatrix HYPRE_CSRMatrixCreate ( HYPRE_Int num_rows , HYPRE_Int num_cols , HYPRE_Int *row_sizes );
HYPRE_Int HYPRE_CSRMatrixDestroy ( HYPRE_CSRMatrix matrix );
HYPRE_Int HYPRE_CSRMatrixInitialize ( HYPRE_CSRMatrix matrix );
HYPRE_CSRMatrix HYPRE_CSRMatrixRead ( char *file_name );
void HYPRE_CSRMatrixPrint ( HYPRE_CSRMatrix matrix , char *file_name );
HYPRE_Int HYPRE_CSRMatrixGetNumRows ( HYPRE_CSRMatrix matrix , HYPRE_Int *num_rows );

/* HYPRE_mapped_matrix.c */
HYPRE_MappedMatrix HYPRE_MappedMatrixCreate ( void );
HYPRE_Int HYPRE_MappedMatrixDestroy ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixLimitedDestroy ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixInitialize ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixAssemble ( HYPRE_MappedMatrix matrix );
void HYPRE_MappedMatrixPrint ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixGetColIndex ( HYPRE_MappedMatrix matrix , HYPRE_Int j );
void *HYPRE_MappedMatrixGetMatrix ( HYPRE_MappedMatrix matrix );
HYPRE_Int HYPRE_MappedMatrixSetMatrix ( HYPRE_MappedMatrix matrix , void *matrix_data );
HYPRE_Int HYPRE_MappedMatrixSetColMap ( HYPRE_MappedMatrix matrix , HYPRE_Int (*ColMap )(HYPRE_Int ,void *));
HYPRE_Int HYPRE_MappedMatrixSetMapData ( HYPRE_MappedMatrix matrix , void *MapData );

/* HYPRE_multiblock_matrix.c */
HYPRE_MultiblockMatrix HYPRE_MultiblockMatrixCreate ( void );
HYPRE_Int HYPRE_MultiblockMatrixDestroy ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixLimitedDestroy ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixInitialize ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixAssemble ( HYPRE_MultiblockMatrix matrix );
void HYPRE_MultiblockMatrixPrint ( HYPRE_MultiblockMatrix matrix );
HYPRE_Int HYPRE_MultiblockMatrixSetNumSubmatrices ( HYPRE_MultiblockMatrix matrix , HYPRE_Int n );
HYPRE_Int HYPRE_MultiblockMatrixSetSubmatrixType ( HYPRE_MultiblockMatrix matrix , HYPRE_Int j , HYPRE_Int type );

/* HYPRE_vector.c */
HYPRE_Vector HYPRE_VectorCreate ( HYPRE_Int size );
HYPRE_Int HYPRE_VectorDestroy ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorInitialize ( HYPRE_Vector vector );
HYPRE_Int HYPRE_VectorPrint ( HYPRE_Vector vector , char *file_name );
HYPRE_Vector HYPRE_VectorRead ( char *file_name );

/* mapped_matrix.c */
hypre_MappedMatrix *hypre_MappedMatrixCreate ( void );
HYPRE_Int hypre_MappedMatrixDestroy ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixLimitedDestroy ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixInitialize ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixAssemble ( hypre_MappedMatrix *matrix );
void hypre_MappedMatrixPrint ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixGetColIndex ( hypre_MappedMatrix *matrix , HYPRE_Int j );
void *hypre_MappedMatrixGetMatrix ( hypre_MappedMatrix *matrix );
HYPRE_Int hypre_MappedMatrixSetMatrix ( hypre_MappedMatrix *matrix , void *matrix_data );
HYPRE_Int hypre_MappedMatrixSetColMap ( hypre_MappedMatrix *matrix , HYPRE_Int (*ColMap )(HYPRE_Int ,void *));
HYPRE_Int hypre_MappedMatrixSetMapData ( hypre_MappedMatrix *matrix , void *map_data );

/* multiblock_matrix.c */
hypre_MultiblockMatrix *hypre_MultiblockMatrixCreate ( void );
HYPRE_Int hypre_MultiblockMatrixDestroy ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixLimitedDestroy ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixInitialize ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixAssemble ( hypre_MultiblockMatrix *matrix );
void hypre_MultiblockMatrixPrint ( hypre_MultiblockMatrix *matrix );
HYPRE_Int hypre_MultiblockMatrixSetNumSubmatrices ( hypre_MultiblockMatrix *matrix , HYPRE_Int n );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrixType ( hypre_MultiblockMatrix *matrix , HYPRE_Int j , HYPRE_Int type );
HYPRE_Int hypre_MultiblockMatrixSetSubmatrix ( hypre_MultiblockMatrix *matrix , HYPRE_Int j , void *submatrix );

/* vector.c */
hypre_Vector *hypre_SeqVectorCreate ( HYPRE_Int size );
hypre_Vector *hypre_SeqMultiVectorCreate ( HYPRE_Int size , HYPRE_Int num_vectors );
HYPRE_Int hypre_SeqVectorDestroy ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorInitialize ( hypre_Vector *vector );
HYPRE_Int hypre_SeqVectorSetDataOwner ( hypre_Vector *vector , HYPRE_Int owns_data );
hypre_Vector *hypre_SeqVectorRead ( char *file_name );
HYPRE_Int hypre_SeqVectorPrint ( hypre_Vector *vector , char *file_name );
HYPRE_Int hypre_SeqVectorSetConstantValues ( hypre_Vector *v , double value );
HYPRE_Int hypre_SeqVectorSetRandomValues ( hypre_Vector *v , HYPRE_Int seed );
HYPRE_Int hypre_SeqVectorCopy ( hypre_Vector *x , hypre_Vector *y );
hypre_Vector *hypre_SeqVectorCloneDeep ( hypre_Vector *x );
hypre_Vector *hypre_SeqVectorCloneShallow ( hypre_Vector *x );
HYPRE_Int hypre_SeqVectorScale ( double alpha , hypre_Vector *y );
HYPRE_Int hypre_SeqVectorAxpy ( double alpha , hypre_Vector *x , hypre_Vector *y );
double hypre_SeqVectorInnerProd ( hypre_Vector *x , hypre_Vector *y );
double hypre_VectorSumElts ( hypre_Vector *vector );

#ifdef __cplusplus
}
#endif

#endif

