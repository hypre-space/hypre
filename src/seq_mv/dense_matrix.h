/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_DENSE_MATRIX_HEADER
#define hypre_DENSE_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixType
 *--------------------------------------------------------------------------*/

typedef enum hypre_DenseMatrixType_enum
{
  HYPRE_DENSE_MATRIX_STANDARD = 0,
  HYPRE_DENSE_MATRIX_UBATCHED = 1,
  HYPRE_DENSE_MATRIX_VBATCHED = 2
} hypre_DenseMatrixType;

/*--------------------------------------------------------------------------
 * Dense Matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_DenseMatrix_struct
{
   hypre_DenseMatrixType type;               /* Dense matrix type */
   HYPRE_Int             row_major;          /* Flag indicating storage format */
   HYPRE_Int             num_rows;           /* Number of rows of entire matrix */
   HYPRE_Int             num_cols;           /* Number of columns of entire matrix */
   HYPRE_Int             num_coefs;          /* Number of coefficients of entire matrix */
   HYPRE_Int             owns_data;          /* Flag indicating ownership of the data array */
   HYPRE_Complex        *data;               /* Matrix coefficients */
   HYPRE_MemoryLocation  memory_location;    /* Memory location of data array */

   /* Uniform batched dense matrices info */
   HYPRE_Int             num_batches;        /* Number of sub-matrices (batch size) */
   HYPRE_Int             ubatch_num_rows;    /* Number of rows of a sub-matrix */
   HYPRE_Int             ubatch_num_cols;    /* Number of columns of a sub-matrix */
   HYPRE_Int             ubatch_num_coefs;   /* Number of coefficients of a sub-matrix */

   /* Variable batched dense matrices info */
   HYPRE_Int            *vbatch_num_rows;    /* Number of rows of each sub-matrix */
   HYPRE_Int            *vbatch_num_cols;    /* Number of columns of each sub-matrix */
   HYPRE_Int            *vbatch_num_coefs;   /* Number of coefficients of each sub-matrix */
   HYPRE_Int             vbatch_owns_arrays; /* Flag indicating ownership of the arrays above */
   HYPRE_Complex       **data_aop;           /* Array of pointers to data */
} hypre_DenseMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Dense Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_DenseMatrixType(matrix)                 ((matrix) -> type)
#define hypre_DenseMatrixRowMajor(matrix)             ((matrix) -> row_major)
#define hypre_DenseMatrixNumRows(matrix)              ((matrix) -> num_rows)
#define hypre_DenseMatrixNumCols(matrix)              ((matrix) -> num_cols)
#define hypre_DenseMatrixNumCoefs(matrix)             ((matrix) -> num_coefs)
#define hypre_DenseMatrixOwnsData(matrix)             ((matrix) -> owns_data)
#define hypre_DenseMatrixData(matrix)                 ((matrix) -> data)
#define hypre_DenseMatrixMemoryLocation(matrix)       ((matrix) -> memory_location)
#define hypre_DenseMatrixNumBatches(matrix)           ((matrix) -> num_batches)
#define hypre_DenseMatrixUBatchNumRows(matrix)        ((matrix) -> ubatch_num_rows)
#define hypre_DenseMatrixUBatchNumCols(matrix)        ((matrix) -> ubatch_num_cols)
#define hypre_DenseMatrixUBatchNumCoefs(matrix)       ((matrix) -> ubatch_num_coefs)
#define hypre_DenseMatrixVBatchNumRows(matrix)        ((matrix) -> vbatch_num_rows)
#define hypre_DenseMatrixVBatchNumCols(matrix)        ((matrix) -> vbatch_num_cols)
#define hypre_DenseMatrixVBatchNumCoefs(matrix)       ((matrix) -> vbatch_num_coefs)
#define hypre_DenseMatrixVBatchOwnsArrays(matrix)     ((matrix) -> vbatch_owns_arrays)
#define hypre_DenseMatrixDataAOP(matrix)              ((matrix) -> data_aop)

#endif
