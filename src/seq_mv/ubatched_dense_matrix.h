/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_UBATCHED_DENSE_MATRIX_HEADER
#define hypre_UBATCHED_DENSE_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Uniform batched dense matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_UBatchedDenseMatrix_struct
{
   HYPRE_Int             row_major;          /* Flag indicating storage format (false: col major)*/
   HYPRE_Int             num_rows_total;     /* Number of rows of entire matrix */
   HYPRE_Int             num_cols_total;     /* Number of columns of entire matrix */
   HYPRE_Int             num_coefs_total;    /* Number of coefficients of entire matrix */

   /* Local info for a individual batch (sub-matrix) */
   HYPRE_Int             num_batches;        /* Number of sub-matrices (batch size) */
   HYPRE_Int             num_rows;           /* Number of rows per batch */
   HYPRE_Int             num_cols;           /* Number of columns per batch */
   HYPRE_Int             num_coefs;          /* Number of coefficients per batch */

   /* Matrix coefficients array */
   HYPRE_Int             owns_data;          /* Flag indicating ownership of the data array */
   HYPRE_Complex        *data;               /* Matrix coefficients */
   HYPRE_Complex       **data_aop;           /* Array of pointers to data */
   HYPRE_MemoryLocation  memory_location;    /* Memory location of data array */
} hypre_UBatchedDenseMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the uniform batched matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_UBatchedDenseMatrixRowMajor(matrix)        ((matrix) -> row_major)
#define hypre_UBatchedDenseMatrixNumRowsTotal(matrix)    ((matrix) -> num_rows_total)
#define hypre_UBatchedDenseMatrixNumColsTotal(matrix)    ((matrix) -> num_cols_total)
#define hypre_UBatchedDenseMatrixNumCoefsTotal(matrix)   ((matrix) -> num_coefs_total)

#define hypre_UBatchedDenseMatrixNumBatches(matrix)      ((matrix) -> num_batches)
#define hypre_UBatchedDenseMatrixNumRows(matrix)         ((matrix) -> num_rows)
#define hypre_UBatchedDenseMatrixNumCols(matrix)         ((matrix) -> num_cols)
#define hypre_UBatchedDenseMatrixNumCoefs(matrix)        ((matrix) -> num_coefs)

#define hypre_UBatchedDenseMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_UBatchedDenseMatrixData(matrix)            ((matrix) -> data)
#define hypre_UBatchedDenseMatrixDataAOP(matrix)         ((matrix) -> data_aop)
#define hypre_UBatchedDenseMatrixMemoryLocation(matrix)  ((matrix) -> memory_location)

#endif
