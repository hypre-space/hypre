/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_UBATCHED_CSR_MATRIX_HEADER
#define hypre_UBATCHED_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Uniform batched CSR matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_UBatchedCSRMatrix_struct
{
   HYPRE_Int             num_rows_total;     /* Number of rows of entire matrix */
   HYPRE_Int             num_cols_total;     /* Number of columns of entire matrix */
   HYPRE_Int             num_coefs_total;    /* Number of coefficients of entire matrix */

   /* Local info for a individual batch (sub-matrix) */
   HYPRE_Int             num_batches;        /* Number of sub-matrices (batch size) */
   HYPRE_Int             num_rows;           /* Number of rows per batch */
   HYPRE_Int             num_cols;           /* Number of columns per batch */
   HYPRE_Int             num_coefs;          /* Number of coefficients per batch */
   HYPRE_Int            *i;                  /* row pointer per batch */
   HYPRE_Int            *j;                  /* column indices per batch */

   /* Matrix coefficients array */
   HYPRE_Int             owns_data;          /* Flag indicating ownership of the data array */
   HYPRE_Complex        *data;               /* Matrix coefficients */
   HYPRE_Complex       **data_aop;           /* Array of pointers to data */
   HYPRE_MemoryLocation  memory_location;    /* Memory location of data array */
} hypre_UBatchedCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the uniform batched CSR matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_UBatchedCSRMatrixNumRowsTotal(matrix)     ((matrix) -> num_rows_total)
#define hypre_UBatchedCSRMatrixNumColsTotal(matrix)     ((matrix) -> num_cols_total)
#define hypre_UBatchedCSRMatrixNumCoefsTotal(matrix)    ((matrix) -> num_coefs_total)

#define hypre_UBatchedCSRMatrixNumBatches(matrix)       ((matrix) -> num_batches)
#define hypre_UBatchedCSRMatrixNumRows(matrix)          ((matrix) -> num_rows)
#define hypre_UBatchedCSRMatrixNumCols(matrix)          ((matrix) -> num_cols)
#define hypre_UBatchedCSRMatrixNumCoefs(matrix)         ((matrix) -> num_coefs)
#define hypre_UBatchedCSRMatrixI(matrix)                ((matrix) -> i)
#define hypre_UBatchedCSRMatrixJ(matrix)                ((matrix) -> j)

#define hypre_UBatchedCSRMatrixOwnsData(matrix)         ((matrix) -> owns_data)
#define hypre_UBatchedCSRMatrixData(matrix)             ((matrix) -> data)
#define hypre_UBatchedCSRMatrixDataAOP(matrix)          ((matrix) -> data_aop)
#define hypre_UBatchedCSRMatrixMemoryLocation(matrix)   ((matrix) -> memory_location)

#endif
