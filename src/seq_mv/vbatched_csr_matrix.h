/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_VBATCHED_CSR_MATRIX_HEADER
#define hypre_VBATCHED_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Variable batched CSR matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_VBatchedCSRMatrix_struct
{
   HYPRE_Int             num_rows_total;     /* Number of rows of entire matrix */
   HYPRE_Int             num_cols_total;     /* Number of columns of entire matrix */
   HYPRE_Int             num_coefs_total;    /* Number of coefficients of entire matrix */

   /* Local info for a individual batch (sub-matrix) */
   HYPRE_Int             num_batches;        /* Number of sub-matrices (batch size) */
   HYPRE_Int            *row_offsets;        /* Array of row offsets per batch */
   HYPRE_Int            *col_offsets;        /* Array of column offsets per batch */
   HYPRE_Int            *coef_offsets;       /* Array of coefficient offsets per batch */
   HYPRE_Int            *i;                  /* row pointer */
   HYPRE_Int            *j;                  /* column indices */

   /* Matrix coefficients array */
   HYPRE_Int             owns_data;          /* Flag indicating ownership of the data array */
   HYPRE_Complex        *data;               /* Matrix coefficients */
   HYPRE_Complex       **data_aop;           /* Array of pointers to data */
   HYPRE_MemoryLocation  memory_location;    /* Memory location of data array */
} hypre_VBatchedCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the uniform batched CSR matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_VBatchedCSRMatrixNumRowsTotal(matrix)     ((matrix) -> num_rows_total)
#define hypre_VBatchedCSRMatrixNumColsTotal(matrix)     ((matrix) -> num_cols_total)
#define hypre_VBatchedCSRMatrixNumCoefsTotal(matrix)    ((matrix) -> num_coefs_total)

#define hypre_VBatchedCSRMatrixNumBatches(matrix)       ((matrix) -> num_batches)
#define hypre_VBatchedCSRMatrixRowOffsets(matrix)       ((matrix) -> row_offsets)
#define hypre_VBatchedCSRMatrixColOffsets(matrix)       ((matrix) -> col_offsets)
#define hypre_VBatchedCSRMatrixCoefOffsets(matrix)      ((matrix) -> coef_offsets)
#define hypre_VBatchedCSRMatrixI(matrix)                ((matrix) -> i)
#define hypre_VBatchedCSRMatrixJ(matrix)                ((matrix) -> j)

#define hypre_VBatchedCSRMatrixOwnsData(matrix)         ((matrix) -> owns_data)
#define hypre_VBatchedCSRMatrixData(matrix)             ((matrix) -> data)
#define hypre_VBatchedCSRMatrixDataAOP(matrix)          ((matrix) -> data_aop)
#define hypre_VBatchedCSRMatrixMemoryLocation(matrix)   ((matrix) -> memory_location)

#endif
