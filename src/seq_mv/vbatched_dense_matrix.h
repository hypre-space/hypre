/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_VBATCHED_DENSE_MATRIX_HEADER
#define hypre_VBATCHED_DENSE_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Variable batched dense matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_VBatchedDenseMatrix_struct
{
   HYPRE_Int             row_major;          /* Flag indicating storage format (false: col major)*/
   HYPRE_Int             num_rows_total;     /* Number of rows of entire matrix */
   HYPRE_Int             num_cols_total;     /* Number of columns of entire matrix */
   HYPRE_Int             num_coefs_total;    /* Number of coefficients of entire matrix */

   /* Local info for each batch (sub-matrix) components */
   HYPRE_Int             num_batches;        /* Number of sub-matrices (batch size) */
   HYPRE_Int            *row_offsets;        /* Row offsets array */
   HYPRE_Int            *col_offsets;        /* Column offsets array */
   HYPRE_Int            *coef_offsets;       /* Coefficient offsets array */

   /* Matrix coefficients array */
   HYPRE_Int             owns_data;          /* Flag indicating ownership of the data array */
   HYPRE_Complex        *data;               /* Matrix coefficients */
   HYPRE_Complex       **data_aop;           /* Array of pointers to data */
   HYPRE_MemoryLocation  memory_location;    /* Memory location of data array */
} hypre_VBatchedDenseMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the uniform batched matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_VBatchedDenseMatrixRowMajor(matrix)        ((matrix) -> row_major)
#define hypre_VBatchedDenseMatrixNumRowsTotal(matrix)    ((matrix) -> num_rows_total)
#define hypre_VBatchedDenseMatrixNumColsTotal(matrix)    ((matrix) -> num_cols_total)
#define hypre_VBatchedDenseMatrixNumCoefsTotal(matrix)   ((matrix) -> num_coefs_total)

#define hypre_VBatchedDenseMatrixNumBatches(matrix)      ((matrix) -> num_batches)
#define hypre_VBatchedDenseMatrixRowOffsets(matrix)      ((matrix) -> row_offsets)
#define hypre_VBatchedDenseMatrixColOffsets(matrix)      ((matrix) -> col_offsets)
#define hypre_VBatchedDenseMatrixCoefOffsets(matrix)     ((matrix) -> coef_offsets)

#define hypre_VBatchedDenseMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_VBatchedDenseMatrixData(matrix)            ((matrix) -> data)
#define hypre_VBatchedDenseMatrixDataAOP(matrix)         ((matrix) -> data_aop)
#define hypre_VBatchedDenseMatrixMemoryLocation(matrix)  ((matrix) -> memory_location)

#endif
