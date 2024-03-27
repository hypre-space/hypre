/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_DENSE_BLOCK_MATRIX_HEADER
#define hypre_DENSE_BLOCK_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Uniformly blocked dense matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_DenseBlockMatrix_struct
{
   HYPRE_Int             row_major;          /* Flag indicating storage format (false: col major)*/
   HYPRE_Int             num_rows;           /* Number of rows of entire matrix */
   HYPRE_Int             num_cols;           /* Number of columns of entire matrix */
   HYPRE_Int             num_nonzeros;       /* Number of nonzero coefficients of entire matrix */
   HYPRE_Int             num_blocks;         /* Number of sub-matrices (blocks) */

   /* Local info for a individual block (sub-matrix) */
   HYPRE_Int             row_stride;         /* Stride for accessing row values */
   HYPRE_Int             col_stride;         /* Stride for accessing column values */
   HYPRE_Int             num_rows_block;     /* Number of rows per block */
   HYPRE_Int             num_cols_block;     /* Number of columns per block */
   HYPRE_Int             num_nonzeros_block; /* Number of nonzero coefficients per block */

   /* Matrix coefficients array */
   HYPRE_Int             owns_data;          /* Flag indicating ownership of the data array */
   HYPRE_Complex        *data;               /* Matrix coefficients */
   HYPRE_Complex       **data_aop;           /* Array of pointers to data */
   HYPRE_MemoryLocation  memory_location;    /* Memory location of data array */
} hypre_DenseBlockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the uniform batched matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_DenseBlockMatrixRowMajor(matrix)         ((matrix) -> row_major)
#define hypre_DenseBlockMatrixNumRows(matrix)          ((matrix) -> num_rows)
#define hypre_DenseBlockMatrixNumCols(matrix)          ((matrix) -> num_cols)
#define hypre_DenseBlockMatrixNumNonzeros(matrix)      ((matrix) -> num_nonzeros)
#define hypre_DenseBlockMatrixNumBlocks(matrix)        ((matrix) -> num_blocks)

#define hypre_DenseBlockMatrixRowStride(matrix)        ((matrix) -> row_stride)
#define hypre_DenseBlockMatrixColStride(matrix)        ((matrix) -> col_stride)
#define hypre_DenseBlockMatrixNumRowsBlock(matrix)     ((matrix) -> num_rows_block)
#define hypre_DenseBlockMatrixNumColsBlock(matrix)     ((matrix) -> num_cols_block)
#define hypre_DenseBlockMatrixNumNonzerosBlock(matrix) ((matrix) -> num_nonzeros_block)

#define hypre_DenseBlockMatrixOwnsData(matrix)         ((matrix) -> owns_data)
#define hypre_DenseBlockMatrixData(matrix)             ((matrix) -> data)
#define hypre_DenseBlockMatrixDataAOP(matrix)          ((matrix) -> data_aop)
#define hypre_DenseBlockMatrixMemoryLocation(matrix)   ((matrix) -> memory_location)
#define hypre_DenseBlockMatrixDataIJ(matrix, d, i, j) \
   (d[(matrix) -> row_stride * i + \
      (matrix) -> col_stride * j])
#define hypre_DenseBlockMatrixDataBIJ(matrix, b, i, j) \
   ((matrix) -> data[(matrix) -> num_nonzeros_block * b + \
                     (matrix) -> row_stride * i + \
                     (matrix) -> col_stride * j])

#endif
