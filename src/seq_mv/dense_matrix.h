/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_DENSE_MATRIX_HEADER
#define hypre_DENSE_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Dense Matrix data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int             row_major;
   HYPRE_Int             num_rows;
   HYPRE_Int             num_cols;
   HYPRE_Int             num_batches;
   HYPRE_Int             size;
   HYPRE_Int             owns_data;
   HYPRE_Complex        *data;
   HYPRE_Complex       **data_aop;
   HYPRE_MemoryLocation  memory_location; /* memory location of data array */
} hypre_DenseMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Dense Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_DenseMatrixRowMajor(matrix)             ((matrix) -> row_major)
#define hypre_DenseMatrixNumRows(matrix)              ((matrix) -> num_rows)
#define hypre_DenseMatrixNumCols(matrix)              ((matrix) -> num_cols)
#define hypre_DenseMatrixNumBatches(matrix)           ((matrix) -> num_batches)
#define hypre_DenseMatrixSize(matrix)                 ((matrix) -> size)
#define hypre_DenseMatrixOwnsData(matrix)             ((matrix) -> owns_data)
#define hypre_DenseMatrixData(matrix)                 ((matrix) -> data)
#define hypre_DenseMatrixDataAOP(matrix)              ((matrix) -> data_aop)
#define hypre_DenseMatrixMemoryLocation(matrix)       ((matrix) -> memory_location)

#endif
