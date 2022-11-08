/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_DenseMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_DenseMatrix *
hypre_DenseMatrixCreate( HYPRE_Int num_rows,
                         HYPRE_Int num_cols )
{
   hypre_DenseMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_DenseMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_DenseMatrixNumRows(matrix)        = num_rows;
   hypre_DenseMatrixNumCols(matrix)        = num_cols;
   hypre_DenseMatrixSize(matrix)           = num_rows * num_cols;
   hypre_DenseMatrixData(matrix)           = NULL;
   hypre_DenseMatrixDataAOP(matrix)        = NULL;
   hypre_DenseMatrixMemoryLocation(matrix) = hypre_HandleMemoryLocation(hypre_handle());

   /* set defaults */
   hypre_DenseMatrixRowMajor(matrix)       = 1;
   hypre_DenseMatrixOwnsData(matrix)       = 0;
   hypre_DenseMatrixNumBatches(matrix)     = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseMatrixDestroy( hypre_DenseMatrix *matrix )
{
   if (matrix)
   {
      HYPRE_MemoryLocation memory_location = hypre_DenseMatrixMemoryLocation(matrix);

      if (hypre_DenseMatrixOwnsData(matrix))
      {
         hypre_TFree(hypre_DenseMatrixData(matrix), memory_location);
      }

      /* data_aop is always owned by a hypre_DenseMatrix */
      hypre_TFree(hypre_DenseMatrixDataAOP(matrix), memory_location);

      /* Free matrix pointer */
      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixInitialize_v2
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseMatrixInitialize_v2( hypre_DenseMatrix    *matrix,
                                HYPRE_MemoryLocation *memory_location )
{
   HYPRE_Int   size        = hypre_DenseMatrixSize(matrix);
   HYPRE_Int   num_batches = hypre_DenseMatrixNumBatches(matrix);

   hypre_DenseMatrixMemoryLocation(matrix) = memory_location;

   /* Allocate memory for data */
   if (!hypre_DenseMatrixData(matrix) && size)
   {
      hypre_DenseMatrixData(matrix) = hypre_CTAlloc(HYPRE_Complex, size, memory_location);

      if (num_batches > 1)
      {
         hypre_DenseMatrixDataAOP(matrix) = hypre_TAlloc(HYPRE_Complex,
                                                         num_batches,
                                                         memory_location);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseMatrixInitialize( hypre_DenseMatrix *matrix )
{
   return hypre_DenseMatrixInitialize(matrix, hypre_DenseMatrixMemoryLocation(matrix));
}
