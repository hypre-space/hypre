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

   hypre_DenseMatrixData(matrix)           = NULL;
   hypre_DenseMatrixNumRows(matrix)        = num_rows;
   hypre_DenseMatrixNumCols(matrix)        = num_cols;
   hypre_DenseMatrixMemoryLocation(matrix) = hypre_HandleMemoryLocation(hypre_handle());

   /* set defaults */
   hypre_DenseMatrixOwnsData(matrix)       = 1;

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

      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}
