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

   hypre_DenseMatrixType(matrix)             = HYPRE_DENSE_MATRIX_STANDARD;
   hypre_DenseMatrixNumRows(matrix)          = num_rows;
   hypre_DenseMatrixNumCols(matrix)          = num_cols;
   hypre_DenseMatrixNumCoefs(matrix)         = num_rows * num_cols;
   hypre_DenseMatrixData(matrix)             = NULL;
   hypre_DenseMatrixDataAOP(matrix)          = NULL;
   hypre_DenseMatrixVBatchNumRows(matrix)    = NULL;
   hypre_DenseMatrixVBatchNumCols(matrix)    = NULL;
   hypre_DenseMatrixVBatchNumCoefs(matrix)   = NULL;
   hypre_DenseMatrixMemoryLocation(matrix)   = hypre_HandleMemoryLocation(hypre_handle());

   /* set defaults */
   hypre_DenseMatrixRowMajor(matrix)         = 1;
   hypre_DenseMatrixOwnsData(matrix)         = 0;
   hypre_DenseMatrixNumBatches(matrix)       = 1;
   hypre_DenseMatrixUBatchNumRows(matrix)    = num_rows;
   hypre_DenseMatrixUBatchNumCols(matrix)    = num_cols;
   hypre_DenseMatrixUBatchNumCoefs(matrix)   = hypre_DenseMatrixNumCoefs(matrix);
   hypre_DenseMatrixVBatchOwnsArrays(matrix) = 0;

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

      /* Free variable batched dense matrices info */
      if (hypre_DenseMatrixVBatchOwnsArrays(matrix))
      {
         hypre_TFree(hypre_DenseMatrixVBatchNumCoefs(matrix), memory_location);
         if (hypre_DenseMatrixVBatchNumRows(matrix) !=
             hypre_DenseMatrixVBatchNumCols(matrix))
         {
            hypre_TFree(hypre_DenseMatrixVBatchNumRows(matrix), memory_location);
            hypre_TFree(hypre_DenseMatrixVBatchNumCols(matrix), memory_location);
         }
         else
         {
            hypre_TFree(hypre_DenseMatrixVBatchNumRows(matrix), memory_location);
         }
      }

      /* Free matrix pointer */
      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixSetBatchedUniform
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseMatrixSetBatchedUniform( hypre_DenseMatrix  *matrix,
                                    HYPRE_Int           num_batches )
{
   HYPRE_Int  num_rows  = hypre_DenseMatrixNumRows(matrix);
   HYPRE_Int  num_cols  = hypre_DenseMatrixNumCols(matrix);
   HYPRE_Int  num_coefs = hypre_DenseMatrixNumCoefs(matrix);

   if (num_batches < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Number of batches need to be greater than 0\n");
      return hypre_error_flag;
   }

   hypre_DenseMatrixType(matrix)           = HYPRE_DENSE_MATRIX_UBATCHED;
   hypre_DenseMatrixNumBatches(matrix)     = num_batches;
   hypre_DenseMatrixUBatchNumRows(matrix)  = 1 + ((num_rows - 1) / num_batches);
   hypre_DenseMatrixUBatchNumCols(matrix)  = 1 + ((num_cols - 1) / num_batches);
   hypre_DenseMatrixUBatchNumCoefs(matrix) = 1 + ((num_coefs - 1) / num_batches);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixSetBatchedVariable
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseMatrixSetBatchedVariable( hypre_DenseMatrix  *matrix,
                                     HYPRE_Int           num_batches,
                                     HYPRE_Int          *vbatch_num_rows,
                                     HYPRE_Int          *vbatch_num_cols,
                                     HYPRE_Int          *vbatch_num_coefs )
{
   if (num_batches < 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Number of batches need to be greater than 0\n");
      return hypre_error_flag;
   }

   hypre_DenseMatrixType(matrix)             = HYPRE_DENSE_MATRIX_VBATCHED;
   hypre_DenseMatrixNumBatches(matrix)       = num_batches;
   hypre_DenseMatrixVBatchNumRows(matrix)    = vbatch_num_rows;
   hypre_DenseMatrixVBatchNumCols(matrix)    = vbatch_num_cols;
   hypre_DenseMatrixVBatchNumCoefs(matrix)   = vbatch_num_coefs;
   hypre_DenseMatrixVBatchOwnsArrays(matrix) = 1;

   /* Reset defaults referring to uniform batched matrices */
   hypre_DenseMatrixUBatchNumRows(matrix)     = 0;
   hypre_DenseMatrixUBatchNumCols(matrix)     = 0;
   hypre_DenseMatrixUBatchNumCoefs(matrix)    = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixInitialize_v2
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseMatrixInitialize_v2( hypre_DenseMatrix    *matrix,
                                HYPRE_MemoryLocation  memory_location )
{
   HYPRE_Int   num_coefs   = hypre_DenseMatrixNumCoefs(matrix);
   HYPRE_Int   num_batches = hypre_DenseMatrixNumBatches(matrix);

   hypre_DenseMatrixMemoryLocation(matrix) = memory_location;

   /* Allocate memory for data */
   if (!hypre_DenseMatrixData(matrix) && num_coefs)
   {
      hypre_DenseMatrixData(matrix) = hypre_CTAlloc(HYPRE_Complex, num_coefs, memory_location);

      if (num_batches > 1)
      {
         hypre_DenseMatrixDataAOP(matrix) = hypre_TAlloc(HYPRE_Complex *,
                                                         num_batches,
                                                         memory_location);
      }
   }

   hypre_DenseMatrixOwnsData(matrix) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseMatrixInitialize( hypre_DenseMatrix *matrix )
{
   return hypre_DenseMatrixInitialize_v2(matrix, hypre_DenseMatrixMemoryLocation(matrix));
}
