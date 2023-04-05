/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_UBatchedDenseMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_UBatchedDenseMatrix *
hypre_UBatchedDenseMatrixCreate( HYPRE_Int row_major,
                                 HYPRE_Int num_batches,
                                 HYPRE_Int num_rows_total,
                                 HYPRE_Int num_cols_total )
{
   hypre_UBatchedDenseMatrix  *A;

   A = hypre_TAlloc(hypre_UBatchedDenseMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_UBatchedDenseMatrixRowMajor(A)       = row_major;
   hypre_UBatchedDenseMatrixNumBatches(A)     = num_batches;
   hypre_UBatchedDenseMatrixNumRowsTotal(A)   = num_rows_total;
   hypre_UBatchedDenseMatrixNumColsTotal(A)   = num_cols_total;
   hypre_UBatchedDenseMatrixNumRows(A)        = 1 + ((num_rows_total - 1) / num_batches);
   hypre_UBatchedDenseMatrixNumCols(A)        = 1 + ((num_cols_total - 1) / num_batches);

   hypre_UBatchedDenseMatrixNumCoefs(A)       = hypre_UBatchedDenseMatrixNumRows(A) *
                                                     hypre_UBatchedDenseMatrixNumCols(A);
   hypre_UBatchedDenseMatrixNumCoefsTotal(A)  = hypre_UBatchedDenseMatrixNumCoefs(A) *
                                                     hypre_UBatchedDenseMatrixNumBatches(A);

   hypre_UBatchedDenseMatrixOwnsData(A)       = 0;
   hypre_UBatchedDenseMatrixData(A)           = NULL;
   hypre_UBatchedDenseMatrixDataAOP(A)        = NULL;
   hypre_UBatchedDenseMatrixMemoryLocation(A) = hypre_HandleMemoryLocation(hypre_handle());

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseMatrixDestroy( hypre_UBatchedDenseMatrix *A )
{
   if (A)
   {
      HYPRE_MemoryLocation memory_location = hypre_UBatchedDenseMatrixMemoryLocation(A);

      if (hypre_UBatchedDenseMatrixOwnsData(A))
      {
         hypre_TFree(hypre_UBatchedDenseMatrixData(A), memory_location);
      }

      /* data_aop is always owned by a hypre_UBatchedDenseMatrix */
      hypre_TFree(hypre_UBatchedDenseMatrixDataAOP(A), memory_location);

      /* Free matrix pointer */
      hypre_TFree(A, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseMatrixInitialize_v2
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseMatrixInitialize_v2( hypre_UBatchedDenseMatrix  *A,
                                        HYPRE_MemoryLocation        memory_location )
{
   HYPRE_Int   num_coefs_total = hypre_UBatchedDenseMatrixNumCoefsTotal(A);
   HYPRE_Int   num_batches     = hypre_UBatchedDenseMatrixNumBatches(A);

   hypre_UBatchedDenseMatrixMemoryLocation(A) = memory_location;

   /* Allocate memory for data */
   if (!hypre_UBatchedDenseMatrixData(A) && num_coefs_total)
   {
      hypre_UBatchedDenseMatrixData(A) = hypre_CTAlloc(HYPRE_Complex,
                                                            num_coefs_total,
                                                            memory_location);
      hypre_UBatchedDenseMatrixOwnsData(A) = 1;

      if (num_batches > 1)
      {
         hypre_UBatchedDenseMatrixDataAOP(A) = hypre_TAlloc(HYPRE_Complex *,
                                                                 num_batches,
                                                                 memory_location);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseMatrixInitialize( hypre_UBatchedDenseMatrix *A )
{
   return hypre_UBatchedDenseMatrixInitialize_v2(A, hypre_UBatchedDenseMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseMatrixCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedDenseMatrixCopy( hypre_UBatchedDenseMatrix *A,
                               hypre_UBatchedDenseMatrix *B )
{
   /* Copy coeficients from matrix A to B */
   hypre_TMemcpy(hypre_UBatchedDenseMatrixData(B),
                 hypre_UBatchedDenseMatrixData(A),
                 HYPRE_Complex,
                 hypre_UBatchedDenseMatrixNumCoefsTotal(A),
                 hypre_UBatchedDenseMatrixMemoryLocation(B),
                 hypre_UBatchedDenseMatrixMemoryLocation(A));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedDenseMatrixClone
 *--------------------------------------------------------------------------*/

hypre_UBatchedDenseMatrix*
hypre_UBatchedDenseMatrixClone( hypre_UBatchedDenseMatrix *A,
                                HYPRE_Int                  copy_data )
{
   HYPRE_Int row_major      = hypre_UBatchedDenseMatrixRowMajor(A);
   HYPRE_Int num_batches    = hypre_UBatchedDenseMatrixNumBatches(A);
   HYPRE_Int num_rows_total = hypre_UBatchedDenseMatrixNumRowsTotal(A);
   HYPRE_Int num_cols_total = hypre_UBatchedDenseMatrixNumColsTotal(A);

   hypre_UBatchedDenseMatrix  *B;

   /* Create new matrix */
   B = hypre_UBatchedDenseMatrixCreate(row_major, num_batches,
                                       num_rows_total, num_cols_total);

   /* Initialize matrix */
   hypre_UBatchedDenseMatrixInitialize_v2(B, hypre_UBatchedDenseMatrixMemoryLocation(A));

   /* Copy data array */
   if (copy_data)
   {
      hypre_UBatchedDenseMatrixCopy(A, B);
   }

   return B;
}
