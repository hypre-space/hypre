/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_UBatchedCSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_UBatchedCSRMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_UBatchedCSRMatrix *
hypre_UBatchedCSRMatrixCreate( HYPRE_Int num_batches,
                               HYPRE_Int num_rows_total,
                               HYPRE_Int num_cols_total,
                               HYPRE_Int num_coefs_total )
{
   hypre_UBatchedCSRMatrix  *A;

   A = hypre_TAlloc(hypre_UBatchedCSRMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_UBatchedCSRMatrixNumBatches(A)     = num_batches;
   hypre_UBatchedCSRMatrixNumRowsTotal(A)   = num_rows_total;
   hypre_UBatchedCSRMatrixNumColsTotal(A)   = num_cols_total;
   hypre_UBatchedCSRMatrixNumCoefsTotal(A)  = num_coefs_total;

   hypre_UBatchedCSRMatrixNumRows(A)        = 1 + ((num_rows_total - 1) / num_batches);
   hypre_UBatchedCSRMatrixNumCols(A)        = 1 + ((num_cols_total - 1) / num_batches);
   hypre_UBatchedCSRMatrixNumCoefs(A)       = 1 + ((num_coefs_total - 1) / num_batches);
   hypre_UBatchedCSRMatrixI(A)              = NULL;
   hypre_UBatchedCSRMatrixJ(A)              = NULL;

   hypre_UBatchedCSRMatrixOwnsData(A)       = 0;
   hypre_UBatchedCSRMatrixData(A)           = NULL;
   hypre_UBatchedCSRMatrixDataAOP(A)        = NULL;
   hypre_UBatchedCSRMatrixMemoryLocation(A) = hypre_HandleMemoryLocation(hypre_handle());

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedCSRMatrixDestroy( hypre_UBatchedCSRMatrix *A )
{
   if (A)
   {
      HYPRE_MemoryLocation memory_location = hypre_UBatchedCSRMatrixMemoryLocation(A);

      if (hypre_UBatchedCSRMatrixOwnsData(A))
      {
         hypre_TFree(hypre_UBatchedCSRMatrixData(A), memory_location);
      }

      /* data_aop is always owned by a hypre_UBatchedCSRMatrix */
      hypre_TFree(hypre_UBatchedCSRMatrixDataAOP(A), memory_location);

      /* Free matrix pointer */
      hypre_TFree(A, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedCSRMatrixInitialize_v2
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedCSRMatrixInitialize_v2( hypre_UBatchedCSRMatrix  *A,
                                      HYPRE_MemoryLocation      memory_location )
{
   HYPRE_Int   num_coefs_total = hypre_UBatchedCSRMatrixNumCoefsTotal(A);
   HYPRE_Int   num_coefs       = hypre_UBatchedCSRMatrixNumCoefs(A);
   HYPRE_Int   num_rows        = hypre_UBatchedCSRMatrixNumRows(A);
   HYPRE_Int   num_batches     = hypre_UBatchedCSRMatrixNumBatches(A);

   hypre_assert(num_coefs_total >= num_coefs);
   hypre_assert(num_rows >= 0);
   hypre_UBatchedCSRMatrixMemoryLocation(A) = memory_location;

   /* Allocate memory for data */
   if (!hypre_UBatchedCSRMatrixData(A) && num_coefs_total)
   {
      hypre_UBatchedCSRMatrixData(A) = hypre_CTAlloc(HYPRE_Complex,
                                                     num_coefs_total,
                                                     memory_location);
      hypre_UBatchedCSRMatrixOwnsData(A) = 1;

      if (num_batches > 1)
      {
         hypre_UBatchedCSRMatrixDataAOP(A) = hypre_TAlloc(HYPRE_Complex *,
                                                          num_batches,
                                                          memory_location);
      }
   }

   /* Allocate memory for row pointer array */
   if (!hypre_UBatchedCSRMatrixI(A) && num_rows)
   {
      hypre_UBatchedCSRMatrixI(A) = hypre_CTAlloc(HYPRE_Int,
                                                  num_rows + 1,
                                                  memory_location);
   }

   /* Allocate memory for column indices array */
   if (!hypre_UBatchedCSRMatrixJ(A) && num_coefs)
   {
      hypre_UBatchedCSRMatrixJ(A) = hypre_CTAlloc(HYPRE_Int,
                                                  num_coefs,
                                                  memory_location);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedCSRMatrixInitialize( hypre_UBatchedCSRMatrix *A )
{
   return hypre_UBatchedCSRMatrixInitialize_v2(A, hypre_UBatchedCSRMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedCSRMatrixCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UBatchedCSRMatrixCopy( hypre_UBatchedCSRMatrix *A,
                             hypre_UBatchedCSRMatrix *B )
{
   /* Copy coeficients from matrix A to B */
   hypre_TMemcpy(hypre_UBatchedCSRMatrixData(B),
                 hypre_UBatchedCSRMatrixData(A),
                 HYPRE_Complex,
                 hypre_UBatchedCSRMatrixNumCoefsTotal(A),
                 hypre_UBatchedCSRMatrixMemoryLocation(B),
                 hypre_UBatchedCSRMatrixMemoryLocation(A));

   /* Copy row pointer array from matrix A to B */
   hypre_TMemcpy(hypre_UBatchedCSRMatrixI(B),
                 hypre_UBatchedCSRMatrixI(A),
                 HYPRE_Complex,
                 hypre_UBatchedCSRMatrixNumRows(A) + 1,
                 hypre_UBatchedCSRMatrixMemoryLocation(B),
                 hypre_UBatchedCSRMatrixMemoryLocation(A));

   /* Copy coeficients from matrix A to B */
   hypre_TMemcpy(hypre_UBatchedCSRMatrixJ(B),
                 hypre_UBatchedCSRMatrixJ(A),
                 HYPRE_Complex,
                 hypre_UBatchedCSRMatrixNumCoefs(A),
                 hypre_UBatchedCSRMatrixMemoryLocation(B),
                 hypre_UBatchedCSRMatrixMemoryLocation(A));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_UBatchedCSRMatrixClone
 *--------------------------------------------------------------------------*/

hypre_UBatchedCSRMatrix*
hypre_UBatchedCSRMatrixClone( hypre_UBatchedCSRMatrix *A,
                              HYPRE_Int                copy_data )
{
   HYPRE_Int num_batches     = hypre_UBatchedCSRMatrixNumBatches(A);
   HYPRE_Int num_rows_total  = hypre_UBatchedCSRMatrixNumRowsTotal(A);
   HYPRE_Int num_cols_total  = hypre_UBatchedCSRMatrixNumColsTotal(A);
   HYPRE_Int num_coefs_total = hypre_UBatchedCSRMatrixNumCoefsTotal(A);

   hypre_UBatchedCSRMatrix   *B;

   /* Create new matrix */
   B = hypre_UBatchedCSRMatrixCreate(num_batches, num_rows_total,
                                     num_cols_total, num_coefs_total);

   /* Initialize matrix */
   hypre_UBatchedCSRMatrixInitialize_v2(B, hypre_UBatchedCSRMatrixMemoryLocation(A));

   /* Copy row pointer, column indices, and data arrays */
   if (copy_data)
   {
      hypre_UBatchedCSRMatrixCopy(A, B);
   }

   return B;
}
