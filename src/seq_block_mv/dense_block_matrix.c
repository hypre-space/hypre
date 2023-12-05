/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_DenseBlockMatrix class.
 *
 *****************************************************************************/

#include "_hypre_seq_block_mv.h"

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_DenseBlockMatrix *
hypre_DenseBlockMatrixCreate( HYPRE_Int  row_major,
                              HYPRE_Int  num_rows,
                              HYPRE_Int  num_cols,
                              HYPRE_Int  num_rows_block )
{
   hypre_DenseBlockMatrix  *A;
   HYPRE_Int                num_blocks = 1 + ((num_rows - 1) / num_rows_block);

   A = hypre_TAlloc(hypre_DenseBlockMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_DenseBlockMatrixRowMajor(A)       = row_major;
   hypre_DenseBlockMatrixNumRowsBlock(A)   = num_rows_block;
   hypre_DenseBlockMatrixNumColsBlock(A)   = 1 + ((num_cols - 1) / num_blocks);
   hypre_DenseBlockMatrixNumBlocks(A)      = num_blocks;
   hypre_DenseBlockMatrixNumRows(A)        = num_blocks * hypre_DenseBlockMatrixNumRowsBlock(A);
   hypre_DenseBlockMatrixNumCols(A)        = num_blocks * hypre_DenseBlockMatrixNumColsBlock(A);
   hypre_DenseBlockMatrixNumCoefsBlock(A)  = hypre_DenseBlockMatrixNumRowsBlock(A) *
                                             hypre_DenseBlockMatrixNumColsBlock(A);
   hypre_DenseBlockMatrixNumCoefs(A)       = num_blocks * hypre_DenseBlockMatrixNumCoefsBlock(A);
   hypre_DenseBlockMatrixOwnsData(A)       = 0;
   hypre_DenseBlockMatrixData(A)           = NULL;
   hypre_DenseBlockMatrixDataAOP(A)        = NULL;
   hypre_DenseBlockMatrixMemoryLocation(A) = hypre_HandleMemoryLocation(hypre_handle());

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixCreateByBlock
 *--------------------------------------------------------------------------*/

hypre_DenseBlockMatrix *
hypre_DenseBlockMatrixCreateByBlock( HYPRE_Int  row_major,
                                     HYPRE_Int  num_blocks,
                                     HYPRE_Int  num_rows_block )
{
   return hypre_DenseBlockMatrixCreate(row_major,
                                       num_blocks * num_rows_block,
                                       num_blocks * num_rows_block,
                                       num_rows_block);
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixDestroy( hypre_DenseBlockMatrix *A )
{
   if (A)
   {
      HYPRE_MemoryLocation memory_location = hypre_DenseBlockMatrixMemoryLocation(A);

      if (hypre_DenseBlockMatrixOwnsData(A))
      {
         hypre_TFree(hypre_DenseBlockMatrixData(A), memory_location);
      }

      /* data_aop is always owned by a hypre_DenseBlockMatrix */
      hypre_TFree(hypre_DenseBlockMatrixDataAOP(A), memory_location);

      /* Free matrix pointer */
      hypre_TFree(A, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixInitializeOn
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixInitializeOn( hypre_DenseBlockMatrix  *A,
                                    HYPRE_MemoryLocation     memory_location )
{
   HYPRE_Int   num_coefs  = hypre_DenseBlockMatrixNumCoefs(A);
   HYPRE_Int   num_blocks = hypre_DenseBlockMatrixNumBlocks(A);

   hypre_DenseBlockMatrixMemoryLocation(A) = memory_location;

   /* Allocate memory for data */
   if (!hypre_DenseBlockMatrixData(A) && num_coefs)
   {
      hypre_DenseBlockMatrixData(A) = hypre_CTAlloc(HYPRE_Complex,
                                                    num_coefs,
                                                    memory_location);
      hypre_DenseBlockMatrixOwnsData(A) = 1;

      if (num_blocks > 1)
      {
         hypre_DenseBlockMatrixDataAOP(A) = hypre_TAlloc(HYPRE_Complex *,
                                                         num_blocks,
                                                         memory_location);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixInitialize( hypre_DenseBlockMatrix *A )
{
   return hypre_DenseBlockMatrixInitializeOn(A, hypre_DenseBlockMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixCopy( hypre_DenseBlockMatrix *A,
                            hypre_DenseBlockMatrix *B )
{
   /* Copy coeficients from matrix A to B */
   hypre_TMemcpy(hypre_DenseBlockMatrixData(B),
                 hypre_DenseBlockMatrixData(A),
                 HYPRE_Complex,
                 hypre_DenseBlockMatrixNumCoefs(A),
                 hypre_DenseBlockMatrixMemoryLocation(B),
                 hypre_DenseBlockMatrixMemoryLocation(A));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixClone
 *--------------------------------------------------------------------------*/

hypre_DenseBlockMatrix*
hypre_DenseBlockMatrixClone( hypre_DenseBlockMatrix *A,
                             HYPRE_Int               copy_data )
{
   HYPRE_Int row_major      = hypre_DenseBlockMatrixRowMajor(A);
   HYPRE_Int num_rows       = hypre_DenseBlockMatrixNumRows(A);
   HYPRE_Int num_cols       = hypre_DenseBlockMatrixNumCols(A);
   HYPRE_Int num_rows_block = hypre_DenseBlockMatrixNumRowsBlock(A);

   hypre_DenseBlockMatrix  *B;

   /* Create new matrix */
   B = hypre_DenseBlockMatrixCreate(row_major, num_rows, num_cols, num_rows_block);

   /* Initialize matrix */
   hypre_DenseBlockMatrixInitializeOn(B, hypre_DenseBlockMatrixMemoryLocation(A));

   /* Copy data array */
   if (copy_data)
   {
      hypre_DenseBlockMatrixCopy(A, B);
   }

   return B;
}
