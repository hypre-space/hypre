/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_block_mv.h"

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixMultiplyHost
 *
 * TODO (VPM): implement special cases such as (locally):
 *    1) A = 1x2 and B = 2x2
 *    2) A = 1x3 and B = 3x3
 *    3) A = 1x4 and B = 4x4
 *
 * TODO (VPM): use lapack's dgemm for large matrices (local blocks).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixMultiplyHost( hypre_DenseBlockMatrix  *A,
                                    hypre_DenseBlockMatrix  *B,
                                    hypre_DenseBlockMatrix  *C)
{
   HYPRE_Int       num_blocks       = hypre_DenseBlockMatrixNumBlocks(A);
   HYPRE_Int       num_rows_block_C = hypre_DenseBlockMatrixNumRowsBlock(C);
   HYPRE_Int       num_cols_block_C = hypre_DenseBlockMatrixNumColsBlock(C);
   HYPRE_Int       num_rows_block_B = hypre_DenseBlockMatrixNumRowsBlock(B);

   HYPRE_Int       num_nonzeros_block_A = hypre_DenseBlockMatrixNumNonzerosBlock(A);
   HYPRE_Int       num_nonzeros_block_B = hypre_DenseBlockMatrixNumNonzerosBlock(B);
   HYPRE_Int       num_nonzeros_block_C = hypre_DenseBlockMatrixNumNonzerosBlock(C);

   HYPRE_Int       ib;

#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(ib) HYPRE_SMP_SCHEDULE
#endif
   for (ib = 0; ib < num_blocks; ib++)
   {
      HYPRE_Int       i, j, k;
      HYPRE_Complex  *data_A = hypre_DenseBlockMatrixData(A) + ib * num_nonzeros_block_A;
      HYPRE_Complex  *data_B = hypre_DenseBlockMatrixData(B) + ib * num_nonzeros_block_B;
      HYPRE_Complex  *data_C = hypre_DenseBlockMatrixData(C) + ib * num_nonzeros_block_C;

      for (i = 0; i < num_rows_block_C; i++)
      {
         for (j = 0; j < num_cols_block_C; j++)
         {
            for (k = 0; k < num_rows_block_B; k++)
            {
               /* C[i][j] += A[i][k] * B[k][j]; */
               hypre_DenseBlockMatrixDataIJ(C, data_C, i, j) +=
                  hypre_DenseBlockMatrixDataIJ(A, data_A, i, k) *
                  hypre_DenseBlockMatrixDataIJ(B, data_B, k, j);
            }
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseBlockMatrixMultiply
 *
 * Computes: C = A * B.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseBlockMatrixMultiply( hypre_DenseBlockMatrix   *A,
                                hypre_DenseBlockMatrix   *B,
                                hypre_DenseBlockMatrix  **C_ptr)
{
   hypre_DenseBlockMatrix  *C = *C_ptr;

   /* Check if multiplication makes sense */
   if (hypre_DenseBlockMatrixNumCols(A) != hypre_DenseBlockMatrixNumRows(B))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "cols(A) != rows(B)");
      return hypre_error_flag;
   }

   if (hypre_DenseBlockMatrixNumColsBlock(A) != hypre_DenseBlockMatrixNumRowsBlock(B))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "local cols(A) != local rows(B)");
      return hypre_error_flag;
   }

   /* Create and initialize output matrix if necessary */
   if (!C)
   {
      /* Use same storage layout as A */
      C = hypre_DenseBlockMatrixCreate(hypre_DenseBlockMatrixRowMajor(A),
                                       hypre_DenseBlockMatrixNumRows(A),
                                       hypre_DenseBlockMatrixNumCols(B),
                                       hypre_DenseBlockMatrixNumRowsBlock(A),
                                       hypre_DenseBlockMatrixNumColsBlock(B));
      hypre_DenseBlockMatrixInitializeOn(C, hypre_DenseBlockMatrixMemoryLocation(A));
   }
   else
   {
      /* Reset output coefficients to zero */
      hypre_Memset(hypre_DenseBlockMatrixData(C), 0,
                   hypre_DenseBlockMatrixNumNonzeros(C) * sizeof(HYPRE_Complex),
                   hypre_DenseBlockMatrixMemoryLocation(C));
   }

   /* Compute matrix C */
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2(hypre_DenseBlockMatrixMemoryLocation(A),
                                                     hypre_DenseBlockMatrixMemoryLocation(B));

   if (exec == HYPRE_EXEC_DEVICE)
   {
      /* TODO (VPM): Implement hypre_DenseBlockMatrixMultiplyDevice */
      hypre_DenseBlockMatrixMigrate(A, HYPRE_MEMORY_HOST);
      hypre_DenseBlockMatrixMigrate(B, HYPRE_MEMORY_HOST);
      hypre_DenseBlockMatrixMigrate(C, HYPRE_MEMORY_HOST);
      hypre_DenseBlockMatrixMultiplyHost(A, B, C);
      hypre_DenseBlockMatrixMigrate(A, HYPRE_MEMORY_DEVICE);
      hypre_DenseBlockMatrixMigrate(B, HYPRE_MEMORY_DEVICE);
      hypre_DenseBlockMatrixMigrate(C, HYPRE_MEMORY_DEVICE);
   }
   else
#endif
   {
      hypre_DenseBlockMatrixMultiplyHost(A, B, C);
   }

   /* Set output pointer */
   *C_ptr = C;

   return hypre_error_flag;
}
