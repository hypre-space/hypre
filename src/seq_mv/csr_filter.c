/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Methods for matrix truncation/filtering
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTruncateDiag
 *
 * Truncates the input matrix to its diagonal portion.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTruncateDiag(hypre_CSRMatrix *A)
{
   HYPRE_MemoryLocation  memory_location = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_Int             num_rows        = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex        *A_a;
   HYPRE_Int            *A_i, *A_j;

   /* Extract diagonal */
   A_a = hypre_TAlloc(HYPRE_Complex, num_rows, memory_location);
   hypre_CSRMatrixExtractDiagonal(A, A_a, 0);

   /* Free old matrix data */
   hypre_TFree(hypre_CSRMatrixData(A), memory_location);
   hypre_TFree(hypre_CSRMatrixI(A), memory_location);
   hypre_TFree(hypre_CSRMatrixJ(A), memory_location);

   /* Update matrix sparsity pattern */
   A_i = hypre_TAlloc(HYPRE_Int, num_rows + 1, memory_location);
   A_j = hypre_TAlloc(HYPRE_Int, num_rows, memory_location);
   hypre_IntSequence(memory_location, num_rows + 1, A_i);
   hypre_IntSequence(memory_location, num_rows, A_j);

   /* Update matrix pointers and number of nonzero entries */
   hypre_CSRMatrixNumNonzeros(A) = num_rows;
   hypre_CSRMatrixI(A) = A_i;
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_a;

   return hypre_error_flag;
}
