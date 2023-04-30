/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
HYPRE_Int
hypre_CSRMatrixMatvecOMPOffload( HYPRE_Int        trans,
                                 HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset )
{
   hypre_CSRMatrix *B;

   if (trans)
   {
      hypre_CSRMatrixTranspose(A, &B, 1);

      /* HYPRE_CUDA_CALL(cudaDeviceSynchronize()); */
   }
   else
   {
      B = A;
   }

   HYPRE_Int      A_nrows  = hypre_CSRMatrixNumRows(B);
   HYPRE_Complex *A_data   = hypre_CSRMatrixData(B);
   HYPRE_Int     *A_i      = hypre_CSRMatrixI(B);
   HYPRE_Int     *A_j      = hypre_CSRMatrixJ(B);
   HYPRE_Complex *x_data   = hypre_VectorData(x);
   HYPRE_Complex *y_data   = hypre_VectorData(y);
   HYPRE_Int      i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(A_data, A_i, A_j, y_data, x_data)
   for (i = offset; i < A_nrows; i++)
   {
      HYPRE_Complex tempx = 0.0;
      HYPRE_Int j;
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         tempx += A_data[j] * x_data[A_j[j]];
      }
      y_data[i] = alpha * tempx + beta * y_data[i];
   }

   /* HYPRE_CUDA_CALL(cudaDeviceSynchronize()); */

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_DEVICE_OPENMP) */

