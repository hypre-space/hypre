/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#if defined(HYPRE_USING_CUDA)
/* y = alpha * A * x + beta * b */
HYPRE_Int
hypre_CSRMatrixMatvecDevice( HYPRE_Int        trans,
                             HYPRE_Complex    alpha,
                             hypre_CSRMatrix *A,
                             hypre_Vector    *x,
                             HYPRE_Complex    beta,
                             hypre_Vector    *b,
                             hypre_Vector    *y,
                             HYPRE_Int        offset )
{
#ifdef HYPRE_BIGINT
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR: hypre_CSRMatvecDevice should not be called when bigint is enabled!");
#else

   // TODO
   hypre_assert(offset == 0);

   if (x == y)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice\n");
   }

   if (trans)
   {
      hypre_assert(hypre_CSRMatrixNumCols(A) == hypre_VectorSize(y));
      hypre_assert(hypre_CSRMatrixNumRows(A) == hypre_VectorSize(x));
   }
   else
   {
      hypre_assert(hypre_CSRMatrixNumCols(A) == hypre_VectorSize(x));
      hypre_assert(hypre_CSRMatrixNumRows(A) == hypre_VectorSize(y));
   }
   hypre_assert(hypre_VectorSize(y) == hypre_VectorSize(b));

   cusparseHandle_t   handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr  = hypre_HandleCusparseMatDescr(hypre_handle());

   //hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);

   //if (b != y)
   //{
   //   hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
   //}

   if (b != y)
   {
      HYPRE_THRUST_CALL( copy_n,
                         hypre_VectorData(b),
                         hypre_VectorSize(y) - offset,
                         hypre_VectorData(y) );
   }

   if (hypre_CSRMatrixNumNonzeros(A) <= 0)
   {
      hypre_SeqVectorScale(beta, y);

      return hypre_error_flag;
   }

   if (trans)
   {
      HYPRE_Complex *csc_a = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumNonzeros(A), HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_j = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumNonzeros(A), HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_i = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumCols(A)+1,   HYPRE_MEMORY_DEVICE);

      HYPRE_CUSPARSE_CALL( cusparseDcsr2csc(handle, hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A),
                                            hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixData(A),
                                            hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), csc_a, csc_j, csc_i,
                                            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO) );

      HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          hypre_CSRMatrixNumCols(A), hypre_CSRMatrixNumRows(A),
                                          hypre_CSRMatrixNumNonzeros(A),
                                          &alpha, descr,
                                          csc_a, csc_i, csc_j,
                                          hypre_VectorData(x), &beta, hypre_VectorData(y)) );

      hypre_TFree(csc_a, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_i, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_j, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          hypre_CSRMatrixNumRows(A) - offset, hypre_CSRMatrixNumCols(A),
                                          hypre_CSRMatrixNumNonzeros(A),
                                          &alpha, descr,
                                          hypre_CSRMatrixData(A), hypre_CSRMatrixI(A) + offset, hypre_CSRMatrixJ(A),
                                          hypre_VectorData(x), &beta, hypre_VectorData(y) + offset) );
   }

   hypre_SyncCudaComputeStream(hypre_handle());
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_CSRMatrixMatvecDeviceBIGINT( HYPRE_Complex    alpha,
                                   hypre_CSRMatrix *A,
                                   hypre_Vector    *x,
                                   HYPRE_Complex    beta,
                                   hypre_Vector    *b,
                                   hypre_Vector    *y,
                                   HYPRE_Int offset )
{
#ifdef HYPRE_BIGINT
#error "TODO BigInt"
#endif
  return 0;
}

#endif

