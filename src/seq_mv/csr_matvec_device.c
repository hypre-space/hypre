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

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle());

   //hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);


   //if (b != y)
   //{
   //   hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
   //}

   if (b != y)
   {
      HYPRE_THRUST_CALL( copy_n, b->data, y->size-offset, y->data );
   }

   if (x == y)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice\n");
   }

   // TODO
   if (offset != 0)
   {
      hypre_printf("WARNING:: Offset is not zero in hypre_CSRMatrixMatvecDevice :: \n");
   }

   hypre_assert(offset == 0);

   if (trans)
   {
      HYPRE_Complex *csc_a = hypre_TAlloc(HYPRE_Complex, A->num_nonzeros, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_j = hypre_TAlloc(HYPRE_Int,     A->num_nonzeros, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_i = hypre_TAlloc(HYPRE_Int,     A->num_cols+1,   HYPRE_MEMORY_DEVICE);

      HYPRE_CUSPARSE_CALL( cusparseDcsr2csc(handle, A->num_rows, A->num_cols, A->num_nonzeros,
                           A->data, A->i, A->j, csc_a, csc_j, csc_i,
                           CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO) );

      HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           A->num_cols, A->num_rows, A->num_nonzeros,
                           &alpha, descr,
                           csc_a, csc_i, csc_j,
                           x->data, &beta, y->data) );

      hypre_TFree(csc_a, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_i, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_j, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           A->num_rows-offset, A->num_cols, A->num_nonzeros,
                           &alpha, descr,
                           A->data, A->i+offset, A->j,
                           x->data, &beta, y->data+offset) );
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

