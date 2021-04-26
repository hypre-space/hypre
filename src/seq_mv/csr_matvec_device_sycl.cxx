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
#include "csr_matrix_sycl_utils.hpp"

#if defined(HYPRE_USING_SYCL)

/* y = alpha * A * x + beta * y
 * This function is supposed to be only used inside the other functions in this file
 */
static inline HYPRE_Int
hypre_CSRMatrixMatvecDevice2( HYPRE_Int        trans,
                              HYPRE_Complex    alpha,
                              hypre_CSRMatrix *A,
                              hypre_Vector    *x,
                              HYPRE_Complex    beta,
                              hypre_Vector    *y,
                              HYPRE_Int        offset )
{
   if (hypre_VectorData(x) == hypre_VectorData(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice2");
   }

#ifdef HYPRE_USING_ONEMKLSPARSE
   hypre_CSRMatrixMatvecOnemklsparse(trans, alpha, A, x, beta, y, offset);
#else // #ifdef HYPRE_USING_ONEMKLSPARSE
#error HYPRE SPMV TODO
#endif

   return hypre_error_flag;
}

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
#if defined(HYPRE_USING_SYCL)
   hypre_GpuProfilingPushRange("CSRMatrixMatvec");
#endif

   // TODO: RL: do we need offset > 0 at all?
   hypre_assert(offset == 0);

   HYPRE_Int nx = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
   HYPRE_Int ny = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);

   //RL: Note the "<=", since the vectors sometimes can be temporary work spaces that have
   //    large sizes than the needed (such as in par_cheby.c)
   hypre_assert(ny <= hypre_VectorSize(y));
   hypre_assert(nx <= hypre_VectorSize(x));
   hypre_assert(ny <= hypre_VectorSize(b));

   //hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);
   //if (hypre_VectorData(b) != hypre_VectorData(y))
   //{
   //   hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
   //}

   if (hypre_VectorData(b) != hypre_VectorData(y))
   {
      hypre_TMemcpy( hypre_VectorData(y) + offset,
                     hypre_VectorData(b) + offset,
                     HYPRE_Complex,
                     ny - offset,
                     hypre_VectorMemoryLocation(y),
                     hypre_VectorMemoryLocation(b) );

   }

   if (hypre_CSRMatrixNumNonzeros(A) <= 0 || alpha == 0.0)
   {
      hypre_SeqVectorScale(beta, y);
   }
   else
   {
      hypre_CSRMatrixMatvecDevice2(trans, alpha, A, x, beta, y, offset);
   }

   hypre_SyncSyclComputeQueue(hypre_handle());

#if defined(HYPRE_USING_SYCL)
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

#if defined(HYPRE_USING_ONEMKLSPARSE)

HYPRE_Int
hypre_CSRMatrixMatvecMaskedDevice( HYPRE_Complex    alpha,
                                   hypre_CSRMatrix *A,
                                   hypre_Vector    *x,
                                   HYPRE_Complex    beta,
                                   hypre_Vector    *b,
                                   hypre_Vector    *y,
                                   HYPRE_Int       *mask,
                                   HYPRE_Int        size_of_mask )
{
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "hypre_CSRMatrixMatvecMaskedDevice not implemented for onemkl::SPARSE!\n");
}

HYPRE_Int
hypre_CSRMatrixMatvecOnemklsparse( HYPRE_Int        trans,
                                   HYPRE_Complex    alpha,
                                   hypre_CSRMatrix *A,
                                   hypre_Vector    *x,
                                   HYPRE_Complex    beta,
                                   hypre_Vector    *y,
                                   HYPRE_Int        offset )
{
   sycl::queue* handle = hypre_HandleSyclComputeQueue(hypre_handle());
   hypre_CSRMatrix *AT;

   oneapi::mkl::sparse::matrix_handle_t matA_handle;

   if (trans)
   {
      /* We handle the transpose explicitly to ensure the same output each run
       * and for potential performance improvement memory for AT */
      hypre_CSRMatrixTransposeDevice(A, &AT, 1);
      matA_handle = hypre_CSRMatrixToOnemklsparseSpMat(AT, offset, matA_handle);
   }
   else
   {
      matA_handle = hypre_CSRMatrixToOnemklsparseSpMat(A, offset, matA_handle);
   }

   /* SpMV */
   auto event = HYPRE_SYCL_CALL( oneapi::mkl::sparse::gemv(*handle,
                                                           oneapi::mkl::transpose::nontrans,
                                                           alpha,
                                                           matA_handle,
                                                           hypre_VectorData(x),
                                                           beta,
                                                           hypre_VectorData(y) + offset) );
   event.wait();

   if (trans)
   {
      hypre_CSRMatrixDestroy(AT);
   }

   /* This function releases the internals and sets matrix_handle_t to NULL allocated 
      in hypre_CSRMatrixToOnemklsparseSpMat() */
   HYPRE_SYCL_CALL( oneapi::mkl::sparse::release_matrix_handle(matA_handle) );

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_ONEMKLSPARSE)

#endif // #if defined(HYPRE_USING_SYCL)
