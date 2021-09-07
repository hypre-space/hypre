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
#include "seq_mv.hpp"

#if defined(HYPRE_USING_GPU)

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

#ifdef HYPRE_USING_CUSPARSE
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   /* Luke E: The generic API is techinically supported on 10.1,10.2 as a preview,
    * with Dscrmv being deprecated. However, there are limitations.
    * While in Cuda < 11, there are specific mentions of using csr2csc involving
    * transposed matrix products with dcsrm*,
    * they are not present in SpMV interface.
    */
   hypre_CSRMatrixMatvecCusparseNewAPI(trans, alpha, A, x, beta, y, offset);
#else
   hypre_CSRMatrixMatvecCusparseOldAPI(trans, alpha, A, x, beta, y, offset);
#endif
#elif defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_CSRMatrixMatvecOMPOffload(trans, alpha, A, x, beta, y, offset);
#elif defined(HYPRE_USING_ROCSPARSE)
   hypre_CSRMatrixMatvecRocsparse(trans, alpha, A, x, beta, y, offset);
#else // #ifdef HYPRE_USING_CUSPARSE
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
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
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

   hypre_SyncCudaComputeStream(hypre_handle());

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

HYPRE_Int
hypre_CSRMatrixMatvecCusparseNewAPI( HYPRE_Int        trans,
                                     HYPRE_Complex    alpha,
                                     hypre_CSRMatrix *A,
                                     hypre_Vector    *x,
                                     HYPRE_Complex    beta,
                                     hypre_Vector    *y,
                                     HYPRE_Int        offset )
{
   const cudaDataType        data_type  = hypre_HYPREComplexToCudaDataType();
   const cusparseIndexType_t index_type = hypre_HYPREIntToCusparseIndexType();
   cusparseSpMatDescr_t      matA;
   cusparseHandle_t          handle     = hypre_HandleCusparseHandle(hypre_handle());
   hypre_CSRMatrix          *AT;

   if (trans)
   {
      /* We handle the transpose explicitly to ensure the same output each run
       * and for potential performance improvement memory for AT */
      hypre_CSRMatrixTransposeDevice(A, &AT, 1);
      matA = hypre_CSRMatrixToCusparseSpMat(AT, offset);
   }
   else
   {
      matA = hypre_CSRMatrixToCusparseSpMat(A, offset);
   }

   /* SpMV */
   size_t bufferSize = 0;
   char  *dBuffer    = NULL;
   HYPRE_Int x_size_override = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
   HYPRE_Int y_size_override = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);
   cusparseDnVecDescr_t vecX = hypre_VectorToCusparseDnVec(x,      0, x_size_override);
   cusparseDnVecDescr_t vecY = hypre_VectorToCusparseDnVec(y, offset, y_size_override - offset);

   HYPRE_CUSPARSE_CALL( cusparseSpMV_bufferSize(handle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha,
                                                matA,
                                                vecX,
                                                &beta,
                                                vecY,
                                                data_type,
                                                CUSPARSE_CSRMV_ALG2,
                                                &bufferSize) );

   dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);

   HYPRE_CUSPARSE_CALL( cusparseSpMV(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha,
                                     matA,
                                     vecX,
                                     &beta,
                                     vecY,
                                     data_type,
                                     CUSPARSE_CSRMV_ALG2,
                                     dBuffer) );

   hypre_SyncCudaComputeStream(hypre_handle());

   if (trans)
   {
      hypre_CSRMatrixDestroy(AT);
   }
   hypre_TFree(dBuffer, HYPRE_MEMORY_DEVICE);
   /* This function releases the host memory allocated for the sparse matrix descriptor */
   HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
   HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecX));
   HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecY));

   return hypre_error_flag;
}

#else // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

HYPRE_Int
hypre_CSRMatrixMatvecCusparseOldAPI( HYPRE_Int        trans,
                                     HYPRE_Complex    alpha,
                                     hypre_CSRMatrix *A,
                                     hypre_Vector    *x,
                                     HYPRE_Complex    beta,
                                     hypre_Vector    *y,
                                     HYPRE_Int        offset )
{
#ifdef HYPRE_BIGINT
#error "ERROR: cusparse old API should not be used when bigint is enabled!"
#endif
   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A);
   hypre_CSRMatrix *B;

   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(A, &B, 1);
   }
   else
   {
      B = A;
   }

   HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       hypre_CSRMatrixNumRows(B) - offset,
                                       hypre_CSRMatrixNumCols(B),
                                       hypre_CSRMatrixNumNonzeros(B),
                                       &alpha,
                                       descr,
                                       hypre_CSRMatrixData(B),
                                       hypre_CSRMatrixI(B) + offset,
                                       hypre_CSRMatrixJ(B),
                                       hypre_VectorData(x),
                                       &beta,
                                       hypre_VectorData(y) + offset) );


   if (trans)
   {
      hypre_CSRMatrixDestroy(B);
   }

   return hypre_error_flag;
}

#endif // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#endif // #if defined(HYPRE_USING_CUSPARSE)

#if defined(HYPRE_USING_ROCSPARSE)
HYPRE_Int
hypre_CSRMatrixMatvecRocsparse( HYPRE_Int        trans,
                                HYPRE_Complex    alpha,
                                hypre_CSRMatrix *A,
                                hypre_Vector    *x,
                                HYPRE_Complex    beta,
                                hypre_Vector    *y,
                                HYPRE_Int        offset )
{
   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());
   rocsparse_mat_descr descr = hypre_CSRMatrixGPUMatDescr(A);
   rocsparse_mat_info info = hypre_CSRMatrixGPUMatInfo(A);

   hypre_CSRMatrix *B;

   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(A, &B, 1);
   }
   else
   {
      B = A;
   }

   HYPRE_ROCSPARSE_CALL( rocsparse_dcsrmv(handle,
                                          rocsparse_operation_none,
                                          hypre_CSRMatrixNumRows(B) - offset,
                                          hypre_CSRMatrixNumCols(B),
                                          hypre_CSRMatrixNumNonzeros(B),
                                          &alpha,
                                          descr,
                                          hypre_CSRMatrixData(B),
                                          hypre_CSRMatrixI(B) + offset,
                                          hypre_CSRMatrixJ(B),
                                          info,
                                          hypre_VectorData(x),
                                          &beta,
                                          hypre_VectorData(y) + offset) );

   if (trans)
   {
      hypre_CSRMatrixDestroy(B);
   }

   return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)

#endif // #if defined(HYPRE_USING_GPU)

