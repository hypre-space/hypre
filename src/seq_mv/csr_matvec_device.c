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
#include <cuda_runtime_api.h>
#include "csr_matrix_cuda_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif


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

//The generic API is techinically supported on 10.1,10.2 as a preview, with Dscrmv being deprecated. However, there are limitations.
//While in Cuda < 11, there are specific mentions of using csr2csc involving transposed matrix products with dcsrm*, they are not present in SpMV interface.
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 10010)
   //Cusparse does not seem to handle the case when a vector has size 0
   if((A->num_cols == 0) || (A->num_rows - offset == 0))
   {
   }
   else
   {
      cusparseSpMatDescr_t matA = hypre_CSRMatToCuda(A, offset);

      int y_size_override = 0;
      int x_size_override = 0;
      if((trans?A->num_cols:A->num_rows) != y->size) {
         hypre_printf("WARNING: A matrix-vector product with mismatching dimensions is attempted (likely y incorrect) | %i x %i T:%i , %i \n", A->num_rows, A->num_cols, trans, y->size);
         y_size_override = trans?A->num_cols:A->num_rows;
      }
      if((trans?A->num_rows:A->num_cols) != x->size) {
         hypre_printf("WARNING: A matrix-vector product with mismatching dimensions is attempted (likely x incorrect) | %i x %i T:%i , %i \n", A->num_rows, A->num_cols, trans, x->size);
         x_size_override = trans?A->num_rows:A->num_cols;
      }
      cusparseDnVecDescr_t vecX = hypre_VecToCuda(x, offset, x_size_override);
      cusparseDnVecDescr_t vecY = hypre_VecToCuda(y, offset, y_size_override);
      void* dBuffer = NULL;
      size_t bufferSize;

      const cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG2;
      const cusparseOperation_t oper = trans?CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE;
      const cudaDataType data_type = hypre_getCudaDataTypeComplex();

      //Initial tests indicate that handling the transpose using the oper parameter does not result in degradation
      //Thus it is not handled explicitly currently
      if(trans)
      {
      }

      HYPRE_CUSPARSE_CALL(cusparseSpMV_bufferSize(handle, oper, &alpha, matA, vecX, &beta, vecY, data_type, alg, &bufferSize));
      dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
      HYPRE_CUSPARSE_CALL(cusparseSpMV(handle, oper, &alpha, matA, vecX, &beta, vecY, data_type, alg, dBuffer));

      HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
      HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecX));
      HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecY));
      HYPRE_CUDA_CALL(cudaFree(dBuffer));
   }
#else
   cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle());

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
#endif

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

