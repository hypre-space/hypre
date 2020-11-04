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

   HYPRE_Int nx = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
   HYPRE_Int ny = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);

   //RL: Note the "<=", since the vectors sometimes can be temporary work spaces that have
   //    large sizes than the needed (such as in par_cheby.c)
   hypre_assert(ny <= hypre_VectorSize(y));
   hypre_assert(nx <= hypre_VectorSize(x));
   hypre_assert(ny <= hypre_VectorSize(b));

   cusparseHandle_t   handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr  = hypre_HandleCusparseMatDescr(hypre_handle());

   //hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);

   //if (b != y)
   //{
   //   hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
   //}

   if (hypre_VectorData(b) != hypre_VectorData(y))
   {
      HYPRE_THRUST_CALL( copy_n,
                         hypre_VectorData(b) + offset,
                         ny - offset,
                         hypre_VectorData(y) + offset);
   }

   if (hypre_CSRMatrixNumNonzeros(A) <= 0 || alpha == 0.0)
   {
      hypre_SeqVectorScale(beta, y);
   }
   else
   {
      if (trans)
      {
         HYPRE_Complex *csc_a = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumNonzeros(A), HYPRE_MEMORY_DEVICE);
         HYPRE_Int     *csc_j = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumNonzeros(A), HYPRE_MEMORY_DEVICE);
         HYPRE_Int     *csc_i = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumCols(A)+1,   HYPRE_MEMORY_DEVICE);

         HYPRE_CUSPARSE_CALL( cusparseDcsr2csc(handle,
                                               hypre_CSRMatrixNumRows(A),
                                               hypre_CSRMatrixNumCols(A),
                                               hypre_CSRMatrixNumNonzeros(A),
                                               hypre_CSRMatrixData(A),
                                               hypre_CSRMatrixI(A),
                                               hypre_CSRMatrixJ(A),
                                               csc_a,
                                               csc_j,
                                               csc_i,
                                               CUSPARSE_ACTION_NUMERIC,
                                               CUSPARSE_INDEX_BASE_ZERO) );

         HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             hypre_CSRMatrixNumCols(A) - offset,
                                             hypre_CSRMatrixNumRows(A),
                                             hypre_CSRMatrixNumNonzeros(A),
                                             &alpha,
                                             descr,
                                             csc_a,
                                             csc_i + offset,
                                             csc_j,
                                             hypre_VectorData(x),
                                             &beta,
                                             hypre_VectorData(y) + offset) );

         hypre_TFree(csc_a, HYPRE_MEMORY_DEVICE);
         hypre_TFree(csc_i, HYPRE_MEMORY_DEVICE);
         hypre_TFree(csc_j, HYPRE_MEMORY_DEVICE);
      }
      else
      {
         HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             hypre_CSRMatrixNumRows(A) - offset,
                                             hypre_CSRMatrixNumCols(A),
                                             hypre_CSRMatrixNumNonzeros(A),
                                             &alpha,
                                             descr,
                                             hypre_CSRMatrixData(A),
                                             hypre_CSRMatrixI(A) + offset,
                                             hypre_CSRMatrixJ(A),
                                             hypre_VectorData(x),
                                             &beta,
                                             hypre_VectorData(y) + offset) );
      }
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

HYPRE_Int
hypre_CSRMatrixMatvecMaskedDevice( HYPRE_Int        trans,
                                   HYPRE_Complex    alpha,
                                   hypre_CSRMatrix *A,
                                   hypre_Vector    *x,
                                   HYPRE_Complex    beta,
                                   hypre_Vector    *b,
                                   hypre_Vector    *y,
                                   HYPRE_Int       *mask,
                                   HYPRE_Int        size_of_mask,
                                   HYPRE_Int        offset )
{
#ifdef HYPRE_BIGINT
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR: hypre_CSRMatrixMatvecMaskedDevice should not be called when bigint is enabled!");
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecMaskedDevice\n");
   }

   // TODO
   if (offset != 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING:: Offset is not zero in hypre_CSRMatrixMatvecMaskedDevice :: \n");
   }

   hypre_assert(offset == 0);

   if (trans)
   {
      HYPRE_Complex *csc_a = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumNonzeros(A), HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_j = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumNonzeros(A), HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_i = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumCols(A)+1,   HYPRE_MEMORY_DEVICE);

      HYPRE_CUSPARSE_CALL( cusparseDcsr2csc(handle,
                                            hypre_CSRMatrixNumRows(A),
                                            hypre_CSRMatrixNumCols(A),
                                            hypre_CSRMatrixNumNonzeros(A),
                                            hypre_CSRMatrixData(A),
                                            hypre_CSRMatrixI(A),
                                            hypre_CSRMatrixJ(A),
                                            csc_a,
                                            csc_j,
                                            csc_i,
                                            CUSPARSE_ACTION_NUMERIC,
                                            CUSPARSE_INDEX_BASE_ZERO) );

      HYPRE_CUSPARSE_CALL( cusparseDbsrxmv(handle,
                                           CUSPARSE_DIRECTION_ROW,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           size_of_mask,
                                           hypre_CSRMatrixNumRows(A),
                                           hypre_CSRMatrixNumCols(A),
                                           hypre_CSRMatrixNumNonzeros(A),
                                           &alpha,
                                           descr,
                                           csc_a,
                                           mask,
                                           csc_i,
                                           csc_i+1,
                                           csc_j,
                                           1,
                                           hypre_VectorData(x),
                                           &beta,
                                           hypre_VectorData(y)) );

      hypre_TFree(csc_a, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_i, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_j, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          hypre_CSRMatrixNumRows(A)-offset,
                                          hypre_CSRMatrixNumCols(A),
                                          hypre_CSRMatrixNumNonzeros(A),
                                          &alpha,
                                          descr,
                                          hypre_CSRMatrixData(A),
                                          hypre_CSRMatrixI(A)+offset,
                                          hypre_CSRMatrixJ(A),
                                          hypre_VectorData(x),
                                          &beta,
                                          hypre_VectorData(y)+offset) );

      if (!(hypre_CSRMatrixNumRows(A) < 1) &&
          !(hypre_CSRMatrixNumCols(A) < 1) &&
          !(hypre_CSRMatrixNumNonzeros(A) < 1))
      {
         HYPRE_CUSPARSE_CALL(cusparseDbsrxmv(handle,
                                             CUSPARSE_DIRECTION_ROW,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             size_of_mask,
                                             hypre_CSRMatrixNumRows(A),
                                             hypre_CSRMatrixNumCols(A),
                                             hypre_CSRMatrixNumNonzeros(A),
                                             &alpha,
                                             descr,
                                             hypre_CSRMatrixData(A),
                                             mask,
                                             hypre_CSRMatrixI(A)+offset,
                                             hypre_CSRMatrixI(A)+offset+1,
                                             hypre_CSRMatrixJ(A),
                                             1,
                                             hypre_VectorData(x),
                                             &beta,
                                             hypre_VectorData(y)) );
      }
   }

   hypre_SyncCudaComputeStream(hypre_handle());
#endif

   return hypre_error_flag;
}

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
   if(A->num_nonzeros == 0)
   {
      hypre_SeqVectorScale(beta, y);
   }
   else
   {
      cusparseSpMatDescr_t matA = hypre_CSRMatToCuda(A, offset);

      int y_size_override = 0;
      int x_size_override = 0;
      if((trans?A->num_cols:A->num_rows) != y->size)
      {
#if defined(HYPRE_DEBUG) && HYPRE_MV_CHECK_VECTOR_SIZES
         hypre_printf("WARNING: A matrix-vector product with mismatching dimensions is attempted (likely y incorrect) | %i x %i T:%i , %i [%s : %i] \n", A->num_rows, A->num_cols, trans, y->size, __FILE__, __LINE__);
#endif
         y_size_override = trans?A->num_cols:A->num_rows;
      }
      if((trans?A->num_rows:A->num_cols) != x->size)
      {
#if defined(HYPRE_DEBUG) && HYPRE_MV_CHECK_VECTOR_SIZES
         hypre_printf("WARNING: A matrix-vector product with mismatching dimensions is attempted (likely x incorrect) | %i x %i T:%i , %i [%s : %i] \n", A->num_rows, A->num_cols, trans, x->size, __FILE__, __LINE__);
#endif
         x_size_override = trans?A->num_rows:A->num_cols;
      }
      cusparseDnVecDescr_t vecX = hypre_VecToCuda(x, offset, x_size_override);
      cusparseDnVecDescr_t vecY = hypre_VecToCuda(y, offset, y_size_override);
      void* dBuffer = NULL;
      size_t bufferSize;

      const cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG2;
      const cusparseOperation_t oper = CUSPARSE_OPERATION_NON_TRANSPOSE;
      const cudaDataType data_type = hypre_getCudaDataTypeComplex();

      //We handle the transpose explicitly to ensure the same output each run
      //and for potential performance improvement
      if(trans)
      {

         HYPRE_Complex *csc_a = hypre_TAlloc(HYPRE_Complex, A->num_nonzeros, HYPRE_MEMORY_DEVICE);
         HYPRE_Int     *csc_j = hypre_TAlloc(HYPRE_Int,     A->num_nonzeros, HYPRE_MEMORY_DEVICE);
         HYPRE_Int     *csc_i = hypre_TAlloc(HYPRE_Int,     A->num_cols+1,   HYPRE_MEMORY_DEVICE);

         size_t bufferSize = 0;
         size_t *buffer;
         HYPRE_CUSPARSE_CALL( cusparseCsr2cscEx2_bufferSize(handle, A->num_rows, A->num_cols, A->num_nonzeros,
                              A->data, A->i, A->j, csc_a, csc_i, csc_j,
                              CUDA_R_64F,CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                              CUSPARSE_CSR2CSC_ALG1, &bufferSize));
         buffer = (size_t*) hypre_TAlloc(char,     bufferSize,    HYPRE_MEMORY_DEVICE);

         HYPRE_CUSPARSE_CALL( cusparseCsr2cscEx2(handle, A->num_rows, A->num_cols, A->num_nonzeros,
                              A->data, A->i, A->j, csc_a, csc_i, csc_j,
                              CUDA_R_64F,CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                              CUSPARSE_CSR2CSC_ALG1, buffer));

         hypre_TFree(buffer, HYPRE_MEMORY_DEVICE);

         cusparseSpMatDescr_t matAT;

         const cudaDataType data_type = hypre_getCudaDataTypeComplex();
         const cusparseIndexType_t index_type = hypre_getCusparseIndexTypeInt();
         const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;

         HYPRE_CUSPARSE_CALL(cusparseCreateCsr(&matAT, A->num_cols, A->num_rows, A->num_nonzeros, csc_i, csc_j, csc_a, index_type, index_type, index_base, data_type));


         HYPRE_CUSPARSE_CALL(cusparseSpMV_bufferSize(handle, oper, &alpha, matAT, vecX, &beta, vecY, data_type, alg, &bufferSize));
         dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
         HYPRE_CUSPARSE_CALL(cusparseSpMV(handle, oper, &alpha, matAT, vecX, &beta, vecY, data_type, alg, dBuffer));

         hypre_TFree(csc_a, HYPRE_MEMORY_DEVICE);
         hypre_TFree(csc_i, HYPRE_MEMORY_DEVICE);
         hypre_TFree(csc_j, HYPRE_MEMORY_DEVICE);
         HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matAT));
      }
      else
      {
         HYPRE_CUSPARSE_CALL(cusparseSpMV_bufferSize(handle, oper, &alpha, matA, vecX, &beta, vecY, data_type, alg, &bufferSize));
         dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
         HYPRE_CUSPARSE_CALL(cusparseSpMV(handle, oper, &alpha, matA, vecX, &beta, vecY, data_type, alg, dBuffer));
      }


      HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
      HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecX));
      HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecY));
      hypre_TFree(dBuffer, HYPRE_MEMORY_DEVICE);
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


#endif
