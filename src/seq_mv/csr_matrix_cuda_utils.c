#include "csr_matrix_cuda_utils.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_CUSPARSE)

#ifndef false
#define false 0
#endif

#if (CUDART_VERSION >= 8000)
//Oldest online documentation currently available, so I cannot confirm the
//existence in earlier cuda versions

/*
 * @brief Determines the associated CudaDataType for the HYPRE_Complex typedef
 * @return Returns cuda data type corresponding with HYPRE_Complex
 *
 * @todo Should be known compile time
 * @todo Support more sizes
 * @todo Support complex
 * @warning Only works for Single and Double precision
 * @note Perhaps some typedefs should be added where HYPRE_Complex is typedef'd
 */
cudaDataType hypre_getCudaDataTypeComplex()
{
   if(sizeof(char)*CHAR_BIT!=8)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported char size");
      hypre_assert(false);
   }

#if defined(HYPRE_COMPLEX)
#error "Complex types not yet supported"
#endif

   if((sizeof(HYPRE_Complex) != 8) && (sizeof(HYPRE_Complex) != 4))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported HYPRE_Complex size");
      assert(false);
   }

   if(sizeof(HYPRE_Complex) == 8)
   {
      return CUDA_R_64F;
   }
   if(sizeof(HYPRE_Complex) == 4)
   {
      return CUDA_R_32F;
   }
   hypre_assert(false);
   return CUDA_R_64F;
}

/*
 * @brief Sorts an unsorted CSR INPLACE
 * @param[in] cusparsehandle A Cusparse handle
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnzA Number of nonzeroes
 * @param[in] *d_ia (Unsorted) Row indices
 * @param[in,out] *d_ja_sorted On Start: Unsorted column indices. On return: Sorted column indices
 * @param[in,out] *d_a_sorted On Start: Unsorted values. On Return: Sorted values corresponding with column indices
 */
void hypre_sortCSRCusparse(cusparseHandle_t cusparsehandle, HYPRE_Int n, HYPRE_Int m, HYPRE_Int nnzA, const HYPRE_Int *d_ia, HYPRE_Int *d_ja_sorted, HYPRE_Complex *d_a_sorted)
{
   csru2csrInfo_t sortInfoA;

   cusparseMatDescr_t descrA=0;
   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descrA) );



   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   HYPRE_CUSPARSE_CALL(cusparseCreateCsru2csrInfo(&sortInfoA));
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsru2csr_bufferSizeExt(cusparsehandle, n, m, nnzA, d_a_sorted, d_ia, d_ja_sorted, sortInfoA, &pBufferSizeInBytes));
      pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);
      HYPRE_CUSPARSE_CALL(cusparseDcsru2csr(cusparsehandle, n, m, nnzA, descrA, d_a_sorted, d_ia, d_ja_sorted, sortInfoA, pBuffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsru2csr_bufferSizeExt(cusparsehandle, n, m, nnzA, (float *) d_a_sorted, d_ia, d_ja_sorted, sortInfoA, &pBufferSizeInBytes));
      pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);
      HYPRE_CUSPARSE_CALL(cusparseScsru2csr(cusparsehandle, n, m, nnzA, descrA, (float *)d_a_sorted, d_ia, d_ja_sorted, sortInfoA, pBuffer));
   }
   hypre_TFree(pBuffer, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsru2csrInfo(sortInfoA));
}
#endif

#if (CUDART_VERSION >= 10010)

/*
 * @brief Creates a cuSPARSE CSR descriptor from a hypre_CSRMatrix
 * @param[in] *A Pointer to hypre_CSRMatrix
 * @param[in] offset Row offset
 * @return cuSPARSE CSR Descriptor
 * @warning Assumes CSRMatrix has base 0
 */
cusparseSpMatDescr_t hypre_CSRMatToCuda(const hypre_CSRMatrix *A, HYPRE_Int offset)
{
   const cudaDataType data_type = hypre_getCudaDataTypeComplex();
   const cusparseIndexType_t index_type = hypre_getCusparseIndexTypeInt();
   const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;

   cusparseSpMatDescr_t matA;
   HYPRE_CUSPARSE_CALL(cusparseCreateCsr(&matA, A->num_rows-offset, A->num_cols, A->num_nonzeros, A->i+offset, A->j, A->data, index_type, index_type, index_base, data_type));
   return matA;
}


/*
 * @brief Creates a cuSPARSE dense vector descriptor from a hypre_Vector
 * @param[in] *x Pointer to a hypre_Vector
 * @param[in] offset Row offset
 * @return cuSPARSE dense vector descriptor
 * @warning Assumes CSRMatrix uses doubles for values
 */
cusparseDnVecDescr_t hypre_VecToCuda(const hypre_Vector *x, HYPRE_Int offset, HYPRE_Int size_override)
{
   const cudaDataType data_type = hypre_getCudaDataTypeComplex();
   cusparseDnVecDescr_t vecX;
   HYPRE_CUSPARSE_CALL(cusparseCreateDnVec(&vecX, (size_override>0)?size_override:x->size-offset, x->data+offset, data_type));
   return vecX;
}

/*
 * @brief Determines the associated CudaDataType for the HYPRE_Complex typedef
 * @return Returns cuda data type corresponding with HYPRE_Complex
 *
 * @todo Should be known compile time
 * @warning Only works for Single and Double precision
 * @note Perhaps some typedefs should be added where HYPRE_Complex is typedef'd
 */

cusparseIndexType_t hypre_getCusparseIndexTypeInt()
{

   if(sizeof(char)*CHAR_BIT!=8)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported char size");
      hypre_assert(false);
   }

   if((sizeof(HYPRE_Int) != 8) && (sizeof(HYPRE_Int) != 4))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported HYPRE_Int size");
      hypre_assert(false);
   }

   if(sizeof(HYPRE_Int) == 4)
   {
      return CUSPARSE_INDEX_32I;
   }
   if(sizeof(HYPRE_Int) == 8)
   {
      return CUSPARSE_INDEX_64I;
   }
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ERROR:  Unsupported HYPRE_Int size");
   hypre_assert(false);
   return CUSPARSE_INDEX_32I;
}




/*
 * @brief Creates a cuda csr descriptor for a raw CSR matrix
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnz Number of nonzeroes
 * @param[in] *i Row indices
 * @param[in] *j Colmn indices
 * @param[in] *data Values
 * @return Descriptor
 */
cusparseSpMatDescr_t hypre_CSRMatRawToCuda(HYPRE_Int n, HYPRE_Int m, HYPRE_Int nnz,
      HYPRE_Int *i, HYPRE_Int *j, HYPRE_Complex *data)
{

   const cudaDataType data_type = hypre_getCudaDataTypeComplex();
   const cusparseIndexType_t index_type = hypre_getCusparseIndexTypeInt();
   const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;

   cusparseSpMatDescr_t matA;
   HYPRE_CUSPARSE_CALL(cusparseCreateCsr(&matA, n, m, nnz, i, j, data, index_type, index_type, index_base, data_type));
   return matA;
}


/*
 * @brief A near drop-in replacement for cusparseDcsrmv. It is much better to
 * use  hypre_CSRMatrixMatvecDevice if possible, which has much more error checking.
 * @warning Use hypre_CSRMatrixMatvecDevice instead
 */
void hypre_cusparse_csrmv(cusparseHandle_t handle, cusparseOperation_t oper,HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnz, HYPRE_Complex* alpha,
      HYPRE_Complex* d_a, HYPRE_Int* d_i, HYPRE_Int* d_j, HYPRE_Complex* d_x, HYPRE_Complex* beta,
      HYPRE_Complex* d_y) {

      const cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG2;
      hypre_assert(oper == CUSPARSE_OPERATION_NON_TRANSPOSE);
      const cudaDataType data_type = hypre_getCudaDataTypeComplex();
      cusparseDnVecDescr_t vecX;
      cusparseDnVecDescr_t vecY;
      HYPRE_CUSPARSE_CALL(cusparseCreateDnVec(&vecX, n, d_x, data_type));
      HYPRE_CUSPARSE_CALL(cusparseCreateDnVec(&vecY, m, d_y, data_type));
      const cusparseIndexType_t index_type = hypre_getCusparseIndexTypeInt();
      const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;
      cusparseSpMatDescr_t matA;
      HYPRE_CUSPARSE_CALL(cusparseCreateCsr(&matA, m, n, nnz, d_i, d_j, d_a, index_type, index_type, index_base, data_type));
      void* dBuffer = NULL;
      size_t bufferSize;
      HYPRE_CUSPARSE_CALL(cusparseSpMV_bufferSize(handle, oper, alpha, matA, vecX, beta, vecY, data_type, alg, &bufferSize));
      dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
      HYPRE_CUSPARSE_CALL(cusparseSpMV(handle, oper, alpha, matA, vecX, beta, vecY, data_type, alg, dBuffer));
      HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
      HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecX));
      HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecY));
      hypre_TFree(dBuffer, HYPRE_MEMORY_DEVICE);
}

#endif

#endif
