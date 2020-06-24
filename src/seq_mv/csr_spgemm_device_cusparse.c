/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA)

#if (defined(HYPRE_BIGINT) || defined(HYPRE_MIXEDINT) || defined(HYPRE_SINGLE) || defined(HYPRE_LONG_DOUBLE))
#error "Cuda datatypes not dynamically determined"
#endif

/*
 * @brief Sorts an unsorted CSR
 * @param[in] cusparsehandle A Cusparse handle
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnzA Number of nonzeroes
 * @param[in] *d_ia (Unsorted) Row indices
 * @param[in] *d_ja (Unsorted) Column indices
 * @param[in] *d_a (Unsorted) Values
 * @param[in,out] *d_ja_sorted Pre-allocated! On return: Sorted row Indices
 * @param[in,out] *d_a_sorted Pre-allocated! On return: Sorted values
 * @warning Requires d_ja_sorted and d_a_sorted to be preallocated
 */
void hypre_sortCSR(cusparseHandle_t cusparsehandle, HYPRE_Int n, HYPRE_Int m,
                   HYPRE_Int nnzA,
                   HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ja_sorted, HYPRE_Complex *d_a_sorted) {
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

#if (CUDART_VERSION >= 11000)

/*
 * @brief Determines the associated CudaDataType for the HYPRE_Complex typedef
 * @return Returns cuda data type corresponding with HYPRE_Complex
 *
 * @todo Should be known compile time
 * @warning Only works for Single and Double precision
 * @note Perhaps some typedefs should be added where HYPRE_Complex is typedef'd
 */
cudaDataType hypre_getCudaDataTypeComplex() 
{ 
   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;
  if(isDoublePrecision) 
  {
     return CUDA_R_64F;
  }
  if(isSinglePrecision) 
  {
     return CUDA_R_32F;
  }
  hypre_assert(false);
  return CUDA_R_64F;
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
 * @TODO Move to a separate file
 * @TODO Determine cuda types from hypre types 
 * @TODO Does HYPRE_Int type impact these functions?
 */
cusparseSpMatDescr_t hypre_CSRMatRawToCuda(HYPRE_Int n, HYPRE_Int m, HYPRE_Int nnz,
HYPRE_Int *i, HYPRE_Int *j, HYPRE_Complex *data) {

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   hypre_assert(isDoublePrecision);

   const cudaDataType data_type = hypre_getCudaDataTypeComplex();
   const cusparseIndexType_t index_type = CUSPARSE_INDEX_32I;
   const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;

   cusparseSpMatDescr_t matA;
   HYPRE_CUSPARSE_CALL(cusparseCreateCsr(&matA, n, m, nnz, i, j, data, index_type, index_type, index_base, data_type));
   return matA;
}




/*
 * @brief Uses Cusparse to calculate a sparse-matrix x sparse-matrix product in CSRS format. Supports Cusparse generic API (10.x+)
 *
 * @param[in] m Number of rows of A,C
 * @param[in] k Number of columns of B,C
 * @param[in] n Number of columns of A, number of rows of B
 * @param[in] nnzA Number of nonzeros in A
 * @param[in] *d_ia Array containing the row pointers of A
 * @param[in] *d_ja Array containing the column indices of A
 * @param[in] *d_a Array containing values of A
 * @param[in] nnzB Number of nonzeros in B
 * @param[in] *d_ib Array containing the row pointers of B
 * @param[in] *d_jb Array containing the column indices of B
 * @param[in] *d_b Array containing values of B
 * @param[out] *d_ic Array containing the row pointers of C
 * @param[out] *d_jc Array containing the column indices of C
 * @param[out] *d_c Array containing values of C
 * @warning Only supports opA=opB=CUSPARSE_OPERATION_NON_TRANSPOSE
 * @note This call now has support for the cusparse generic API function calls, as it appears that the other types of SpM x SpM calls are getting deprecated in the near future.
 * Unfortunately, as of now (2020-06-24), it appears these functions have minimal documentation, and are not very mature.
 */
HYPRE_Int
hypreDevice_CSRSpGemmCusparseGenericAPI(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                              HYPRE_Int nnzA,
                              HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                              HYPRE_Int nnzB,
                              HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                              HYPRE_Int *nnzC_out,
                              HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out)
{
   cusparseHandle_t cusparsehandle = 0;//hypre_HandleCusparseHandle(hypre_handle());
   HYPRE_CUSPARSE_CALL( cusparseCreate(&cusparsehandle) );

#if 0
   //Sort the A,B matrices. 
   //Initial tests indicate that the new API does not each row to be sorted, but
   //in case this is not true, this commented out part sorts it
   //TODO: Determine with 100% certainty if sorting is (not) required

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;
   hypre_assert(isDoublePrecision || isSinglePrecision);
   cusparseMatDescr_t descrA=0;
   cusparseMatDescr_t descrB=0;

   /* initialize cusparse library */

   /* create and setup matrix descriptor */
   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descrA) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO) );

   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descrB) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO) );

   HYPRE_Int  *d_ja_sorted, *d_jb_sorted;
   HYPRE_Complex *d_a_sorted, *d_b_sorted;

   d_a_sorted  = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   d_b_sorted  = hypre_TAlloc(HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE);
   d_ja_sorted = hypre_TAlloc(HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE);
   d_jb_sorted = hypre_TAlloc(HYPRE_Int,     nnzB, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(d_ja_sorted, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_a_sorted, d_a, HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jb_sorted, d_jb, HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_b_sorted, d_b, HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_sortCSR(cusparsehandle, m, k, nnzA,  d_ia, d_ja, d_a, d_ja_sorted, d_a_sorted);
   hypre_sortCSR(cusparsehandle, k, n, nnzB,  d_ib, d_jb, d_b, d_jb_sorted, d_b_sorted);
#endif
   //Initialize the descriptors for the mats
   cusparseSpMatDescr_t matA = hypre_CSRMatRawToCuda(m,k,nnzA,d_ia,d_ja,d_a);
   cusparseSpMatDescr_t matB = hypre_CSRMatRawToCuda(k,n,nnzB,d_ib,d_jb,d_b);
   cusparseSpMatDescr_t matC = hypre_CSRMatRawToCuda(m,n,0,NULL,NULL,NULL);
   cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
   cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

   cusparseSpGEMMDescr_t spgemmDesc;
   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_createDescr(&spgemmDesc));

   cudaDataType        computeType = CUDA_R_64F;
   HYPRE_Complex alpha = 1.0;
   HYPRE_Complex beta = 0.0;
   size_t bufferSize1;
   size_t bufferSize2;
   void *dBuffer1 = NULL;
   void *dBuffer2 = NULL;

   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));
   HYPRE_CUDA_CALL(cudaMalloc(&dBuffer1, bufferSize1));
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));


   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_compute(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));
   HYPRE_CUDA_CALL(cudaMalloc(&dBuffer2, bufferSize2));
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_compute(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));


   //TODO Investigate typing
   int64_t C_num_rows, C_num_cols, nnzC;
   HYPRE_Int  *d_ic, *d_jc;
   HYPRE_Complex  *d_c;

   HYPRE_CUSPARSE_CALL(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &nnzC));
   hypre_assert(C_num_rows == m);
   hypre_assert(C_num_cols == n);
   d_ic = hypre_TAlloc(HYPRE_Int,     C_num_rows+1, HYPRE_MEMORY_DEVICE);
   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_c  = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL(cusparseCsrSetPointers(matC, d_ic, d_jc, d_c));

   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_copy(cusparsehandle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
spgemmDesc));

   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_destroyDescr(spgemmDesc));
   HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
   HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matB));
   HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matC));
   HYPRE_CUSPARSE_CALL(cusparseDestroy(cusparsehandle));

   HYPRE_CUDA_CALL(cudaFree(dBuffer1));
   HYPRE_CUDA_CALL(cudaFree(dBuffer2));
   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out = d_c;

   return hypre_error_flag;
}



#endif

HYPRE_Int
hypreDevice_CSRSpGemmCusparse(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                              HYPRE_Int nnzA,
                              HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                              HYPRE_Int nnzB,
                              HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                              HYPRE_Int *nnzC_out,
                              HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out)
{
#if (CUDART_VERSION >= 11000)
return 
hypreDevice_CSRSpGemmCusparseGenericAPI(m, k, n,
                              nnzA,
                              d_ia, d_ja, d_a,
                              nnzB,
                              d_ib, d_jb, d_b,
                              nnzC_out, d_ic_out, d_jc_out, d_c_out);
#else
   HYPRE_Int  *d_ic, *d_jc, baseC, nnzC;
   HYPRE_Int  *d_ja_sorted, *d_jb_sorted;
   HYPRE_Complex *d_c, *d_a_sorted, *d_b_sorted;

   d_a_sorted  = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   d_b_sorted  = hypre_TAlloc(HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE);
   d_ja_sorted = hypre_TAlloc(HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE);
   d_jb_sorted = hypre_TAlloc(HYPRE_Int,     nnzB, HYPRE_MEMORY_DEVICE);

   cusparseHandle_t cusparsehandle=0;
   cusparseMatDescr_t descrA=0, descrB=0, descrC=0;

   /* initialize cusparse library */
   HYPRE_CUSPARSE_CALL( cusparseCreate(&cusparsehandle) );

   /* create and setup matrix descriptor */
   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descrA) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO) );

   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descrB) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO) );

   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descrC) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO) );

   cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
   cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   hypre_TMemcpy(d_ja_sorted, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_a_sorted, d_a, HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jb_sorted, d_jb, HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_b_sorted, d_b, HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_sortCSR(cusparsehandle, m, k, nnzA,  d_ia, d_ja, d_a, d_ja_sorted, d_a_sorted);
   hypre_sortCSR(cusparsehandle, k, n, nnzB,  d_ib, d_jb, d_b, d_jb_sorted, d_b_sorted);



   // nnzTotalDevHostPtr points to host memory
   HYPRE_Int *nnzTotalDevHostPtr = &nnzC;
   HYPRE_CUSPARSE_CALL( cusparseSetPointerMode(cusparsehandle, CUSPARSE_POINTER_MODE_HOST) );

   d_ic = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);

#if (CUDART_VERSION < 11000)
   HYPRE_CUSPARSE_CALL(
         cusparseXcsrgemmNnz(cusparsehandle, transA, transB,
                             m, n, k,
                             descrA, nnzA, d_ia, d_ja_sorted,
                             descrB, nnzB, d_ib, d_jb_sorted,
                             descrC,       d_ic, nnzTotalDevHostPtr )
         );
#else
#error UNSUPPORTED
#endif

   if (NULL != nnzTotalDevHostPtr)
   {
      nnzC = *nnzTotalDevHostPtr;
   } else
   {
      hypre_TMemcpy(&nnzC,  d_ic + m, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(&baseC, d_ic,     HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      nnzC -= baseC;
   }

   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_c  = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   if (isDoublePrecision)
   {
#if (CUDART_VERSION < 11000)
      HYPRE_CUSPARSE_CALL(
            cusparseDcsrgemm(cusparsehandle, transA, transB, m, n, k,
                             descrA, nnzA, d_a_sorted, d_ia, d_ja_sorted,
                             descrB, nnzB, d_b_sorted, d_ib, d_jb_sorted,
                             descrC,       d_c, d_ic, d_jc)
            );
#else
#error UNSUPPORTED
#endif
   } else if (isSinglePrecision)
   {
#if (CUDART_VERSION < 11000)
      HYPRE_CUSPARSE_CALL(
            cusparseScsrgemm(cusparsehandle, transA, transB, m, n, k,
                             descrA, nnzA, (float *) d_a_sorted, d_ia, d_ja_sorted,
                             descrB, nnzB, (float *) d_b_sorted, d_ib, d_jb_sorted,
                             descrC,       (float *) d_c, d_ic, d_jc)
            );
#else
#error UNSUPPORTED
#endif
   }

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   hypre_TFree(d_a_sorted,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_b_sorted,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ja_sorted, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jb_sorted, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
#endif
}

#endif /* HYPRE_USING_CUDA */
