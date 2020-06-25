/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "hypre_cuda_utils.h"
#include "csr_matrix_cuda_utils.h"

#if defined(HYPRE_USING_CUDA)



#if (CUDART_VERSION >= 11000)


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

   hypre_sortCSR(cusparsehandle, m, k, nnzA,  d_ia, d_ja_sorted, d_a_sorted);
   hypre_sortCSR(cusparsehandle, k, n, nnzB,  d_ib, d_jb_sorted, d_b_sorted);
#endif
   //Initialize the descriptors for the mats
   cusparseSpMatDescr_t matA = hypre_CSRMatRawToCuda(m,k,nnzA,d_ia,d_ja,d_a);
   cusparseSpMatDescr_t matB = hypre_CSRMatRawToCuda(k,n,nnzB,d_ib,d_jb,d_b);
   cusparseSpMatDescr_t matC = hypre_CSRMatRawToCuda(m,n,0,NULL,NULL,NULL);
   cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
   cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

   /* Create the SpGEMM Descriptor */
   cusparseSpGEMMDescr_t spgemmDesc;
   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_createDescr(&spgemmDesc));

   cudaDataType        computeType = CUDA_R_64F;
   HYPRE_Complex alpha = 1.0;
   HYPRE_Complex beta = 0.0;
   size_t bufferSize1;
   size_t bufferSize2;
   void *dBuffer1 = NULL;
   void *dBuffer2 = NULL;

   /* Do work estimation */
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));
   HYPRE_CUDA_CALL(cudaMalloc(&dBuffer1, bufferSize1));
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));


   /* Do computation */
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_compute(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));
   HYPRE_CUDA_CALL(cudaMalloc(&dBuffer2, bufferSize2));
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_compute(cusparsehandle, opA, opB, &alpha, matA, matB, &beta,  matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));


   //TODO Investigate typing
   int64_t C_num_rows, C_num_cols, nnzC;
   HYPRE_Int  *d_ic, *d_jc;
   HYPRE_Complex  *d_c;

   /* Get required information for C */
   HYPRE_CUSPARSE_CALL(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &nnzC));
   hypre_assert(C_num_rows == m);
   hypre_assert(C_num_cols == n);
   d_ic = hypre_TAlloc(HYPRE_Int,     C_num_rows+1, HYPRE_MEMORY_DEVICE);
   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_c  = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   /* Setup the required descriptor for C */
   HYPRE_CUSPARSE_CALL(cusparseCsrSetPointers(matC, d_ic, d_jc, d_c));

   /* Copy the data into C */
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_copy(cusparsehandle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
spgemmDesc));

   /* Cleanup the data */
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_destroyDescr(spgemmDesc));
   HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
   HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matB));
   HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matC));
   HYPRE_CUSPARSE_CALL(cusparseDestroy(cusparsehandle));

   HYPRE_CUDA_CALL(cudaFree(dBuffer1));
   HYPRE_CUDA_CALL(cudaFree(dBuffer2));

   /* Assign the output */
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

   /* Allocate space for sorted arrays */
   d_a_sorted  = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   d_b_sorted  = hypre_TAlloc(HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE);
   d_ja_sorted = hypre_TAlloc(HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE);
   d_jb_sorted = hypre_TAlloc(HYPRE_Int,     nnzB, HYPRE_MEMORY_DEVICE);

   cusparseHandle_t cusparsehandle=0;
   cusparseMatDescr_t descrA=0, descrB=0, descrC=0;

   /* initialize cusparse library */
   HYPRE_CUSPARSE_CALL( cusparseCreate(&cusparsehandle) );

   /* create and setup matrix descriptors */
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

   /* Copy the unsorted over as the initial "sorted" */
   hypre_TMemcpy(d_ja_sorted, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_a_sorted, d_a, HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jb_sorted, d_jb, HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_b_sorted, d_b, HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* Sort each of the CSR matrices */
   hypre_sortCSR(cusparsehandle, m, k, nnzA,  d_ia, d_ja_sorted, d_a_sorted);
   hypre_sortCSR(cusparsehandle, k, n, nnzB,  d_ib, d_jb_sorted, d_b_sorted);



   // nnzTotalDevHostPtr points to host memory
   HYPRE_Int *nnzTotalDevHostPtr = &nnzC;
   HYPRE_CUSPARSE_CALL( cusparseSetPointerMode(cusparsehandle, CUSPARSE_POINTER_MODE_HOST) );

   d_ic = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);

   HYPRE_CUSPARSE_CALL(
         cusparseXcsrgemmNnz(cusparsehandle, transA, transB,
                             m, n, k,
                             descrA, nnzA, d_ia, d_ja_sorted,
                             descrB, nnzB, d_ib, d_jb_sorted,
                             descrC,       d_ic, nnzTotalDevHostPtr )
         );

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
      HYPRE_CUSPARSE_CALL(
            cusparseDcsrgemm(cusparsehandle, transA, transB, m, n, k,
                             descrA, nnzA, d_a_sorted, d_ia, d_ja_sorted,
                             descrB, nnzB, d_b_sorted, d_ib, d_jb_sorted,
                             descrC,       d_c, d_ic, d_jc)
            );
   } else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(
            cusparseScsrgemm(cusparsehandle, transA, transB, m, n, k,
                             descrA, nnzA, (float *) d_a_sorted, d_ia, d_ja_sorted,
                             descrB, nnzB, (float *) d_b_sorted, d_ib, d_jb_sorted,
                             descrC,       (float *) d_c, d_ic, d_jc)
            );
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
