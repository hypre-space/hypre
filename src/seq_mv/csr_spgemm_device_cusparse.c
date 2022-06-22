/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_CUSPARSE)

HYPRE_Int
hypreDevice_CSRSpGemmCusparse(HYPRE_Int          m,
                              HYPRE_Int          k,
                              HYPRE_Int          n,
                              cusparseMatDescr_t descr_A,
                              HYPRE_Int          nnzA,
                              HYPRE_Int         *d_ia,
                              HYPRE_Int         *d_ja,
                              HYPRE_Complex     *d_a,
                              cusparseMatDescr_t descr_B,
                              HYPRE_Int          nnzB,
                              HYPRE_Int         *d_ib,
                              HYPRE_Int         *d_jb,
                              HYPRE_Complex     *d_b,
                              cusparseMatDescr_t descr_C,
                              HYPRE_Int         *nnzC_out,
                              HYPRE_Int        **d_ic_out,
                              HYPRE_Int        **d_jc_out,
                              HYPRE_Complex    **d_c_out)
{
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   hypreDevice_CSRSpGemmCusparseGenericAPI(m, k, n,
                                           nnzA, d_ia, d_ja, d_a,
                                           nnzB, d_ib, d_jb, d_b,
                                           nnzC_out, d_ic_out, d_jc_out, d_c_out);
#else
   hypreDevice_CSRSpGemmCusparseOldAPI(m, k, n,
                                       descr_A, nnzA, d_ia, d_ja, d_a,
                                       descr_B, nnzB, d_ib, d_jb, d_b,
                                       descr_C, nnzC_out, d_ic_out, d_jc_out, d_c_out);
#endif
   return hypre_error_flag;
}

#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

/*
 * @brief Uses Cusparse to calculate a sparse-matrix x sparse-matrix product in CSRS format. Supports Cusparse generic API (11+)
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
 * @param[out] *nnzC_out Pointer to address with number of nonzeros in C
 * @param[out] *d_ic_out Array containing the row pointers of C
 * @param[out] *d_jc_out Array containing the column indices of C
 * @param[out] *d_c_out Array containing values of C
 */

HYPRE_Int
hypreDevice_CSRSpGemmCusparseGenericAPI(HYPRE_Int       m,
                                        HYPRE_Int       k,
                                        HYPRE_Int       n,
                                        HYPRE_Int       nnzA,
                                        HYPRE_Int      *d_ia,
                                        HYPRE_Int      *d_ja,
                                        HYPRE_Complex  *d_a,
                                        HYPRE_Int       nnzB,
                                        HYPRE_Int      *d_ib,
                                        HYPRE_Int      *d_jb,
                                        HYPRE_Complex  *d_b,
                                        HYPRE_Int      *nnzC_out,
                                        HYPRE_Int     **d_ic_out,
                                        HYPRE_Int     **d_jc_out,
                                        HYPRE_Complex **d_c_out)
{
   cusparseHandle_t cusparsehandle = hypre_HandleCusparseHandle(hypre_handle());

   //Initialize the descriptors for the mats
   cusparseSpMatDescr_t matA = hypre_CSRMatrixToCusparseSpMat_core(m, k, 0, nnzA, d_ia, d_ja, d_a);
   cusparseSpMatDescr_t matB = hypre_CSRMatrixToCusparseSpMat_core(k, n, 0, nnzB, d_ib, d_jb, d_b);
   cusparseSpMatDescr_t matC = hypre_CSRMatrixToCusparseSpMat_core(m, n, 0, 0,    NULL, NULL, NULL);
   cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
   cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

   /* Create the SpGEMM Descriptor */
   cusparseSpGEMMDescr_t spgemmDesc;
   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_createDescr(&spgemmDesc) );

   cudaDataType computeType = hypre_HYPREComplexToCudaDataType();
   HYPRE_Complex alpha = 1.0;
   HYPRE_Complex beta = 0.0;
   size_t bufferSize1;
   size_t bufferSize2;
   void *dBuffer1 = NULL;
   void *dBuffer2 = NULL;

#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1, t2;
#endif

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   t1 = hypre_MPI_Wtime();
#endif

   /* Do work estimation */
   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB,
                                                      &alpha, matA, matB, &beta, matC,
                                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, NULL) );
   dBuffer1 = hypre_TAlloc(char, bufferSize1, HYPRE_MEMORY_DEVICE);

   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB,
                                                      &alpha, matA, matB, &beta, matC,
                                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, dBuffer1) );

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_printf("WorkEst %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
   t1 = hypre_MPI_Wtime();
#endif

   /* Do computation */
   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_compute(cusparsehandle, opA, opB,
                                               &alpha, matA, matB, &beta, matC,
                                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, NULL) );

   dBuffer2  = hypre_TAlloc(char, bufferSize2, HYPRE_MEMORY_DEVICE);

   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_compute(cusparsehandle, opA, opB,
                                               &alpha, matA, matB, &beta, matC,
                                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, dBuffer2) );

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_printf("Compute %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
   t1 = hypre_MPI_Wtime();
#endif

   /* Required by cusparse api (as of 11) to be int64_t */
   int64_t C_num_rows, C_num_cols, nnzC;
   HYPRE_Int *d_ic, *d_jc;
   HYPRE_Complex *d_c;

   /* Get required information for C */
   HYPRE_CUSPARSE_CALL( cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &nnzC) );

   hypre_assert(C_num_rows == m);
   hypre_assert(C_num_cols == n);

   d_ic = hypre_TAlloc(HYPRE_Int,     C_num_rows + 1, HYPRE_MEMORY_DEVICE);
   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC,         HYPRE_MEMORY_DEVICE);
   d_c  = hypre_TAlloc(HYPRE_Complex, nnzC,         HYPRE_MEMORY_DEVICE);

   /* Setup the required descriptor for C */
   HYPRE_CUSPARSE_CALL(cusparseCsrSetPointers(matC, d_ic, d_jc, d_c));

   /* Copy the data into C */
   HYPRE_CUSPARSE_CALL(cusparseSpGEMM_copy( cusparsehandle, opA, opB,
                                            &alpha, matA, matB, &beta, matC,
                                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc) );

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_printf("Copy %f\n", t2);
#endif

   /* Cleanup the data */
   HYPRE_CUSPARSE_CALL( cusparseSpGEMM_destroyDescr(spgemmDesc) );
   HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matA) );
   HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matB) );
   HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matC) );

   hypre_TFree(dBuffer1, HYPRE_MEMORY_DEVICE);
   hypre_TFree(dBuffer2, HYPRE_MEMORY_DEVICE);

   /* Assign the output */
   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out = d_c;

   return hypre_error_flag;
}

#else

HYPRE_Int
hypreDevice_CSRSpGemmCusparseOldAPI(HYPRE_Int          m,
                                    HYPRE_Int          k,
                                    HYPRE_Int          n,
                                    cusparseMatDescr_t descr_A,
                                    HYPRE_Int          nnzA,
                                    HYPRE_Int         *d_ia,
                                    HYPRE_Int         *d_ja,
                                    HYPRE_Complex     *d_a,
                                    cusparseMatDescr_t descr_B,
                                    HYPRE_Int          nnzB,
                                    HYPRE_Int         *d_ib,
                                    HYPRE_Int         *d_jb,
                                    HYPRE_Complex     *d_b,
                                    cusparseMatDescr_t descr_C,
                                    HYPRE_Int         *nnzC_out,
                                    HYPRE_Int        **d_ic_out,
                                    HYPRE_Int        **d_jc_out,
                                    HYPRE_Complex    **d_c_out)
{
   HYPRE_Int  *d_ic, *d_jc, baseC, nnzC;
   HYPRE_Int  *d_ja_sorted, *d_jb_sorted;
   HYPRE_Complex *d_c, *d_a_sorted, *d_b_sorted;

#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1, t2;
#endif

#ifdef HYPRE_SPGEMM_TIMING
   t1 = hypre_MPI_Wtime();
#endif

   /* Allocate space for sorted arrays */
   d_a_sorted  = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   d_b_sorted  = hypre_TAlloc(HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE);
   d_ja_sorted = hypre_TAlloc(HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE);
   d_jb_sorted = hypre_TAlloc(HYPRE_Int,     nnzB, HYPRE_MEMORY_DEVICE);

   cusparseHandle_t cusparsehandle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
   cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

   /* Copy the unsorted over as the initial "sorted" */
   hypre_TMemcpy(d_ja_sorted, d_ja, HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_a_sorted,  d_a,  HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jb_sorted, d_jb, HYPRE_Int,     nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_b_sorted,  d_b,  HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* Sort each of the CSR matrices */
   hypre_SortCSRCusparse(m, k, nnzA, descr_A, d_ia, d_ja_sorted, d_a_sorted);
   hypre_SortCSRCusparse(k, n, nnzB, descr_B, d_ib, d_jb_sorted, d_b_sorted);

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_printf("sort %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
   t1 = hypre_MPI_Wtime();
#endif

   // nnzTotalDevHostPtr points to host memory
   HYPRE_Int *nnzTotalDevHostPtr = &nnzC;
   HYPRE_CUSPARSE_CALL( cusparseSetPointerMode(cusparsehandle, CUSPARSE_POINTER_MODE_HOST) );

   d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);

   HYPRE_CUSPARSE_CALL( cusparseXcsrgemmNnz(cusparsehandle, transA, transB,
                                            m, n, k,
                                            descr_A, nnzA, d_ia, d_ja_sorted,
                                            descr_B, nnzB, d_ib, d_jb_sorted,
                                            descr_C,       d_ic, nnzTotalDevHostPtr ) );

   /* RL: this if is always true (code copied from cusparse manual */
   if (NULL != nnzTotalDevHostPtr)
   {
      nnzC = *nnzTotalDevHostPtr;
   }
   else
   {
      hypre_TMemcpy(&nnzC,  d_ic + m, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(&baseC, d_ic,     HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      nnzC -= baseC;
   }

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_printf("csrgemmNnz %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
   t1 = hypre_MPI_Wtime();
#endif

   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_c  = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   HYPRE_CUSPARSE_CALL( hypre_cusparse_csrgemm(cusparsehandle, transA, transB, m, n, k,
                                               descr_A, nnzA, d_a_sorted, d_ia, d_ja_sorted,
                                               descr_B, nnzB, d_b_sorted, d_ib, d_jb_sorted,
                                               descr_C,       d_c, d_ic, d_jc) );

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   hypre_printf("csrgemm %f\n", t2);
#endif

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   hypre_TFree(d_a_sorted,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_b_sorted,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ja_sorted, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jb_sorted, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION */
#endif /* #if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_CUSPARSE) */

