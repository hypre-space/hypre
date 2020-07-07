/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA)

HYPRE_Int
hypreDevice_CSRSpGemmCusparse(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                              HYPRE_Int nnzA,
                              HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                              HYPRE_Int nnzB,
                              HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                              HYPRE_Int *nnzC_out,
                              HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out)
{
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

   // Sort A
   hypre_TMemcpy(d_ja_sorted, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;
   HYPRE_Int *P = NULL;
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort_bufferSizeExt(cusparsehandle, m, k, nnzA, d_ia, d_ja_sorted, &pBufferSizeInBytes) );
   pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);
   P       = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL( cusparseCreateIdentityPermutation(cusparsehandle, nnzA, P) );
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort(cusparsehandle, m, k, nnzA, descrA, d_ia, d_ja_sorted, P, pBuffer) );
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseDgthr(cusparsehandle, nnzA, d_a, d_a_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseSgthr(cusparsehandle, nnzA, (float *) d_a, (float *) d_a_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }
   hypre_TFree(pBuffer, HYPRE_MEMORY_DEVICE);
   hypre_TFree(P, HYPRE_MEMORY_DEVICE);

   // Sort B
   hypre_TMemcpy(d_jb_sorted, d_jb, HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort_bufferSizeExt(cusparsehandle, k, n, nnzB, d_ib, d_jb_sorted, &pBufferSizeInBytes) );
   pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);
   P       = hypre_TAlloc(HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL( cusparseCreateIdentityPermutation(cusparsehandle, nnzB, P) );
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort(cusparsehandle, k, n, nnzB, descrB, d_ib, d_jb_sorted, P, pBuffer) );
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseDgthr(cusparsehandle, nnzB, d_b, d_b_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseSgthr(cusparsehandle, nnzB, (float *) d_b, (float *) d_b_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }

   hypre_TFree(pBuffer, HYPRE_MEMORY_DEVICE);
   hypre_TFree(P, HYPRE_MEMORY_DEVICE);

   // nnzTotalDevHostPtr points to host memory
   HYPRE_Int *nnzTotalDevHostPtr = &nnzC;
   HYPRE_CUSPARSE_CALL( cusparseSetPointerMode(cusparsehandle, CUSPARSE_POINTER_MODE_HOST) );

   d_ic = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);

   //
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
}

#endif /* HYPRE_USING_CUDA */
