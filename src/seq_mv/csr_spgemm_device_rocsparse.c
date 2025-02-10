/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_HIP) && defined(HYPRE_USING_ROCSPARSE)

HYPRE_Int
hypreDevice_CSRSpGemmRocsparse(HYPRE_Int           m,
                               HYPRE_Int           k,
                               HYPRE_Int           n,
                               rocsparse_mat_descr descrA,
                               HYPRE_Int           nnzA,
                               HYPRE_Int          *d_ia,
                               HYPRE_Int          *d_ja,
                               HYPRE_Complex      *d_a,
                               rocsparse_mat_descr descrB,
                               HYPRE_Int           nnzB,
                               HYPRE_Int          *d_ib,
                               HYPRE_Int          *d_jb,
                               HYPRE_Complex      *d_b,
                               rocsparse_mat_descr descrC,
                               rocsparse_mat_info  infoC,
                               HYPRE_Int          *nnzC_out,
                               HYPRE_Int         **d_ic_out,
                               HYPRE_Int         **d_jc_out,
                               HYPRE_Complex     **d_c_out)
{
   HYPRE_Int  *d_ic, *d_jc, baseC, nnzC;
   HYPRE_Int  *d_ja_sorted, *d_jb_sorted;
   HYPRE_Complex *d_c, *d_a_sorted, *d_b_sorted;

   d_a_sorted  = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   d_b_sorted  = hypre_TAlloc(HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE);
   d_ja_sorted = hypre_TAlloc(HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE);
   d_jb_sorted = hypre_TAlloc(HYPRE_Int,     nnzB, HYPRE_MEMORY_DEVICE);

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   rocsparse_operation transA = rocsparse_operation_none;
   rocsparse_operation transB = rocsparse_operation_none;

   /* Copy the unsorted over as the initial "sorted" */
   hypre_TMemcpy(d_ja_sorted, d_ja, HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_a_sorted,  d_a,  HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_jb_sorted, d_jb, HYPRE_Int,     nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_b_sorted,  d_b,  HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* For rocSPARSE, the CSR SpGEMM implementation does not require the columns to be sorted! */
   /* RL: for matrices with long rows, it seemed that the sorting is still needed */
   /* VPM: Adding sorting back since it is necessary for correctness in a few cases */
#if 1
   hypre_SortCSRRocsparse(m, k, nnzA, descrA, d_ia, d_ja_sorted, d_a_sorted);
   hypre_SortCSRRocsparse(k, n, nnzB, descrB, d_ib, d_jb_sorted, d_b_sorted);
#endif

   // nnzTotalDevHostPtr points to host memory
   HYPRE_Int *nnzTotalDevHostPtr = &nnzC;
   HYPRE_ROCSPARSE_CALL( rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host) );

   d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);

   // For rocsparse, we need an extra buffer for computing the
   // csrgemmnnz and the csrgemm
   //
   // Once the buffer is allocated, we can use the same allocated
   // buffer for both the csrgemm_nnz and csrgemm
   //
   // Note that rocsparse csrgemms do: C = \alpha*A*B +\beta*D
   // So we hardcode \alpha=1, D to nothing, and pass NULL for beta
   // to indicate \beta = 0 to match the cusparse behavior.
   HYPRE_Complex alpha = 1.0;

   size_t rs_buffer_size = 0;
   void *rs_buffer;

   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrgemm_buffer_size(handle,
                                                             transA, transB,
                                                             m, n, k,
                                                             &alpha, // \alpha = 1
                                                             descrA, nnzA, d_ia, d_ja_sorted,
                                                             descrB, nnzB, d_ib, d_jb_sorted,
                                                             NULL, // \beta = 0
                                                             NULL,   0,    NULL, NULL, // D is nothing
                                                             infoC, &rs_buffer_size) );

   rs_buffer = hypre_TAlloc(char, rs_buffer_size, HYPRE_MEMORY_DEVICE);

   // Note that rocsparse csrgemms do: C = \alpha*A*B +\beta*D
   // So we hardcode \alpha=1, D to nothing, and \beta = 0
   // to match the cusparse behavior
   HYPRE_ROCSPARSE_CALL( rocsparse_csrgemm_nnz(handle, transA, transB,
                                               m, n, k,
                                               descrA, nnzA, d_ia, d_ja_sorted,
                                               descrB, nnzB, d_ib, d_jb_sorted,
                                               NULL,   0,    NULL, NULL, // D is nothing
                                               descrC,       d_ic, nnzTotalDevHostPtr,
                                               infoC, rs_buffer) );

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

   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_c  = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrgemm(handle, transA, transB,
                                                 m, n, k,
                                                 &alpha, // alpha = 1
                                                 descrA, nnzA, d_a_sorted, d_ia, d_ja_sorted,
                                                 descrB, nnzB, d_b_sorted, d_ib, d_jb_sorted,
                                                 NULL, // beta = 0
                                                 NULL,   0,    NULL,       NULL, NULL, // D is nothing
                                                 descrC,       d_c, d_ic, d_jc,
                                                 infoC, rs_buffer) );

   // Free up the memory needed by rocsparse
   hypre_TFree(rs_buffer, HYPRE_MEMORY_DEVICE);

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

#endif // defined(HYPRE_USING_HIP) && defined(HYPRE_USING_ROCSPARSE)
