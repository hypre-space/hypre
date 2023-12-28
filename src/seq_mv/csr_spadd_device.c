/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_GPU)

HYPRE_Int
hypreDevice_CSRSpAdd( HYPRE_Complex     alpha,
                      hypre_CSRMatrix  *A,
                      HYPRE_Complex     beta,
                      hypre_CSRMatrix  *B,
                      hypre_CSRMatrix **C_ptr )
{
   HYPRE_Int        nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int        ncols  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int        nnzA    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int        nnzB    = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Complex   *A_a      = hypre_CSRMatrixData(A);
   HYPRE_Int       *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int       *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Complex   *B_a      = hypre_CSRMatrixData(B);
   HYPRE_Int       *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int       *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Complex   *C_a      = NULL;
   HYPRE_Int       *C_i      = NULL;
   HYPRE_Int       *C_j      = NULL;
   HYPRE_Int        nnzC     = 0;
   hypre_CSRMatrix *C        = NULL;

   *C_ptr = C = hypre_CSRMatrixCreate(nrows, ncols, 0);
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   /* trivial case */
   if (nnzA == 0 && nnzB == 0)
   {
      C_i = hypre_CTAlloc(HYPRE_Int,     nrows + 1, HYPRE_MEMORY_DEVICE);
      C_j = hypre_CTAlloc(HYPRE_Int,     0,         HYPRE_MEMORY_DEVICE);
      C_a = hypre_CTAlloc(HYPRE_Complex, 0,         HYPRE_MEMORY_DEVICE);
      nnzC = 0;
   }
   else if (nnzA == 0)
   {
      C_i = hypre_TAlloc(HYPRE_Int,     nrows + 1, HYPRE_MEMORY_DEVICE);
      C_j = hypre_TAlloc(HYPRE_Int,     nnzB,      HYPRE_MEMORY_DEVICE);
      C_a = hypre_TAlloc(HYPRE_Complex, nnzB,      HYPRE_MEMORY_DEVICE);
      nnzC = nnzB;
      hypre_TMemcpy(C_i, B_i, HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(C_j, B_j, HYPRE_Int, nnzB,      HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypreDevice_ComplexScalen(B_a, nnzB, C_a, beta);
   }
   else if (nnzB == 0)
   {
      C_i = hypre_TAlloc(HYPRE_Int,     nrows + 1, HYPRE_MEMORY_DEVICE);
      C_j = hypre_TAlloc(HYPRE_Int,     nnzA,      HYPRE_MEMORY_DEVICE);
      C_a = hypre_TAlloc(HYPRE_Complex, nnzA,      HYPRE_MEMORY_DEVICE);
      nnzC = nnzA;
      hypre_TMemcpy(C_i, A_i, HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(C_j, A_j, HYPRE_Int, nnzA,      HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypreDevice_ComplexScalen(A_a, nnzA, C_a, alpha);
   }
   else if (hypre_HandleSpAddUseVendor(hypre_handle()))
   {
#if defined(HYPRE_USING_CUSPARSE)
      hypre_CSRMatrixSortRowOutOfPlace(A);
      hypre_CSRMatrixSortRowOutOfPlace(B);
      hypreDevice_CSRSpAddCusparse(nrows, ncols, nnzA, nnzB,
                                   A_i, hypre_CSRMatrixSortedJ(A), alpha, hypre_CSRMatrixSortedData(A),
                                   hypre_CSRMatrixGPUMatDescr(A),
                                   B_i, hypre_CSRMatrixSortedJ(B), beta, hypre_CSRMatrixSortedData(B),
                                   hypre_CSRMatrixGPUMatDescr(B),
                                   &nnzC, &C_i, &C_j, &C_a, hypre_CSRMatrixGPUMatDescr(C));
#elif defined(HYPRE_USING_ROCSPARSE)
#elif defined(HYPRE_USING_ONEMKLSPARSE)
#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Attempting to use device sparse matrix library for SpAdd without having compiled support for it!\n");
#endif
   }
   else
   {
      if (1 == hypre_HandleSpAddAlgorithm(hypre_handle()))
      {
         hypreDevice_CSRSpAdd1(nrows, ncols, nnzA, nnzB,
                               A_i, A_j, alpha, A_a,
                               B_i, B_j, beta, B_a,
                               &nnzC, &C_i, &C_j, &C_a);
      }
      else
      {
         hypreDevice_CSRSpAdd2(nrows, nrows, ncols, nnzA, nnzB,
                               A_i, A_j, alpha, A_a, NULL,
                               B_i, B_j, beta, B_a, NULL,
                               NULL, &nnzC, &C_i, &C_j, &C_a);
      }
   }

   hypre_CSRMatrixI(C)           = C_i;
   hypre_CSRMatrixJ(C)           = C_j;
   hypre_CSRMatrixData(C)        = C_a;
   hypre_CSRMatrixNumNonzeros(C) = nnzC;

   return hypre_error_flag;
}
__global__ void
hypreGPUKernel_II( hypre_DeviceItem &item,
                   HYPRE_Int         n,
                   HYPRE_Complex     alpha,
                   HYPRE_Complex     beta,
                   HYPRE_Int        *ja,
                   HYPRE_Complex    *aa )
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   const HYPRE_Int nnz = n << 1;
   if (i >= nnz) { return; }
   /* 1 if odd, 0 if even */
   const HYPRE_Int s = i & 1;
   ja[i] = (i >> 1) + s * n;
   if (aa) { aa[i] = alpha + s * (beta - alpha); }
}

HYPRE_Int
hypreDevice_CSRSpAdd1( HYPRE_Int       nrows,
                       HYPRE_Int       ncols,
                       HYPRE_Int       nnzA,
                       HYPRE_Int       nnzB,
                       HYPRE_Int      *d_ia,
                       HYPRE_Int      *d_ja,
                       HYPRE_Complex   alpha,
                       HYPRE_Complex  *d_aa,
                       HYPRE_Int      *d_ib,
                       HYPRE_Int      *d_jb,
                       HYPRE_Complex   beta,
                       HYPRE_Complex  *d_ab,
                       HYPRE_Int      *nnzC_out,
                       HYPRE_Int     **d_ic_out,
                       HYPRE_Int     **d_jc_out,
                       HYPRE_Complex **d_ac_out)
{
   hypre_CSRMatrix *A = hypre_CSRMatrixCreate(nrows, ncols, nnzA);
   hypre_CSRMatrixData(A) = d_aa;
   hypre_CSRMatrixI(A) = d_ia;
   hypre_CSRMatrixJ(A) = d_ja;
   hypre_CSRMatrixMemoryLocation(A) = HYPRE_MEMORY_DEVICE;

   hypre_CSRMatrix *B = hypre_CSRMatrixCreate(nrows, ncols, nnzB);
   hypre_CSRMatrixData(B) = d_ab;
   hypre_CSRMatrixI(B) = d_ib;
   hypre_CSRMatrixJ(B) = d_jb;
   hypre_CSRMatrixMemoryLocation(B) = HYPRE_MEMORY_DEVICE;

   hypre_CSRMatrix *AB = hypre_CSRMatrixStack2Device(A, B);

   HYPRE_Int nnzE = 2 * nrows;
   hypre_CSRMatrix *E = hypre_CSRMatrixCreate(nrows, 2 * nrows, nnzE);
   HYPRE_Int *ie = hypre_TAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *je = hypre_TAlloc(HYPRE_Int, nnzE, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *ae = NULL;
   if ( hypre_HandleSpgemmUseVendor(hypre_handle()) || alpha != 1.0 || beta != 1.0)
   {
      ae = hypre_TAlloc(HYPRE_Complex, nnzE, HYPRE_MEMORY_DEVICE);
   }
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nnzE, "thread", bDim);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_II, gDim, bDim, nrows, alpha, beta, je, ae );
   HYPRE_THRUST_CALL( transform,
                      thrust::counting_iterator<HYPRE_Int>(0),
                      thrust::counting_iterator<HYPRE_Int>(nrows + 1),
                      ie,
                      _1 * 2);
   hypre_CSRMatrixData(E) = ae;
   hypre_CSRMatrixI(E) = ie;
   hypre_CSRMatrixJ(E) = je;
   hypre_CSRMatrixMemoryLocation(E) = HYPRE_MEMORY_DEVICE;

   hypre_CSRMatrix *C = hypre_CSRMatrixMultiplyDevice(E, AB);

   *nnzC_out = hypre_CSRMatrixNumNonzeros(C);
   *d_ic_out = hypre_CSRMatrixI(C);
   *d_jc_out = hypre_CSRMatrixJ(C);
   *d_ac_out = hypre_CSRMatrixData(C);

   hypre_CSRMatrixI(C) = NULL;
   hypre_CSRMatrixJ(C) = NULL;
   hypre_CSRMatrixData(C) = NULL;

   hypre_TFree(A, HYPRE_MEMORY_HOST);
   hypre_TFree(B, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixDestroy(E);
   hypre_CSRMatrixDestroy(AB);
   hypre_CSRMatrixDestroy(C);

   return hypre_error_flag;
}

/* This function effectively does (in Matlab notation)
 *              C := alpha * A(:, a_colmap)
 *              C(num_b, :) += beta * B(:, b_colmap)
 *
 * if num_b != NULL: A is ma x n and B is mb x n. len(num_b) == mb.
 *                   All numbers in num_b must be in [0,...,ma-1]
 *
 * if num_b == NULL: C = alpha * A + beta * B. ma == mb
 *
 * if d_ja_map/d_jb_map == NULL, it is [0:n)
 */
HYPRE_Int
hypreDevice_CSRSpAdd2( HYPRE_Int       ma, /* num of rows of A */
                       HYPRE_Int       mb, /* num of rows of B */
                       HYPRE_Int       n,  /* not used actually */
                       HYPRE_Int       nnzA,
                       HYPRE_Int       nnzB,
                       HYPRE_Int      *d_ia,
                       HYPRE_Int      *d_ja,
                       HYPRE_Complex   alpha,
                       HYPRE_Complex  *d_aa,
                       HYPRE_Int      *d_ja_map,
                       HYPRE_Int      *d_ib,
                       HYPRE_Int      *d_jb,
                       HYPRE_Complex   beta,
                       HYPRE_Complex  *d_ab,
                       HYPRE_Int      *d_jb_map,
                       HYPRE_Int      *d_num_b,
                       HYPRE_Int      *nnzC_out,
                       HYPRE_Int     **d_ic_out,
                       HYPRE_Int     **d_jc_out,
                       HYPRE_Complex **d_ac_out)
{
   /* trivial case */
   if (nnzA == 0 && nnzB == 0)
   {
      *d_ic_out = hypre_CTAlloc(HYPRE_Int, ma + 1, HYPRE_MEMORY_DEVICE);
      *d_jc_out = hypre_CTAlloc(HYPRE_Int,      0, HYPRE_MEMORY_DEVICE);
      *d_ac_out = hypre_CTAlloc(HYPRE_Complex,  0, HYPRE_MEMORY_DEVICE);
      *nnzC_out = 0;

      return hypre_error_flag;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPADD] -= hypre_MPI_Wtime();
#endif

   /* expansion size */
   HYPRE_Int nnzT = nnzA + nnzB, nnzC;
   HYPRE_Int *d_it, *d_jt, *d_it_cp, *d_jt_cp, *d_ic, *d_jc;
   HYPRE_Complex *d_at, *d_at_cp, *d_ac;

   /* some trick here for memory alignment. maybe not worth it at all */
   HYPRE_Int align = 32;
   HYPRE_Int nnzT2 = (nnzT + align - 1) / align * align;
   char *work_mem = hypre_TAlloc(char, (4 * sizeof(HYPRE_Int) + 2 * sizeof(HYPRE_Complex)) * nnzT2,
                                 HYPRE_MEMORY_DEVICE);
   char *work_mem_saved = work_mem;

   //d_it = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   //d_jt = hypre_TAlloc(HYPRE_Int, nnzT, HYPRE_MEMORY_DEVICE);
   //d_at = hypre_TAlloc(HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE);
   d_it = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_jt = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_at = (HYPRE_Complex *) work_mem;
   work_mem += sizeof(HYPRE_Complex) * nnzT2;

   /* expansion: j */
   if (d_ja_map)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather(d_ja, d_ja + nnzA, d_ja_map, d_jt);
#else
      HYPRE_THRUST_CALL(gather, d_ja, d_ja + nnzA, d_ja_map, d_jt);
#endif
   }
   else
   {
      hypre_TMemcpy(d_jt, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }
   if (d_jb_map)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_gather(d_jb, d_jb + nnzB, d_jb_map, d_jt + nnzA);
#else
      HYPRE_THRUST_CALL(gather, d_jb, d_jb + nnzB, d_jb_map, d_jt + nnzA);
#endif
   }
   else
   {
      hypre_TMemcpy(d_jt + nnzA, d_jb, HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }

   /* expansion: a */
   if (alpha == 1.0)
   {
      hypre_TMemcpy(d_at, d_aa, HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypreDevice_ComplexScalen( d_aa, nnzA, d_at, alpha );
   }

   if (beta == 1.0)
   {
      hypre_TMemcpy(d_at + nnzA, d_ab, HYPRE_Complex, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      hypreDevice_ComplexScalen( d_ab, nnzB, d_at + nnzA, beta );
   }

   /* expansion: i */
   hypreDevice_CsrRowPtrsToIndices_v2(ma, nnzA, d_ia, d_it);
   if (d_num_b || mb <= 0)
   {
      hypreDevice_CsrRowPtrsToIndicesWithRowNum(mb, nnzB, d_ib, d_num_b, d_it + nnzA);
   }
   else
   {
      hypre_assert(ma == mb);
      hypreDevice_CsrRowPtrsToIndices_v2(mb, nnzB, d_ib, d_it + nnzA);
   }

   /* make copy of (it, jt, at), since reduce cannot be done in-place */
   //d_it_cp = hypre_TAlloc(HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE);
   //d_jt_cp = hypre_TAlloc(HYPRE_Int,     nnzT, HYPRE_MEMORY_DEVICE);
   //d_at_cp = hypre_TAlloc(HYPRE_Complex, nnzT, HYPRE_MEMORY_DEVICE);
   d_it_cp = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_jt_cp = (HYPRE_Int *) work_mem;
   work_mem += sizeof(HYPRE_Int) * nnzT2;
   d_at_cp = (HYPRE_Complex *) work_mem;
   work_mem += sizeof(HYPRE_Complex) * nnzT2;

   hypre_assert( (size_t) (work_mem - work_mem_saved) == (4 * sizeof(HYPRE_Int) + 2 * sizeof(
                                                             HYPRE_Complex)) * ((size_t)nnzT2) );

   /* sort: lexicographical order (row, col): hypreDevice_StableSortByTupleKey */
   hypreDevice_StableSortByTupleKey(nnzT, d_it, d_jt, d_at, 0);

   /* compress */
   /* returns end: so nnz = end - start */
   nnzC = hypreDevice_ReduceByTupleKey(nnzT, d_it, d_jt, d_at, d_it_cp, d_jt_cp, d_at_cp);

   /* allocate final C */
   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_ac = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(d_jc, d_jt_cp, HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(d_ac, d_at_cp, HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(ma, nnzC, d_it_cp);

#ifdef HYPRE_DEBUG
   HYPRE_Int tmp_nnzC;
   hypre_TMemcpy(&tmp_nnzC, &d_ic[ma], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_assert(nnzC == tmp_nnzC);
#endif

   /*
   hypre_TFree(d_it,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jt,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_at,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_it_cp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jt_cp, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_at_cp, HYPRE_MEMORY_DEVICE);
   */
   hypre_TFree(work_mem_saved, HYPRE_MEMORY_DEVICE);

   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPADD] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)
HYPRE_Int
hypreDevice_CSRSpAddCusparse(HYPRE_Int          nrows,
                             HYPRE_Int          ncols,
                             HYPRE_Int          nnzA,
                             HYPRE_Int          nnzB,
                             HYPRE_Int         *d_ia,
                             HYPRE_Int         *d_ja,
                             HYPRE_Complex      alpha,
                             HYPRE_Complex     *d_aa,
                             cusparseMatDescr_t descrA,
                             HYPRE_Int         *d_ib,
                             HYPRE_Int         *d_jb,
                             HYPRE_Complex      beta,
                             HYPRE_Complex     *d_ab,
                             cusparseMatDescr_t descrB,
                             HYPRE_Int         *nnzC_out,
                             HYPRE_Int        **d_ic_out,
                             HYPRE_Int        **d_jc_out,
                             HYPRE_Complex    **d_ac_out,
                             cusparseMatDescr_t descrC)
{
   HYPRE_Int     *d_ic = NULL, *d_jc = NULL;
   HYPRE_Complex *d_ac = NULL;
   HYPRE_Int      nnzC = 0;
   size_t         bufferSizeInBytes;
   hypre_int      nnzTotalDevHostPtr;
   char          *buffer = NULL;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   d_ic = hypre_TAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);

   hypre_cusparse_csrgeam2_bufferSizeExt(handle, nrows, ncols,
                                         &alpha,
                                         descrA, nnzA, d_aa, d_ia, d_ja,
                                         &beta,
                                         descrB, nnzB, d_ab, d_ib, d_jb,
                                         descrC,       d_ac, d_ic, d_jc,
                                         &bufferSizeInBytes);

   buffer = hypre_TAlloc(char, bufferSizeInBytes, HYPRE_MEMORY_DEVICE);

   cusparseXcsrgeam2Nnz(handle, nrows, ncols,
                        descrA, nnzA, d_ia, d_ja,
                        descrB, nnzB, d_ib, d_jb,
                        descrC, d_ic, &nnzTotalDevHostPtr,
                        buffer);

   nnzC = nnzTotalDevHostPtr;

   d_jc = hypre_TAlloc(HYPRE_Int,     nnzC, HYPRE_MEMORY_DEVICE);
   d_ac = hypre_TAlloc(HYPRE_Complex, nnzC, HYPRE_MEMORY_DEVICE);

   hypre_cusparse_csrgeam2(handle, nrows, ncols,
                           &alpha,
                           descrA, nnzA, d_aa, d_ia, d_ja,
                           &beta,
                           descrB, nnzB, d_ab, d_ib, d_jb,
                           descrC,       d_ac, d_ic, d_jc,
                           buffer);

   hypre_TFree(buffer, HYPRE_MEMORY_DEVICE);

   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

   return hypre_error_flag;
}
#endif

#if defined(HYPRE_USING_CUSPARSE)
HYPRE_Int
hypreDevice_CSRSpAddRocsparse(HYPRE_Int nrows,
                              HYPRE_Int ncols,
                              HYPRE_Int nnzA,
                              HYPRE_Int nnzB,
                              HYPRE_Int *d_ia,
                              HYPRE_Int *d_ja,
                              HYPRE_Complex alpha,
                              HYPRE_Complex *d_aa,
                              HYPRE_Int *d_ib,
                              HYPRE_Int *d_jb,
                              HYPRE_Complex beta,
                              HYPRE_Complex *d_ab,
                              HYPRE_Int *nnzC_out,
                              HYPRE_Int **d_ic_out,
                              HYPRE_Int **d_jc_out,
                              HYPRE_Complex **d_ac_out)
{
   return hypre_error_flag;
}
#endif

#endif // defined(HYPRE_USING_GPU)
