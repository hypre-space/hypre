/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUSPARSE)

HYPRE_Int
hypreDevice_CSRSpTransCusparse(HYPRE_Int   m,        HYPRE_Int   n,        HYPRE_Int       nnzA,
                               HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_aa,
                               HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out,
                               HYPRE_Int   want_data)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] -= hypre_MPI_Wtime();
#endif

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseAction_t action = want_data ? CUSPARSE_ACTION_NUMERIC : CUSPARSE_ACTION_SYMBOLIC;
   HYPRE_Complex *csc_a;
   if (want_data)
   {
      csc_a = hypre_TAlloc(HYPRE_Complex, nnzA,  HYPRE_MEMORY_DEVICE);
   }
   else
   {
      csc_a = NULL;
      d_aa = NULL;
   }
   HYPRE_Int *csc_j = hypre_TAlloc(HYPRE_Int, nnzA,  HYPRE_MEMORY_DEVICE);
   HYPRE_Int *csc_i = hypre_TAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_DEVICE);

#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   size_t bufferSize = 0;
   const cudaDataType data_type = hypre_HYPREComplexToCudaDataType();

   HYPRE_CUSPARSE_CALL( cusparseCsr2cscEx2_bufferSize(handle,
                                                      m, n, nnzA,
                                                      d_aa, d_ia, d_ja,
                                                      csc_a, csc_i, csc_j,
                                                      data_type,
                                                      action,
                                                      CUSPARSE_INDEX_BASE_ZERO,
                                                      CUSPARSE_CSR2CSC_ALG1,
                                                      &bufferSize) );

   char *dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);

   HYPRE_CUSPARSE_CALL( cusparseCsr2cscEx2(handle,
                                           m, n, nnzA,
                                           d_aa, d_ia, d_ja,
                                           csc_a, csc_i, csc_j,
                                           data_type,
                                           action,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUSPARSE_CSR2CSC_ALG1,
                                           dBuffer) );

   hypre_TFree(dBuffer, HYPRE_MEMORY_DEVICE);
#else
   HYPRE_CUSPARSE_CALL( hypre_cusparse_csr2csc(handle,
                                               m, n, nnzA,
                                               d_aa, d_ia, d_ja,
                                               csc_a, csc_j, csc_i,
                                               action,
                                               CUSPARSE_INDEX_BASE_ZERO) );
#endif /* #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION */

   *d_ic_out = csc_i;
   *d_jc_out = csc_j;
   *d_ac_out = csc_a;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUSPARSE)


#if defined(HYPRE_USING_ROCSPARSE)
HYPRE_Int
hypreDevice_CSRSpTransRocsparse(HYPRE_Int   m,        HYPRE_Int   n,        HYPRE_Int       nnzA,
                                HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_aa,
                                HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out,
                                HYPRE_Int   want_data)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] -= hypre_MPI_Wtime();
#endif

   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());
   rocsparse_action action = want_data ? rocsparse_action_numeric : rocsparse_action_symbolic;

   HYPRE_Complex *csc_a;
   if (want_data)
   {
      csc_a = hypre_TAlloc(HYPRE_Complex, nnzA,  HYPRE_MEMORY_DEVICE);
   }
   else
   {
      csc_a = NULL;
      d_aa = NULL;
   }
   HYPRE_Int *csc_j = hypre_TAlloc(HYPRE_Int, nnzA,  HYPRE_MEMORY_DEVICE);
   HYPRE_Int *csc_i = hypre_TAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_DEVICE);

   size_t buffer_size = 0;
   HYPRE_ROCSPARSE_CALL( rocsparse_csr2csc_buffer_size(handle,
                                                       m, n, nnzA,
                                                       csc_i, csc_j,
                                                       action,
                                                       &buffer_size) );

   void * buffer;
   buffer = hypre_TAlloc(char, buffer_size, HYPRE_MEMORY_DEVICE);

   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csr2csc(handle,
                                                 m, n, nnzA,
                                                 d_aa, d_ia, d_ja,
                                                 csc_a, csc_j, csc_i,
                                                 action,
                                                 rocsparse_index_base_zero,
                                                 buffer) );

   hypre_TFree(buffer, HYPRE_MEMORY_DEVICE);

   *d_ic_out = csc_i;
   *d_jc_out = csc_j;
   *d_ac_out = csc_a;

#ifdef HYPRE_PROFILE
   hypre_SyncCudaDevice(hypre_handle())
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_ROCSPARSE)

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

HYPRE_Int
hypreDevice_CSRSpTrans(HYPRE_Int   m,        HYPRE_Int   n,        HYPRE_Int       nnzA,
                       HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_aa,
                       HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out,
                       HYPRE_Int   want_data)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int *d_jt, *d_it, *d_pm, *d_ic, *d_jc;
   HYPRE_Complex *d_ac = NULL;
   HYPRE_Int *mem_work = hypre_TAlloc(HYPRE_Int, 3 * nnzA, HYPRE_MEMORY_DEVICE);

   /* allocate C */
   d_jc = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_ac = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   }

   /* permutation vector */
   //d_pm = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   d_pm = mem_work;

   /* expansion: A's row idx */
   //d_it = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   d_it = d_pm + nnzA;
   hypreDevice_CsrRowPtrsToIndices_v2(m, nnzA, d_ia, d_it);

   /* a copy of col idx of A */
   //d_jt = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   d_jt = d_it + nnzA;
   hypre_TMemcpy(d_jt, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* sort: by col */
   HYPRE_THRUST_CALL(sequence, d_pm, d_pm + nnzA);
   HYPRE_THRUST_CALL(stable_sort_by_key, d_jt, d_jt + nnzA, d_pm);
   HYPRE_THRUST_CALL(gather, d_pm, d_pm + nnzA, d_it, d_jc);
   if (want_data)
   {
      HYPRE_THRUST_CALL(gather, d_pm, d_pm + nnzA, d_aa, d_ac);
   }

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(n, nnzA, d_jt);

#ifdef HYPRE_DEBUG
   HYPRE_Int nnzC;
   hypre_TMemcpy(&nnzC, &d_ic[n], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_assert(nnzC == nnzA);
#endif

   /*
   hypre_TFree(d_jt, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_it, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_pm, HYPRE_MEMORY_DEVICE);
   */
   hypre_TFree(mem_work, HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

#if defined(HYPRE_USING_SYCL)
HYPRE_Int
hypreDevice_CSRSpTrans(HYPRE_Int   m,        HYPRE_Int   n,        HYPRE_Int       nnzA,
                       HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_aa,
                       HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out,
                       HYPRE_Int   want_data)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] -= hypre_MPI_Wtime();
#endif

   HYPRE_Int *d_jt, *d_it, *d_pm, *d_ic, *d_jc;
   HYPRE_Complex *d_ac = NULL;
   HYPRE_Int *mem_work = hypre_TAlloc(HYPRE_Int, 3 * nnzA, HYPRE_MEMORY_DEVICE);

   /* allocate C */
   d_jc = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_ac = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
   }

   /* permutation vector */
   d_pm = mem_work;

   /* expansion: A's row idx */
   d_it = d_pm + nnzA;
   hypreDevice_CsrRowPtrsToIndices_v2(m, nnzA, d_ia, d_it);

   /* a copy of col idx of A */
   d_jt = d_it + nnzA;
   hypre_TMemcpy(d_jt, d_ja, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* sort: by col */
   oneapi::dpl::counting_iterator<HYPRE_Int> count(0);
   HYPRE_ONEDPL_CALL( std::copy,
                      count,
                      count + nnzA,
                      d_pm);

   auto zip_jt_pm = oneapi::dpl::make_zip_iterator(d_jt, d_pm);
   HYPRE_ONEDPL_CALL( std::stable_sort,
                      zip_jt_pm,
                      zip_jt_pm + nnzA,
   [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); } );

   auto permuted_it = oneapi::dpl::make_permutation_iterator(d_it, d_pm);
   HYPRE_ONEDPL_CALL( std::copy,
                      permuted_it,
                      permuted_it + nnzA,
                      d_jc );

   if (want_data)
   {
      auto permuted_aa = oneapi::dpl::make_permutation_iterator(d_aa, d_pm);
      HYPRE_ONEDPL_CALL( std::copy,
                         permuted_aa,
                         permuted_aa + nnzA,
                         d_ac );
   }

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(n, nnzA, d_jt);

#ifdef HYPRE_DEBUG
   HYPRE_Int nnzC;
   hypre_TMemcpy(&nnzC, &d_ic[n], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_assert(nnzC == nnzA);
#endif

   hypre_TFree(mem_work, HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPTRANS] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_SYCL)
