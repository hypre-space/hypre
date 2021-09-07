/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

HYPRE_Int
hypreDevice_CSRSpGemm(hypre_CSRMatrix  *A,
                      hypre_CSRMatrix  *B,
                      hypre_CSRMatrix **C_ptr)
{
   HYPRE_Complex    *d_a  = hypre_CSRMatrixData(A);
   HYPRE_Int        *d_ia = hypre_CSRMatrixI(A);
   HYPRE_Int        *d_ja = hypre_CSRMatrixJ(A);
   HYPRE_Int         m    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         k    = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnza = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *d_b  = hypre_CSRMatrixData(B);
   HYPRE_Int        *d_ib = hypre_CSRMatrixI(B);
   HYPRE_Int        *d_jb = hypre_CSRMatrixJ(B);
   HYPRE_Int         n    = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         nnzb = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Complex    *d_c;
   HYPRE_Int        *d_ic;
   HYPRE_Int        *d_jc;
   HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   *C_ptr = C = hypre_CSRMatrixCreate(m, n, 0);
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   /* trivial case */
   if (nnza == 0 || nnzb == 0)
   {
      hypre_CSRMatrixI(C) = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);

      return hypre_error_flag;
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM] -= hypre_MPI_Wtime();
#endif

   /* use CUSPARSE or rocSPARSE*/
   if (hypre_HandleSpgemmUseCusparse(hypre_handle()))
   {
#if defined(HYPRE_USING_CUSPARSE)
      hypreDevice_CSRSpGemmCusparse(m, k, n,
                                    hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                    hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                    hypre_CSRMatrixGPUMatDescr(C), &nnzC, &d_ic, &d_jc, &d_c);
#elif defined(HYPRE_USING_ROCSPARSE)
      hypreDevice_CSRSpGemmRocsparse(m, k, n,
                                     hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                     hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                     hypre_CSRMatrixGPUMatDescr(C), hypre_CSRMatrixGPUMatInfo(C), &nnzC, &d_ic, &d_jc, &d_c);
#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Attempting to use device sparse matrix library for SpGEMM without having compiled support for it!\n");
#endif
   }
   else
   {
      HYPRE_Int m2 = hypre_HandleSpgemmNumPasses(hypre_handle()) < 3 ? m : 2*m;
      HYPRE_Int *d_rc = hypre_TAlloc(HYPRE_Int, m2, HYPRE_MEMORY_DEVICE);

      hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);

      if (hypre_HandleSpgemmNumPasses(hypre_handle()) < 3)
      {
         hypreDevice_CSRSpGemmWithRownnzEstimate(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                                 &d_ic, &d_jc, &d_c, &nnzC);
      }
      else
      {
         HYPRE_Int rownnz_exact;
         /* a binary array to indicate if row nnz counting is failed for a row */
         //HYPRE_Int *d_rf = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
         HYPRE_Int *d_rf = d_rc + m;

         hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);

         /* row nnz is exact if no row failed */
         rownnz_exact = hypreDevice_IntegerReduceSum(m, d_rf) == 0;

         //hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

         hypreDevice_CSRSpGemmWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact,
                                                   &d_ic, &d_jc, &d_c, &nnzC);
      }

      hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);
   }

   hypre_CSRMatrixNumNonzeros(C) = nnzC;
   hypre_CSRMatrixI(C) = d_ic;
   hypre_CSRMatrixJ(C) = d_jc;
   hypre_CSRMatrixData(C) = d_c;

#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPMM] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_CSRMatrixDeviceSpGemmSetRownnzEstimateMethod( HYPRE_Int value )
{
   if (value == 1 || value == 2 || value == 3)
   {
      hypre_HandleCudaData(hypre_handle())->spgemm_rownnz_estimate_method = value;
   }
   else
   {
      return -1;
   }

   return 0;
}

HYPRE_Int
hypre_CSRMatrixDeviceSpGemmSetRownnzEstimateNSamples( HYPRE_Int value )
{
   hypre_HandleCudaData(hypre_handle())->spgemm_rownnz_estimate_nsamples = value;

   return 0;
}

HYPRE_Int
hypre_CSRMatrixDeviceSpGemmSetRownnzEstimateMultFactor( HYPRE_Real value )
{
   if (value > 0.0)
   {
      hypre_HandleCudaData(hypre_handle())->spgemm_rownnz_estimate_mult_factor = value;
   }
   else
   {
      return -1;
   }

   return 0;
}

HYPRE_Int
hypre_CSRMatrixDeviceSpGemmSetHashType( char value )
{
   if (value == 'L' || value == 'Q' || value == 'D')
   {
      hypre_HandleCudaData(hypre_handle())->spgemm_hash_type = value;
   }
   else
   {
      return -1;
   }

   return 0;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

