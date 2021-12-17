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
#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real        t1, t2;
   HYPRE_Real        ta, tb;
#endif

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

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   ta = hypre_MPI_Wtime();
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Attempting to use device sparse matrix library for SpGEMM without having compiled support for it!\n");
#endif
   }
   else
   {
      HYPRE_Int *d_rc = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
      const HYPRE_Int alg = hypre_HandleSpgemmAlgorithm(hypre_handle());
      const HYPRE_Int binned = hypre_HandleSpgemmAlgorithmBinned(hypre_handle());
      const HYPRE_Int row_est_mtd = hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle());

      if (alg == 1)
      {
#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         if (binned)
         {
            hypreDevice_CSRSpGemmRownnzBinned
               (m, k, n, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);
         }
         else
         {
            hypreDevice_CSRSpGemmRownnz
               (m, k, n, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);
         }
#ifdef HYPRE_SPGEMM_TIMING
         hypre_ForceSyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         printf0("Rownnz time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         if (binned)
         {
            hypreDevice_CSRSpGemmNumerWithRownnzUpperboundBinned
               (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, &d_ic, &d_jc, &d_c, &nnzC);
         }
         else
         {
            hypreDevice_CSRSpGemmNumerWithRownnzUpperbound
               (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, &d_ic, &d_jc, &d_c, &nnzC);
         }
#ifdef HYPRE_SPGEMM_TIMING
         hypre_ForceSyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         printf0("SpGemmNumerical time %f\n", t2);
#endif
      }
      else if (alg == 2)
      {
#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, row_est_mtd);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_ForceSyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         printf0("RownnzEst time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmNumerWithRownnzEstimate(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                                      &d_ic, &d_jc, &d_c, &nnzC);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_ForceSyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         printf0("SpGemmNumerical time %f\n", t2);
#endif
      }
      else if (alg == 3)
      {
#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, row_est_mtd);

#ifdef HYPRE_SPGEMM_TIMING
         hypre_ForceSyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         printf0("RownnzEst time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         char *d_rf = hypre_TAlloc(char, m, HYPRE_MEMORY_DEVICE);

         hypreDevice_CSRSpGemmRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc, d_rf);

         /* row nnz is exact if no row failed */
         HYPRE_Int rownnz_exact = !HYPRE_THRUST_CALL( any_of,
                                                      d_rf,
                                                      d_rf + m,
                                                      thrust::identity<char>() );

         hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_TIMING
         hypre_ForceSyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         printf0("RownnzBound time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact, &d_ic, &d_jc, &d_c, &nnzC);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_ForceSyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         printf0("SpGemmNumerical time %f\n", t2);
#endif
      }

      hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);
   }

#ifdef HYPRE_SPGEMM_TIMING
   tb = hypre_MPI_Wtime() - ta;
   printf0("SpGemm time %f\n", tb);
#endif

   hypre_CSRMatrixNumNonzeros(C) = nnzC;
   hypre_CSRMatrixI(C)           = d_ic;
   hypre_CSRMatrixJ(C)           = d_jc;
   hypre_CSRMatrixData(C)        = d_c;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

