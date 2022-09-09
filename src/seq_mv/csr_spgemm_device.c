/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)

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
   hypre_profile_times[HYPRE_TIMER_ID_SPGEMM] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   HYPRE_Real ta = hypre_MPI_Wtime();
#endif

   /* use CUSPARSE or rocSPARSE*/
   if (hypre_HandleSpgemmUseVendor(hypre_handle()))
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
#elif defined(HYPRE_USING_ONEMKLSPARSE)
      hypreDevice_CSRSpGemmOnemklsparse(m, k, n,
                                        hypre_CSRMatrixGPUMatHandle(A), nnza, d_ia, d_ja, d_a,
                                        hypre_CSRMatrixGPUMatHandle(B), nnzb, d_ib, d_jb, d_b,
                                        hypre_CSRMatrixGPUMatHandle(C), &nnzC, &d_ic, &d_jc, &d_c);
#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Attempting to use device sparse matrix library for SpGEMM without having compiled support for it!\n");
#endif
   }
   else
   {
      d_a  = hypre_CSRMatrixPatternOnly(A) ? NULL : d_a;
      d_b  = hypre_CSRMatrixPatternOnly(B) ? NULL : d_b;

      HYPRE_Int *d_rc = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
      const HYPRE_Int alg = hypre_HandleSpgemmAlgorithm(hypre_handle());
      hypre_printf("\n\nWM: debug - in custom spgemm, alg = %d\n\n", alg);

      if (hypre_HandleSpgemmNumBin(hypre_handle()) == 0)
      {
         hypreDevice_CSRSpGemmBinnedGetBlockNumDim();
      }

      if (alg == 1)
      {
         hypreDevice_CSRSpGemmRownnz
         (m, k, n, nnza, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);
         /* WM: debug */
         hypre_printf("WM: debug - final d_rc = ");
         HYPRE_Int i;
         for (i = 0; i < m; i++)
            hypre_printf("%d ", d_rc[i]);
         hypre_printf("\n");

         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, &d_ic, &d_jc, &d_c, &nnzC);
         /* WM: debug */
         hypre_printf("WM: debug - final d_ic = ");
         for (i = 0; i < m + 1; i++)
            hypre_printf("%d ", d_ic[i]);
         hypre_printf("\n");
         /* WM: debug */
         hypre_printf("WM: debug - final d_jc = ");
         for (i = 0; i < nnzC; i++)
            hypre_printf("%d ", d_jc[i]);
         hypre_printf("\n");
         /* WM: debug */
         hypre_printf("WM: debug - final d_c = ");
         for (i = 0; i < nnzC; i++)
            hypre_printf("%e ", d_c[i]);
         hypre_printf("\n");
      }
      else /* if (alg == 3) */
      {
         const HYPRE_Int row_est_mtd = hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle());

         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, row_est_mtd);

         HYPRE_Int rownnz_exact;

         hypreDevice_CSRSpGemmRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc, &rownnz_exact);

         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact, &d_ic, &d_jc, &d_c, &nnzC);
      }

      hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);
   }

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   HYPRE_Real tb = hypre_MPI_Wtime() - ta;
   HYPRE_SPGEMM_PRINT("SpGemm time %f\n", tb);
#endif

   hypre_CSRMatrixNumNonzeros(C) = nnzC;
   hypre_CSRMatrixI(C)           = d_ic;
   hypre_CSRMatrixJ(C)           = d_jc;
   hypre_CSRMatrixData(C)        = d_c;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPGEMM] += hypre_MPI_Wtime();
#endif

   /* WM: debug - check results agains host implementation */
   hypre_printf("WM: debug - check against host\n");
   hypre_CSRMatrix *C_host = hypre_CSRMatrixMultiplyHost(A, B);
   hypre_CSRMatrix *error = hypre_CSRMatrixAddHost(1.0, C, -1.0, C_host);
   HYPRE_Real err_norm = hypre_CSRMatrixFnorm(error); 
   hypre_printf("WM: debug - err_norm = %e\n", err_norm);
   if (err_norm)
   {
      hypre_CSRMatrixPrint(A, "A");
      hypre_CSRMatrixPrint(B, "B");
      hypre_CSRMatrixPrint(C_host, "C_host");
      hypre_CSRMatrixPrint(C, "C_device");
      hypre_CSRMatrixPrint(C_host, "error");
   }
   hypre_CSRMatrixDestroy(error);
   hypre_CSRMatrixDestroy(C_host);
   if (err_norm)
   {
      hypre_MPI_Finalize();
      exit(1);
   }

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL) */

