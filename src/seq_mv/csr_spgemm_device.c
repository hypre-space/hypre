/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "seq_mv.hpp"

//#define HYPRE_SPGEMM_TIMING

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
      HYPRE_Int *d_rc = NULL;

      if (hypre_HandleSpgemmAlgorithm(hypre_handle()) == 1)
      {
         d_rc = hypre_TAlloc(HYPRE_Int, 2 * m, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmRownnz(m, k, n, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_SyncComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("Rownnz time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                                        1 /* exact row nnz */,
                                                        &d_ic, &d_jc, &d_c, &nnzC);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_SyncComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("SpGemmNumerical time %f\n", t2);
#endif
      }
      else if (hypre_HandleSpgemmAlgorithm(hypre_handle()) == 2)
      {
         d_rc = hypre_TAlloc(HYPRE_Int, 2 * m, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_SyncComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("RownnzEst time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmNumerWithRownnzEstimate(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                                      &d_ic, &d_jc, &d_c, &nnzC);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_SyncComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("SpGemmNumerical time %f\n", t2);
#endif
      }
      else
      {
         d_rc = hypre_TAlloc(HYPRE_Int, 2 * m, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_SyncComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("RownnzEst time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc,
                                               d_rc + m);

         /* row nnz is exact if no row failed */
         HYPRE_Int rownnz_exact = !HYPRE_THRUST_CALL( any_of,
                                                      d_rc + m,
                                                      d_rc + 2 * m,
                                                      thrust::identity<HYPRE_Int>() );
#ifdef HYPRE_SPGEMM_TIMING
         hypre_SyncComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("RownnzBound time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_TIMING
         t1 = hypre_MPI_Wtime();
#endif
         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                                        rownnz_exact,
                                                        &d_ic, &d_jc, &d_c, &nnzC);
#ifdef HYPRE_SPGEMM_TIMING
         hypre_SyncComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("SpGemmNumerical time %f\n", t2);
#endif
      }

      hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);
   }

#ifdef HYPRE_SPGEMM_TIMING
   tb = hypre_MPI_Wtime() - ta;
   hypre_printf("SpGemm time %f\n", tb);
#endif

   hypre_CSRMatrixNumNonzeros(C) = nnzC;
   hypre_CSRMatrixI(C)           = d_ic;
   hypre_CSRMatrixJ(C)           = d_jc;
   hypre_CSRMatrixData(C)        = d_c;

#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPMM] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

