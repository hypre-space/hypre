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
   HYPRE_Real        t1, t2;
   HYPRE_Real        ta, tb;

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

   ta = hypre_MPI_Wtime();

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
      HYPRE_Int *d_rc = NULL;

      if (hypre_HandleSpgemmAlgorithm(hypre_handle()) == 1)
      {
         t1 = hypre_MPI_Wtime();
         d_rc = hypre_TAlloc(HYPRE_Int, 2*m, HYPRE_MEMORY_DEVICE);
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("Malloc rc %f\n", t2);

         t1 = hypre_MPI_Wtime();
         hypreDevice_CSRSpGemmRownnz(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);
         hypre_SyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("Rownnz time %f\n", t2);

         t1 = hypre_MPI_Wtime();
         hypreDevice_CSRSpGemmWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1,
                                                   &d_ic, &d_jc, &d_c, &nnzC);
         hypre_SyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("SpGemmNumerical time %f\n", t2);
      }
      else if (hypre_HandleSpgemmAlgorithm(hypre_handle()) == 2)
      {
         t1 = hypre_MPI_Wtime();
         d_rc = hypre_TAlloc(HYPRE_Int, 2*m, HYPRE_MEMORY_DEVICE);
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("Malloc rc %f\n", t2);

         t1 = hypre_MPI_Wtime();
         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);
         hypre_SyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("RownnzEst time %f\n", t2);

         t1 = hypre_MPI_Wtime();
         hypreDevice_CSRSpGemmWithRownnzEstimate(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc,
                                                 &d_ic, &d_jc, &d_c, &nnzC);
         hypre_SyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("SpGemmNumerical time %f\n", t2);
      }
      else
      {
         d_rc = hypre_TAlloc(HYPRE_Int, 2*m, HYPRE_MEMORY_DEVICE);

         t1 = hypre_MPI_Wtime();
         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("RownnzEst time %f\n", t2);

         t1 = hypre_MPI_Wtime();
         hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rc + m);
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("RownnzBound time %f\n", t2);

         /* row nnz is exact if no row failed */
         HYPRE_Int rownnz_exact = hypreDevice_IntegerReduceSum(m, d_rc + m) == 0;

         t1 = hypre_MPI_Wtime();
         hypreDevice_CSRSpGemmWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact,
                                                   &d_ic, &d_jc, &d_c, &nnzC);
         hypre_SyncCudaComputeStream(hypre_handle());
         t2 = hypre_MPI_Wtime() - t1;
         hypre_printf("SpGemmNumerical time %f\n", t2);
      }

      t1 = hypre_MPI_Wtime();
      hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);
      t2 = hypre_MPI_Wtime() - t1;
      hypre_printf("Free rc %f\n", t2);
   }

   tb = hypre_MPI_Wtime() - ta;
   hypre_printf("SpGemm time %f\n", tb);

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

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

