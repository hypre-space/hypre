/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#define HYPRE_SPGEMM_ROWNNZ_BINNED(BIN, SHMEM_HASH_SIZE, GROUP_SIZE, GHASH, CAN_FAIL, RF)  \
{                                                                                          \
   const HYPRE_Int p = h_bin_ptr[BIN - 1];                                                 \
   const HYPRE_Int q = h_bin_ptr[BIN];                                                     \
   const HYPRE_Int bs = q - p;                                                             \
   if (bs)                                                                                 \
   {                                                                                       \
      /* hypre_printf0("bin[%d]: %d rows\n", BIN, bs); */                                        \
      hypre_spgemm_symbolic_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, true>                 \
         ( bs, d_rind + p, k, n, GHASH, d_ia, d_ja, d_ib, d_jb, d_rc, CAN_FAIL, RF );      \
   }                                                                                       \
}

/* in_rc: 0: no input row count
 *        1: input row count est (CURRENTLY ONLY 1)
*/
HYPRE_Int
hypreDevice_CSRSpGemmRownnzUpperboundNoBin( HYPRE_Int  m,
                                            HYPRE_Int  k,
                                            HYPRE_Int  n,
                                            HYPRE_Int *d_ia,
                                            HYPRE_Int *d_ja,
                                            HYPRE_Int *d_ib,
                                            HYPRE_Int *d_jb,
                                            HYPRE_Int  in_rc,
                                            HYPRE_Int *d_rc,
                                            char      *d_rf )
{
   const HYPRE_Int SHMEM_HASH_SIZE = HYPRE_SPGEMM_SYMBL_HASH_SIZE;
   const HYPRE_Int GROUP_SIZE = HYPRE_SPGEMM_BASE_GROUP_SIZE;
   const HYPRE_Int BIN = 5;

   const bool need_ghash = in_rc > 0;
   const bool can_fail = in_rc < 2;

   hypre_spgemm_symbolic_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, false>
      (m, NULL, k, n, need_ghash, d_ia, d_ja, d_ib, d_jb, d_rc, can_fail, d_rf);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpGemmRownnzUpperboundBinned( HYPRE_Int  m,
                                             HYPRE_Int  k,
                                             HYPRE_Int  n,
                                             HYPRE_Int *d_ia,
                                             HYPRE_Int *d_ja,
                                             HYPRE_Int *d_ib,
                                             HYPRE_Int *d_jb,
                                             HYPRE_Int  in_rc,
                                             HYPRE_Int *d_rc,
                                             char      *d_rf )
{
   const bool CAN_FAIL = true;

   /* Binning (bins 3-10) with d_rc */
   HYPRE_Int h_bin_ptr[HYPRE_SPGEMM_MAX_NBIN + 1];
   const char s = 32, t = 3, u = hypre_HandleSpgemmAlgorithmNumBin(hypre_handle());

   HYPRE_Int *d_rind = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);

   hypre_SpGemmCreateBins(m, s, t, u, d_rc, false, d_rind, h_bin_ptr);

   HYPRE_SPGEMM_ROWNNZ_BINNED( 3, HYPRE_SPGEMM_SYMBL_HASH_SIZE /  4,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE /  4, false, CAN_FAIL, d_rf); /* 128,   8 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 4, HYPRE_SPGEMM_SYMBL_HASH_SIZE /  2,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE /  2, false, CAN_FAIL, d_rf); /* 256,   16 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 5, HYPRE_SPGEMM_SYMBL_HASH_SIZE,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE,      false, CAN_FAIL, d_rf); /* 512,   32 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 6, HYPRE_SPGEMM_SYMBL_HASH_SIZE *  2,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE *  2, false, CAN_FAIL, d_rf); /* 1024,  64 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 7, HYPRE_SPGEMM_SYMBL_HASH_SIZE *  4,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE *  4, false, CAN_FAIL, d_rf); /* 2048,  128 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 8, HYPRE_SPGEMM_SYMBL_HASH_SIZE *  8,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE *  8, false, CAN_FAIL, d_rf); /* 4096,  256 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 9, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 16,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE * 16, false, CAN_FAIL, d_rf); /* 8192,  512 */
   HYPRE_SPGEMM_ROWNNZ_BINNED(10, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 32,
                               HYPRE_SPGEMM_BASE_GROUP_SIZE * 32, true,  CAN_FAIL, d_rf); /* 16384, 1024 */

   hypre_TFree(d_rind, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpGemmRownnzUpperbound( HYPRE_Int  m,
                                       HYPRE_Int  k,
                                       HYPRE_Int  n,
                                       HYPRE_Int *d_ia,
                                       HYPRE_Int *d_ja,
                                       HYPRE_Int *d_ib,
                                       HYPRE_Int *d_jb,
                                       HYPRE_Int  in_rc,
                                       HYPRE_Int *d_rc,
                                       HYPRE_Int *rownnz_exact_ptr)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRownnzUpperbound");
#endif

#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1 = hypre_MPI_Wtime();
#endif

   char *d_rf = hypre_TAlloc(char, m, HYPRE_MEMORY_DEVICE);

   const HYPRE_Int binned = hypre_HandleSpgemmAlgorithmBinned(hypre_handle());

   if (binned)
   {
      hypreDevice_CSRSpGemmRownnzUpperboundBinned
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc, d_rf);
   }
   else
   {
      hypreDevice_CSRSpGemmRownnzUpperboundNoBin
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc, d_rf);
   }

   /* row nnz is exact if no row failed */
   *rownnz_exact_ptr = !HYPRE_THRUST_CALL( any_of,
                                           d_rf,
                                           d_rf + m,
                                           thrust::identity<char>() );

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   HYPRE_Real t2 = hypre_MPI_Wtime() - t1;
   hypre_printf0("RownnzBound time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/* in_rc: 0: no input row count  (CURRENTLY ONLY 0)
 *        1: input row count est
 *        2: input row bound
*/
HYPRE_Int
hypreDevice_CSRSpGemmRownnzNoBin( HYPRE_Int  m,
                                  HYPRE_Int  k,
                                  HYPRE_Int  n,
                                  HYPRE_Int *d_ia,
                                  HYPRE_Int *d_ja,
                                  HYPRE_Int *d_ib,
                                  HYPRE_Int *d_jb,
                                  HYPRE_Int  in_rc,
                                  HYPRE_Int *d_rc )
{
   const HYPRE_Int SHMEM_HASH_SIZE = HYPRE_SPGEMM_SYMBL_HASH_SIZE;
   const HYPRE_Int GROUP_SIZE = HYPRE_SPGEMM_BASE_GROUP_SIZE;
   const HYPRE_Int BIN = 5;

   const bool need_ghash = in_rc > 0;
   const bool can_fail = in_rc < 2;

   char *d_rf = can_fail ? hypre_TAlloc(char, m, HYPRE_MEMORY_DEVICE) : NULL;

   hypre_spgemm_symbolic_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, false>
      (m, NULL, k, n, need_ghash, d_ia, d_ja, d_ib, d_jb, d_rc, can_fail, d_rf);

   if (can_fail)
   {
      /* row nnz is exact if no row failed */
      HYPRE_Int num_failed_rows =
         HYPRE_THRUST_CALL( reduce,
                            thrust::make_transform_iterator(d_rf,     type_cast<char, HYPRE_Int>()),
                            thrust::make_transform_iterator(d_rf + m, type_cast<char, HYPRE_Int>()) );

      if (num_failed_rows)
      {
#ifdef HYPRE_SPGEMM_PRINTF
         hypre_printf0("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__,
                 num_failed_rows, num_failed_rows / (m + 0.0) );
#endif
         HYPRE_Int *d_rind = hypre_TAlloc(HYPRE_Int, num_failed_rows, HYPRE_MEMORY_DEVICE);

         HYPRE_Int *new_end =
            HYPRE_THRUST_CALL( copy_if,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(m),
                               d_rf,
                               d_rind,
                               thrust::identity<char>() );

         hypre_assert(new_end - d_rind == num_failed_rows);

         hypre_spgemm_symbolic_rownnz<BIN + 1, 2 * SHMEM_HASH_SIZE, 2 * GROUP_SIZE, true>
            (num_failed_rows, d_rind, k, n, true, d_ia, d_ja, d_ib, d_jb, d_rc, false, NULL);

         hypre_TFree(d_rind, HYPRE_MEMORY_DEVICE);
      }
   }

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/* in_rc: 0: no input row count  (CURRENTLY ONLY 0)
 *        1: input row count est
 *        2: input row bound
*/
HYPRE_Int
hypreDevice_CSRSpGemmRownnzBinned( HYPRE_Int  m,
                                   HYPRE_Int  k,
                                   HYPRE_Int  n,
                                   HYPRE_Int *d_ia,
                                   HYPRE_Int *d_ja,
                                   HYPRE_Int *d_ib,
                                   HYPRE_Int *d_jb,
                                   HYPRE_Int  in_rc,
                                   HYPRE_Int *d_rc )
{
   const char s = 32, t = 1, u = 5;
   HYPRE_Int  h_bin_ptr[HYPRE_SPGEMM_MAX_NBIN + 1];
   HYPRE_Int *d_rind = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);

   hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, 1);

   hypre_SpGemmCreateBins(m, s, t, u, d_rc, false, d_rind, h_bin_ptr);

   HYPRE_SPGEMM_ROWNNZ_BINNED( 1, HYPRE_SPGEMM_SYMBL_HASH_SIZE / 16,
                                  HYPRE_SPGEMM_BASE_GROUP_SIZE / 16, false, false, NULL); /*  32,  2 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 2, HYPRE_SPGEMM_SYMBL_HASH_SIZE /  8,
                                  HYPRE_SPGEMM_BASE_GROUP_SIZE /  8, false, false, NULL); /*  64,  4 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 3, HYPRE_SPGEMM_SYMBL_HASH_SIZE /  4,
                                  HYPRE_SPGEMM_BASE_GROUP_SIZE /  4, false, false, NULL); /* 128,  8 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 4, HYPRE_SPGEMM_SYMBL_HASH_SIZE /  2,
                                  HYPRE_SPGEMM_BASE_GROUP_SIZE /  2, false, false, NULL); /* 256, 16 */

   if (h_bin_ptr[5] > h_bin_ptr[4])
   {
      char *d_rf = hypre_CTAlloc(char, m, HYPRE_MEMORY_DEVICE);

      HYPRE_SPGEMM_ROWNNZ_BINNED( 5, HYPRE_SPGEMM_SYMBL_HASH_SIZE,
                                     HYPRE_SPGEMM_BASE_GROUP_SIZE, false, true, d_rf); /* 512, 32 */

      HYPRE_Int num_failed_rows =
         HYPRE_THRUST_CALL( reduce,
                            thrust::make_transform_iterator(d_rf,     type_cast<char, HYPRE_Int>()),
                            thrust::make_transform_iterator(d_rf + m, type_cast<char, HYPRE_Int>()) );

      if (num_failed_rows)
      {
#ifdef HYPRE_SPGEMM_PRINTF
         hypre_printf0("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__,
                 num_failed_rows, num_failed_rows / (m + 0.0) );
#endif
         HYPRE_Int *new_end =
            HYPRE_THRUST_CALL( copy_if,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(m),
                               d_rf,
                               d_rind,
                               thrust::identity<char>() );

         hypre_assert(new_end - d_rind == num_failed_rows);

         /* Binning (bins 6-10) with d_rc which is a **rownnz-bound** now */
         const char t = 6, u = hypre_HandleSpgemmAlgorithmNumBin(hypre_handle());

         hypre_SpGemmCreateBins(num_failed_rows, s, t, u, d_rc, true, d_rind, h_bin_ptr);

         HYPRE_SPGEMM_ROWNNZ_BINNED( 6, HYPRE_SPGEMM_SYMBL_HASH_SIZE *  2,
                                        HYPRE_SPGEMM_BASE_GROUP_SIZE *  2, false, false, NULL); /* 1024,   64 */
         HYPRE_SPGEMM_ROWNNZ_BINNED( 7, HYPRE_SPGEMM_SYMBL_HASH_SIZE *  4,
                                        HYPRE_SPGEMM_BASE_GROUP_SIZE *  4, false, false, NULL); /* 2048,  128 */
         HYPRE_SPGEMM_ROWNNZ_BINNED( 8, HYPRE_SPGEMM_SYMBL_HASH_SIZE *  8,
                                        HYPRE_SPGEMM_BASE_GROUP_SIZE *  8, false, false, NULL); /* 4096,  256 */
         HYPRE_SPGEMM_ROWNNZ_BINNED( 9, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 16,
                                        HYPRE_SPGEMM_BASE_GROUP_SIZE * 16, false, false, NULL); /* 8192,  512 */
         HYPRE_SPGEMM_ROWNNZ_BINNED(10, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 32,
                                        HYPRE_SPGEMM_BASE_GROUP_SIZE * 32,  true, false, NULL); /* 16384, 1024 */
      }

      hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);
   }

   hypre_TFree(d_rind, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpGemmRownnz( HYPRE_Int  m,
                             HYPRE_Int  k,
                             HYPRE_Int  n,
                             HYPRE_Int *d_ia,
                             HYPRE_Int *d_ja,
                             HYPRE_Int *d_ib,
                             HYPRE_Int *d_jb,
                             HYPRE_Int  in_rc,
                             HYPRE_Int *d_rc )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRownnz");
#endif

#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1 = hypre_MPI_Wtime();
#endif

   const HYPRE_Int binned = hypre_HandleSpgemmAlgorithmBinned(hypre_handle());

   if (binned)
   {
      hypreDevice_CSRSpGemmRownnzBinned
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);
   }
   else
   {
      hypreDevice_CSRSpGemmRownnzNoBin
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);
   }

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   HYPRE_Real t2 = hypre_MPI_Wtime() - t1;
   hypre_printf0("Rownnz time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

