/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/* in_rc: 0: no input row count; 1: input row count est in d_rc; 2: input row bound in d_rc */
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
                                       char      *d_rf )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRownnzUpperbound");
#endif

   const HYPRE_Int SHMEM_HASH_SIZE = HYPRE_SPGEMM_SYMBL_HASH_SIZE;
   const HYPRE_Int GROUP_SIZE = HYPRE_WARP_SIZE;
   const HYPRE_Int BIN = 5;

   const bool need_ghash = in_rc > 0;
   const bool can_fail = in_rc < 2;

   hypre_spgemm_symbolic_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, false>
   (m, NULL, k, n, need_ghash, d_ia, d_ja, d_ib, d_jb, d_rc, can_fail, d_rf);

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/* in_rc: 0: no input row count; 1: input row count est in d_rc; 2: input row bound in d_rc */
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

   const HYPRE_Int SHMEM_HASH_SIZE = HYPRE_SPGEMM_SYMBL_HASH_SIZE;
   const HYPRE_Int GROUP_SIZE = HYPRE_WARP_SIZE;
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
         printf0("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__,
                 num_failed_rows, num_failed_rows / (m + 0.0) );
#endif
         HYPRE_Int *rf_ind = hypre_TAlloc(HYPRE_Int, num_failed_rows, HYPRE_MEMORY_DEVICE);

         HYPRE_Int *new_end =
            HYPRE_THRUST_CALL( copy_if,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(m),
                               d_rf,
                               rf_ind,
                               thrust::identity<char>() );

         hypre_assert(new_end - rf_ind == num_failed_rows);

         hypre_spgemm_symbolic_rownnz<BIN + 1, 2 * SHMEM_HASH_SIZE, 2 * GROUP_SIZE, true>
         (num_failed_rows, rf_ind, k, n, true, d_ia, d_ja, d_ib, d_jb, d_rc, false, NULL);

         hypre_TFree(rf_ind, HYPRE_MEMORY_DEVICE);
      }
   }

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#define HYPRE_SPGEMM_ROWNNZ_BINNED(BIN, SHMEM_HASH_SIZE, GROUP_SIZE, GHASH, CAN_FAIL)  \
{                                                                                      \
   const HYPRE_Int p = h_bin_ptr[BIN - 1];                                             \
   const HYPRE_Int q = h_bin_ptr[BIN];                                                 \
   const HYPRE_Int bs = q - p;                                                         \
   if (bs)                                                                             \
   {                                                                                   \
      /* printf0("bin[%d]: %d rows\n", BIN, bs); */                                    \
      hypre_spgemm_symbolic_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, true>             \
         (bs, d_rind + p, k, n, GHASH, d_ia, d_ja, d_ib, d_jb, d_rc, CAN_FAIL, NULL);  \
      HYPRE_SPGEMM_ROW(_spgemm_nrows, bs);                                             \
   }                                                                                   \
}

HYPRE_Int
hypre_spgemm_symbolic_binned( HYPRE_Int  m,
                              HYPRE_Int *d_rind, /* input: row indices (length of m) */
                              HYPRE_Int  k,
                              HYPRE_Int  n,
                              HYPRE_Int *d_ia,
                              HYPRE_Int *d_ja,
                              HYPRE_Int *d_ib,
                              HYPRE_Int *d_jb,
                              HYPRE_Int *d_rc )
{
#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1, t2;
#endif

   HYPRE_Int h_bin_ptr[HYPRE_SPGEMM_MAX_NBIN + 1];
   HYPRE_Int num_bins = hypre_HandleSpgemmAlgorithmNumBin(hypre_handle());
   const char s = 32, t = 6, u = num_bins;

#if defined(HYPRE_DEBUG)
   HYPRE_Int _spgemm_nrows = 0;
#endif

   /* create binning */
#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   t1 = hypre_MPI_Wtime();
#endif

   hypre_SpGemmCreateBins(m, s, t, u, d_rc, true, d_rind, h_bin_ptr);

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   printf0("%s[%d]: Binning time %f\n", __func__, __LINE__, t2);
#endif

   HYPRE_SPGEMM_ROWNNZ_BINNED( 6, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 2,
                               HYPRE_WARP_SIZE *  2, false, false); /* 1024,   64 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 7, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 4,
                               HYPRE_WARP_SIZE *  4, false, false); /* 2048,  128 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 8, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 8,
                               HYPRE_WARP_SIZE *  8, false, false); /* 4096,  256 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 9, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 16,
                               HYPRE_WARP_SIZE * 16, false, false); /* 8192,  512 */
   HYPRE_SPGEMM_ROWNNZ_BINNED(10, HYPRE_SPGEMM_SYMBL_HASH_SIZE * 32,
                               HYPRE_WARP_SIZE * 32, true, false);  /* 16384, 1024 */

#if defined(HYPRE_DEBUG)
   hypre_assert(_spgemm_nrows == m);
#endif

   return hypre_error_flag;
}

/* in_rc: RL: currently only 0: no input row count; */
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
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRownnzBinned");
#endif

   const bool need_ghash = in_rc > 0;
   const bool can_fail = in_rc < 2;

   char *d_rf = can_fail ? hypre_TAlloc(char, m, HYPRE_MEMORY_DEVICE) : NULL;

   hypre_spgemm_symbolic_rownnz<5, HYPRE_SPGEMM_SYMBL_HASH_SIZE, HYPRE_WARP_SIZE, false>
   (m, NULL, k, n, need_ghash, d_ia, d_ja, d_ib, d_jb, d_rc, can_fail, d_rf);

   if (can_fail)
   {
      HYPRE_Int num_failed_rows =
         HYPRE_THRUST_CALL( reduce,
                            thrust::make_transform_iterator(d_rf,     type_cast<char, HYPRE_Int>()),
                            thrust::make_transform_iterator(d_rf + m, type_cast<char, HYPRE_Int>()) );

      if (num_failed_rows)
      {
#ifdef HYPRE_SPGEMM_PRINTF
         printf0("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__,
                 num_failed_rows, num_failed_rows / (m + 0.0) );
#endif
         HYPRE_Int *rf_ind = hypre_TAlloc(HYPRE_Int, num_failed_rows, HYPRE_MEMORY_DEVICE);

         HYPRE_Int *new_end =
            HYPRE_THRUST_CALL( copy_if,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(m),
                               d_rf,
                               rf_ind,
                               thrust::identity<char>() );

         hypre_assert(new_end - rf_ind == num_failed_rows);

         hypre_spgemm_symbolic_binned(num_failed_rows, rf_ind, k, n, d_ia, d_ja, d_ib, d_jb, d_rc);
      }
   }

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

