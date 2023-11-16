/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_GPU)

/*
 * d_rc: input: nnz (upper bound) of each row
 * exact_rownnz: if d_rc is exact
 */
HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperboundNoBin( HYPRE_Int       m,
                                                     HYPRE_Int       k,
                                                     HYPRE_Int       n,
                                                     HYPRE_Int      *d_ia,
                                                     HYPRE_Int      *d_ja,
                                                     HYPRE_Complex  *d_a,
                                                     HYPRE_Int      *d_ib,
                                                     HYPRE_Int      *d_jb,
                                                     HYPRE_Complex  *d_b,
                                                     HYPRE_Int      *d_rc,
                                                     HYPRE_Int       exact_rownnz,
                                                     HYPRE_Int     **d_ic_out,
                                                     HYPRE_Int     **d_jc_out,
                                                     HYPRE_Complex **d_c_out,
                                                     HYPRE_Int      *nnzC_out )
{
   constexpr HYPRE_Int SHMEM_HASH_SIZE = NUMER_HASH_SIZE[HYPRE_SPGEMM_DEFAULT_BIN];
   constexpr HYPRE_Int GROUP_SIZE = T_GROUP_SIZE[HYPRE_SPGEMM_DEFAULT_BIN];
   const HYPRE_Int BIN = HYPRE_SPGEMM_DEFAULT_BIN;

#ifdef HYPRE_SPGEMM_PRINTF
#if defined(HYPRE_USING_SYCL)
   HYPRE_Int max_rc = HYPRE_ONEDPL_CALL(std::reduce, d_rc, d_rc + m, 0,
                                        sycl::maximum<HYPRE_Int>());
   HYPRE_Int min_rc = HYPRE_ONEDPL_CALL(std::reduce, d_rc, d_rc + m, max_rc,
                                        sycl::minimum<HYPRE_Int>());
#else
   HYPRE_Int max_rc = HYPRE_THRUST_CALL(reduce, d_rc, d_rc + m, 0,      thrust::maximum<HYPRE_Int>());
   HYPRE_Int min_rc = HYPRE_THRUST_CALL(reduce, d_rc, d_rc + m, max_rc, thrust::minimum<HYPRE_Int>());
#endif
   HYPRE_SPGEMM_PRINT("%s[%d]: max RC %d, min RC %d\n", __FILE__, __LINE__, max_rc, min_rc);
#endif

   /* if rc contains exact rownnz: can allocate the final C=(ic,jc,c) directly;
      if rc contains upper bound : it is a temporary space that is more than enough to store C */
   HYPRE_Int     *d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_jc;
   HYPRE_Complex *d_c;
   HYPRE_Int      nnzC = -1;

   hypre_create_ija(m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC);

#ifdef HYPRE_SPGEMM_PRINTF
   HYPRE_SPGEMM_PRINT("%s[%d]: nnzC %d\n", __FILE__, __LINE__, nnzC);
#endif


   /* even with exact rownnz, still may need global hash, since shared hash is smaller than symbol */
   hypre_spgemm_numerical_with_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, false>
   (m, NULL, k, n, true, exact_rownnz, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_ic, d_jc, d_c);

   if (!exact_rownnz)
   {
      hypreDevice_CSRSpGemmNumerPostCopy<T_GROUP_SIZE[5]>(m, d_rc, &nnzC, &d_ic, &d_jc, &d_c);
   }

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   return hypre_error_flag;
}

#define HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED2(BIN, BIN2, SHMEM_HASH_SIZE, GROUP_SIZE, EXACT_ROWNNZ, GHASH)  \
{                                                                                                                \
   const HYPRE_Int p = h_bin_ptr[BIN - 1];                                                                       \
   const HYPRE_Int q = h_bin_ptr[BIN];                                                                           \
   const HYPRE_Int bs = q - p;                                                                                   \
   if (bs)                                                                                                       \
   {                                                                                                             \
      HYPRE_SPGEMM_PRINT("bin[%d]: %d rows, p %d, q %d\n", BIN, bs, p, q);                                       \
      hypre_spgemm_numerical_with_rownnz<BIN2, SHMEM_HASH_SIZE, GROUP_SIZE, true>                                \
         (bs, d_rind + p, k, n, GHASH, EXACT_ROWNNZ, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_ic, d_jc, d_c);   \
   }                                                                                                             \
}

#define HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(BIN, SHMEM_HASH_SIZE, GROUP_SIZE, EXACT_ROWNNZ, GHASH)         \
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED2(BIN, BIN, SHMEM_HASH_SIZE, GROUP_SIZE, EXACT_ROWNNZ, GHASH)

HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperboundBinned( HYPRE_Int       m,
                                                      HYPRE_Int       k,
                                                      HYPRE_Int       n,
                                                      HYPRE_Int      *d_ia,
                                                      HYPRE_Int      *d_ja,
                                                      HYPRE_Complex  *d_a,
                                                      HYPRE_Int      *d_ib,
                                                      HYPRE_Int      *d_jb,
                                                      HYPRE_Complex  *d_b,
                                                      HYPRE_Int      *d_rc,
                                                      HYPRE_Int       exact_rownnz,
                                                      HYPRE_Int     **d_ic_out,
                                                      HYPRE_Int     **d_jc_out,
                                                      HYPRE_Complex **d_c_out,
                                                      HYPRE_Int      *nnzC_out )
{
   /* if rc contains exact rownnz: can allocate the final C=(ic,jc,c) directly;
      if rc contains upper bound : it is a temporary space that is more than enough to store C */
   HYPRE_Int     *d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_jc;
   HYPRE_Complex *d_c;
   HYPRE_Int      nnzC = -1;

   hypre_create_ija(m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC);

#ifdef HYPRE_SPGEMM_PRINTF
   HYPRE_SPGEMM_PRINT("%s[%d]: nnzC %d\n", __FILE__, __LINE__, nnzC);
#endif

   HYPRE_Int *d_rind = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
   HYPRE_Int  h_bin_ptr[HYPRE_SPGEMM_MAX_NBIN + 1];
   //HYPRE_Int  num_bins = hypre_HandleSpgemmNumBin(hypre_handle());
   HYPRE_Int high_bin = hypre_HandleSpgemmHighestBin(hypre_handle())[1];
   const bool hbin9 = 9 == high_bin;
   const char s = NUMER_HASH_SIZE[1] / 2, t = 2, u = high_bin;

   hypre_SpGemmCreateBins(m, s, t, u, d_rc, false, d_rind, h_bin_ptr);

#if 0
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  1, NUMER_HASH_SIZE[ 1], T_GROUP_SIZE[ 1], exact_rownnz,
                                               false);
#endif
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  2, NUMER_HASH_SIZE[ 2], T_GROUP_SIZE[ 2], exact_rownnz,
                                               false);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  3, NUMER_HASH_SIZE[ 3], T_GROUP_SIZE[ 3], exact_rownnz,
                                               false);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  4, NUMER_HASH_SIZE[ 4], T_GROUP_SIZE[ 4], exact_rownnz,
                                               false);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  5, NUMER_HASH_SIZE[ 5], T_GROUP_SIZE[ 5], exact_rownnz,
                                               false);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  6, NUMER_HASH_SIZE[ 6], T_GROUP_SIZE[ 6], exact_rownnz,
                                               false);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  7, NUMER_HASH_SIZE[ 7], T_GROUP_SIZE[ 7], exact_rownnz,
                                               false);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  8, NUMER_HASH_SIZE[ 8], T_GROUP_SIZE[ 8], exact_rownnz,
                                               false);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  9, NUMER_HASH_SIZE[ 9], T_GROUP_SIZE[ 9], exact_rownnz,
                                               hbin9);
   HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 10, NUMER_HASH_SIZE[10], T_GROUP_SIZE[10], exact_rownnz,
                                              true);

   if (!exact_rownnz)
   {
      hypreDevice_CSRSpGemmNumerPostCopy<T_GROUP_SIZE[5]>(m, d_rc, &nnzC, &d_ic, &d_jc, &d_c);
   }

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   hypre_TFree(d_rind, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperbound( HYPRE_Int       m,
                                                HYPRE_Int       k,
                                                HYPRE_Int       n,
                                                HYPRE_Int      *d_ia,
                                                HYPRE_Int      *d_ja,
                                                HYPRE_Complex  *d_a,
                                                HYPRE_Int      *d_ib,
                                                HYPRE_Int      *d_jb,
                                                HYPRE_Complex  *d_b,
                                                HYPRE_Int      *d_rc,
                                                HYPRE_Int       exact_rownnz,
                                                HYPRE_Int     **d_ic_out,
                                                HYPRE_Int     **d_jc_out,
                                                HYPRE_Complex **d_c_out,
                                                HYPRE_Int      *nnzC_out )

{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPGEMM_NUMERIC] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmNumer");
#endif

#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1 = hypre_MPI_Wtime();
#endif

   const HYPRE_Int binned = hypre_HandleSpgemmBinned(hypre_handle());

   if (binned)
   {
      hypreDevice_CSRSpGemmNumerWithRownnzUpperboundBinned
      (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, d_ic_out, d_jc_out, d_c_out, nnzC_out);
   }
   else
   {
      hypreDevice_CSRSpGemmNumerWithRownnzUpperboundNoBin
      (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, d_ic_out, d_jc_out, d_c_out, nnzC_out);
   }

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   HYPRE_Real t2 = hypre_MPI_Wtime() - t1;
   HYPRE_SPGEMM_PRINT("SpGemmNumerical time %f\n", t2);
#endif

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPGEMM_NUMERIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_GPU) */
