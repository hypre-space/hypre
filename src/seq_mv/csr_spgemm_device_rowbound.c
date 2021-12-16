/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Symbolic Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_symbl( volatile HYPRE_Int
                                *HashKeys, /* assumed to be initialized as all -1's */
                                HYPRE_Int   key,      /* assumed to be nonnegative */
                                HYPRE_Int  &count     /* increase by 1 if is a new entry */)
{
   HYPRE_Int j = 0;

#pragma unroll
   for (HYPRE_Int i = 0; i < SHMEM_HASH_SIZE; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (SHMEM_HASH_SIZE - 1);
      }
      else
      {
         j = HashFunc<SHMEM_HASH_SIZE, HASHTYPE>(key, i, j);
      }

      /* try to insert key+1 into slot j */
      HYPRE_Int old = atomicCAS((HYPRE_Int*)(HashKeys + j), -1, key);

      if (old == -1)
      {
         count++;
         return j;
      }
      if (old == key)
      {
         return j;
      }
   }
   return -1;
}

template <char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_symbl( HYPRE_Int   HashSize, /* capacity of the hash table */
                                volatile HYPRE_Int  *HashKeys, /* assumed to be initialized as all -1's */
                                HYPRE_Int   key,      /* assumed to be nonnegative */
                                HYPRE_Int  &count     /* increase by 1 if is a new entry */)
{
   HYPRE_Int j = 0;

#pragma unroll
   for (HYPRE_Int i = 0; i < HashSize; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (HashSize - 1);
      }
      else
      {
         j = HashFunc<HASHTYPE>(HashSize, key, i, j);
      }

      /* try to insert key+1 into slot j */
      HYPRE_Int old = atomicCAS((HYPRE_Int*)(HashKeys + j), -1, key);

      if (old == -1)
      {
         count++;
         return j;
      }
      if (old == key)
      {
         return j;
      }
   }
   return -1;
}

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, HYPRE_Int GROUP_SIZE, bool HAS_GHASH>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_compute_row_symbl( HYPRE_Int  istart_a,
                                HYPRE_Int  iend_a,
                                const HYPRE_Int *ja,
                                const HYPRE_Int *ib,
                                const HYPRE_Int *jb,
                                volatile HYPRE_Int *s_HashKeys,
                                HYPRE_Int  g_HashSize,
                                HYPRE_Int *g_HashKeys,
                                char &failed )
{
   HYPRE_Int num_new_insert = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart_a + threadIdx.y; __any_sync(HYPRE_WARP_FULL_MASK, i < iend_a);
        i += blockDim.y)
   {
      HYPRE_Int rowB = -1;

      if (threadIdx.x == 0 && i < iend_a)
      {
         rowB = read_only_load(ja + i);
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(...);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      rowB = __shfl_sync(HYPRE_WARP_FULL_MASK, rowB, 0, blockDim.x);
      /* open this row of B, collectively */
      HYPRE_Int tmp = 0;
      if (rowB != -1 && threadIdx.x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx.x);
      }
      const HYPRE_Int rowB_start = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 0, blockDim.x);
      const HYPRE_Int rowB_end   = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 1, blockDim.x);

      for (HYPRE_Int k = rowB_start + threadIdx.x; __any_sync(HYPRE_WARP_FULL_MASK, k < rowB_end);
           k += blockDim.x)
      {
         if (k < rowB_end)
         {
            const HYPRE_Int k_idx = read_only_load(jb + k);
            /* first try to insert into shared memory hash table */
            HYPRE_Int pos = hypre_spgemm_hash_insert_symbl<SHMEM_HASH_SIZE, HASHTYPE>
                            (s_HashKeys, k_idx, num_new_insert);

            if (HAS_GHASH && -1 == pos)
            {
               pos = hypre_spgemm_hash_insert_symbl<HASHTYPE>
                     (g_HashSize, g_HashKeys, k_idx, num_new_insert);
            }
            /* if failed again, both hash tables must have been full
               (hash table size estimation was too small).
               Increase the counter anyhow (will lead to over-counting) */
            if (pos == -1)
            {
               num_new_insert ++;
               failed = 1;
            }
         }
      }
   }

   return num_new_insert;
}

template <HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE, bool HAS_RIND, bool CAN_FAIL, char HASHTYPE, bool HAS_GHASH, bool BINNED>
__global__ void
hypre_spgemm_symbolic( const HYPRE_Int               M, /* HYPRE_Int K, HYPRE_Int N, */
                       const HYPRE_Int* __restrict__ rind,
                       const HYPRE_Int* __restrict__ ia,
                       const HYPRE_Int* __restrict__ ja,
                       const HYPRE_Int* __restrict__ ib,
                       const HYPRE_Int* __restrict__ jb,
                       const HYPRE_Int* __restrict__ ig,
                             HYPRE_Int* __restrict__ jg,
                             HYPRE_Int* __restrict__ rc,
                             char*      __restrict__ rf )
{
   /* number of groups in the grid */
   volatile const HYPRE_Int grid_num_groups = get_num_groups() * gridDim.x;
   /* group id inside the block */
   volatile const HYPRE_Int group_id = get_group_id();
   /* group id in the grid */
   volatile const HYPRE_Int grid_group_id = blockIdx.x * get_num_groups() + group_id;
   /* lane id inside the group */
   volatile const HYPRE_Int lane_id = get_lane_id();
   /* lane id inside the warp */
   volatile const HYPRE_Int warp_lane_id = get_warp_lane_id();
   /* shared memory hash table */
   __shared__ volatile HYPRE_Int s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
   /* shared memory hash table for this group */
   volatile HYPRE_Int *group_s_HashKeys = s_HashKeys + group_id * SHMEM_HASH_SIZE;

   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (HYPRE_Int i = grid_group_id; __any_sync(HYPRE_WARP_FULL_MASK, i < M); i += grid_num_groups)
   {
      HYPRE_Int ii = -1;
      char failed = 0;

      if (HAS_RIND)
      {
         group_read<GROUP_SIZE>(rind + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                                ii,
                                GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);
      }
      else
      {
         ii = i;
      }

      /* start/end position of global memory hash table */
      HYPRE_Int istart_g = 0, iend_g = 0, ghash_size = 0;

      if (HAS_GHASH)
      {
         group_read<GROUP_SIZE>(ig + grid_group_id, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                                istart_g, iend_g,
                                GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);

         /* size of global hash table allocated for this row
           (must be power of 2 and >= the actual size of the row of C - shmem hash size) */
         ghash_size = iend_g - istart_g;

         /* initialize group's global memory hash table */
#pragma unroll
         for (HYPRE_Int k = lane_id; k < ghash_size; k += GROUP_SIZE)
         {
            jg[istart_g + k] = -1;
         }
      }

      /* initialize group's shared memory hash table */
      if (GROUP_SIZE >= HYPRE_WARP_SIZE || i < M)
      {
#pragma unroll
         for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
         {
            group_s_HashKeys[k] = -1;
         }
      }

      group_sync<GROUP_SIZE>();

      /* start/end position of row of A */
      HYPRE_Int istart_a = 0, iend_a = 0;

      /* load the start and end position of row ii of A */
      group_read<GROUP_SIZE>(ia + ii, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                             istart_a, iend_a,
                             GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane_id : lane_id);

      /* work with two hash tables */
      HYPRE_Int jsum =
         hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH>(istart_a, iend_a,
                                                                                          ja, ib, jb,
                                                                                          group_s_HashKeys,
                                                                                          ghash_size,
                                                                                          jg + istart_g,
                                                                                          failed);

#if defined(HYPRE_DEBUG)
      hypre_device_assert(CAN_FAIL || failed == 0);
#endif

      /* num of nonzeros of this row (an upper bound) */
      jsum = group_reduce_sum<HYPRE_Int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(jsum);

      /* if this row failed */
      if (CAN_FAIL)
      {
         failed = (char) group_reduce_sum<hypre_int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>((hypre_int) failed);
      }

      if ((GROUP_SIZE >= HYPRE_WARP_SIZE || i < M) && lane_id == 0)
      {
#if defined(HYPRE_DEBUG)
         hypre_device_assert(ii >= 0);
#endif
         rc[ii] = jsum;

         if (CAN_FAIL)
         {
            rf[ii] = failed > 0;
         }
      }
   }
}

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE, bool NEED_GHASH, bool CAN_FAIL, bool BINNED>
HYPRE_Int
hypre_spgemm_rownnz_attempt(HYPRE_Int  m,
                            HYPRE_Int *row_ind, /* input: row indices (length of m) */
                            HYPRE_Int  k,
                            HYPRE_Int  n,
                            char       hash_type,
                            HYPRE_Int *d_ia,
                            HYPRE_Int *d_ja,
                            HYPRE_Int *d_ib,
                            HYPRE_Int *d_jb,
                            HYPRE_Int *d_rc,
                            char      *d_rf  /* output: if symbolic mult. failed for each row  */ )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_CUDA)
   const HYPRE_Int num_groups_per_block  = hypre_min(16 * HYPRE_WARP_SIZE / GROUP_SIZE, 64);
   const HYPRE_Int BDIMX                 = 2;
#elif defined(HYPRE_USING_HIP)
   const HYPRE_Int num_groups_per_block  = hypre_min(16 * HYPRE_WARP_SIZE / GROUP_SIZE, 64);
   const HYPRE_Int BDIMX                 = 4;
#endif
   const HYPRE_Int num_threads_per_block = num_groups_per_block * GROUP_SIZE;
   const HYPRE_Int BDIMY                 = GROUP_SIZE / BDIMX;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
   hypre_assert(bDim.x * bDim.y == GROUP_SIZE);
   // total number of threads used: a group works on a row
   HYPRE_Int num_threads = hypre_min((size_t) m * GROUP_SIZE, HYPRE_MAX_NUM_WARPS * HYPRE_WARP_SIZE);
   // grid dimension (number of blocks)
   dim3 gDim( (num_threads + num_threads_per_block - 1) / num_threads_per_block );
   // number of active groups
   HYPRE_Int num_act_groups = hypre_min(bDim.z * gDim.x, m);

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/
   HYPRE_Int *d_ghash_i = NULL;
   HYPRE_Int *d_ghash_j = NULL;
   HYPRE_Int  ghash_size = 0;

   if (NEED_GHASH)
   {
      hypre_SpGemmCreateGlobalHashTable(m, row_ind, num_act_groups, d_rc, SHMEM_HASH_SIZE,
                                        &d_ghash_i, &d_ghash_j, NULL, &ghash_size);
   }

#ifdef HYPRE_SPGEMM_PRINTF
   printf0("%s[%d]: m %d k %d n %d, HASH %c, SHMEM_HASH_SIZE %d, GROUP_SIZE %d, NEED_GHASH %d, ghash %p size %d\n",
           __func__, __LINE__, m, k, n,
           hash_type, SHMEM_HASH_SIZE, GROUP_SIZE, NEED_GHASH, d_ghash_i, ghash_size);
   printf0("%s[%d]: kernel spec [%d %d %d] x [%d %d %d]\n", __func__, __LINE__, gDim.x, gDim.y, gDim.z,
           bDim.x, bDim.y, bDim.z);
#endif

   /* ---------------------------------------------------------------------------
    * symbolic multiplication:
    * On output, it provides an upper bound of nnz in rows of C
    * ---------------------------------------------------------------------------*/
   const bool has_rind = BINNED || !CAN_FAIL;

   hypre_assert(has_rind == (row_ind != NULL) );

   if (NEED_GHASH && ghash_size)
   {
      HYPRE_CUDA_LAUNCH( (
                           hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, has_rind, CAN_FAIL, 'D', true, BINNED>),
                           gDim, bDim,
                           m, row_ind, /*k, n,*/ d_ia, d_ja, d_ib, d_jb,
                           d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
   else
   {
      HYPRE_CUDA_LAUNCH( (
                           hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, has_rind, CAN_FAIL, 'D', false, BINNED>),
                           gDim, bDim,
                           m, row_ind, /*k, n,*/ d_ia, d_ja, d_ib, d_jb,
                           d_ghash_i, d_ghash_j, d_rc, d_rf );
   }

   if (NEED_GHASH)
   {
      hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

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
#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRowUpperbound");
#endif

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());
   const HYPRE_Int SHMEM_HASH_SIZE = HYPRE_SPGEMM_SYMBL_HASH_SIZE;
   const HYPRE_Int GROUP_SIZE = HYPRE_WARP_SIZE;

   if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   if (2 == in_rc)
   {
      hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, true, false, false>
         (m, NULL, k, n, hash_type, d_ia, d_ja, d_ib, d_jb, d_rc, NULL);
   }
   else if (1 == in_rc)
   {
      hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, true, true, false>
         (m, NULL, k, n, hash_type, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);
   }
   else if (0 == in_rc)
   {
      hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, false, true, false>
         (m, NULL, k, n, hash_type, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);
   }


#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
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
#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRownnz");
#endif

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   char *d_rf = NULL;

   if (in_rc < 2)
   {
      d_rf = hypre_TAlloc(char, m, HYPRE_MEMORY_DEVICE);
   }

   hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, in_rc, d_rc, d_rf);

   if (in_rc < 2)
   {
      const HYPRE_Int SHMEM_HASH_SIZE = HYPRE_SPGEMM_SYMBL_HASH_SIZE;
      const HYPRE_Int GROUP_SIZE = HYPRE_WARP_SIZE;

      /* row nnz is exact if no row failed */
      HYPRE_Int num_failed_rows = HYPRE_THRUST_CALL( reduce,
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

         hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, true, false, 0>
            (num_failed_rows, rf_ind, k, n, hash_type, d_ia, d_ja, d_ib, d_jb, d_rc, NULL);

         hypre_TFree(rf_ind, HYPRE_MEMORY_DEVICE);
      }
   }

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

#define HYPRE_SPGEMM_ROWNNZ_BINNED(b, SHMEM_HASH_SIZE, GROUP_SIZE, GHASH, CAN_FAIL)    \
{                                                                                      \
   const HYPRE_Int p = h_bin_ptr[b - 1];                                               \
   const HYPRE_Int q = h_bin_ptr[b];                                                   \
   const HYPRE_Int bs = q - p;                                                         \
   printf0("bin[%d]: %d rows\n", b, bs);                                               \
   if (bs)                                                                             \
   {                                                                                   \
      hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, GHASH, CAN_FAIL, true>  \
      (bs, d_rc_indice + p, k, n, hash_type, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);      \
   }                                                                                   \
   HYPRE_SPGEMM_ROW(_spgemm_nrows, bs);                                                \
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
#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRownnzBinned");
#endif

#ifdef HYPRE_SPGEMM_TIMING
   HYPRE_Real t1, t2;
#endif

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   HYPRE_Int *d_rc_indice = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
   char      *d_rf        = hypre_TAlloc(char,      m, HYPRE_MEMORY_DEVICE);
   HYPRE_Int  h_bin_ptr[HYPRE_SPGEMM_NBIN + 1];
   const HYPRE_Int s = 32, t = 3, u = 5;

#if defined(HYPRE_DEBUG)
   HYPRE_Int _spgemm_nrows = 0;
#endif

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   t1 = hypre_MPI_Wtime();
#endif

   /* naive overestimate */
   hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, 1);

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   printf0("%s[%d]: Naive upper %f\n", __func__, __LINE__, t2);
#endif

   /* create binning */
#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   t1 = hypre_MPI_Wtime();
#endif

   hypre_SpGemmCreateBins<s, t, u>(m, d_rc, d_rc_indice, h_bin_ptr);

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncCudaComputeStream(hypre_handle());
   t2 = hypre_MPI_Wtime() - t1;
   printf0("%s[%d]: Binning time %f\n", __func__, __LINE__, t2);
#endif

#if 0
   HYPRE_SPGEMM_ROWNNZ_BINNED( 1,  HYPRE_SPGEMM_SYMBL_HASH_SIZE / 16,
                                   HYPRE_WARP_SIZE / 16, false, false);  /* 32,     2 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 2,  HYPRE_SPGEMM_SYMBL_HASH_SIZE /  8,
                                   HYPRE_WARP_SIZE /  8, false, false);  /* 64,     4 */
#endif
   HYPRE_SPGEMM_ROWNNZ_BINNED( 3,  HYPRE_SPGEMM_SYMBL_HASH_SIZE / 4,
                                   HYPRE_WARP_SIZE / 4, false, false);  /* 128,     8 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 4,  HYPRE_SPGEMM_SYMBL_HASH_SIZE / 2,
                                   HYPRE_WARP_SIZE / 2, false, false);  /* 256,    16 */
   HYPRE_SPGEMM_ROWNNZ_BINNED( 5,  HYPRE_SPGEMM_SYMBL_HASH_SIZE,
                                   HYPRE_WARP_SIZE,     false, true);    /* 512,    32 */

#if 0
#endif

#if defined(HYPRE_DEBUG)
   if (_spgemm_nrows != m)
   {
      exit(1);
   }
   hypre_assert(_spgemm_nrows == m);
#endif

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

