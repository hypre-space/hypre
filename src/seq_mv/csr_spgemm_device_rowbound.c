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

template <char HashType>
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
         j = HashFunc<HashType>(HashSize, key, i, j);
      }

      /* try to insert key+1 into slot j */
      HYPRE_Int old = atomicCAS((HYPRE_Int*)(HashKeys+j), -1, key);

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

template <char HashType>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_compute_row_symbl( HYPRE_Int  rowi,
                                HYPRE_Int  warp_lane_id,
                                HYPRE_Int *ia,
                                HYPRE_Int *ja,
                                HYPRE_Int *ib,
                                HYPRE_Int *jb,
                                HYPRE_Int  s_HashSize,
                       volatile HYPRE_Int *s_HashKeys,
                                HYPRE_Int  g_HashSize,
                                HYPRE_Int *g_HashKeys,
                                hypre_int &failed )
{
   /* load the start and end position of row i of A */
   HYPRE_Int j = 0;

   if (warp_lane_id < 2)
   {
      j = read_only_load(ia + rowi + warp_lane_id);
   }
   const HYPRE_Int istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   const HYPRE_Int iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);

   HYPRE_Int num_new_insert = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart; i < iend; i += blockDim.y)
   {
      HYPRE_Int colA = -1;

      if (threadIdx.x == 0 && i + threadIdx.y < iend)
      {
         colA = read_only_load(ja + i + threadIdx.y);
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(warp_lane_id);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      const HYPRE_Int rowB = __shfl_sync(HYPRE_WARP_FULL_MASK, colA, 0, blockDim.x);
      /* open this row of B, collectively */
      HYPRE_Int tmp = 0;
      if (rowB != -1 && threadIdx.x < 2)
      {
         tmp = read_only_load(ib+rowB+threadIdx.x);
      }
      const HYPRE_Int rowB_start = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 0, blockDim.x);
      const HYPRE_Int rowB_end   = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 1, blockDim.x);

      for (HYPRE_Int k = rowB_start + threadIdx.x; __any_sync(HYPRE_WARP_FULL_MASK, k < rowB_end); k += blockDim.x)
      {
         if (k < rowB_end)
         {
            const HYPRE_Int k_idx = read_only_load(jb + k);
            /* first try to insert into shared memory hash table */
            HYPRE_Int pos = hypre_spgemm_hash_insert_symbl<HashType>(s_HashSize, s_HashKeys, k_idx, num_new_insert);

            if (-1 == pos)
            {
               pos = hypre_spgemm_hash_insert_symbl<HashType>(g_HashSize, g_HashKeys, k_idx, num_new_insert);
            }
            /* if failed again, both hash tables must have been full
               (hash table size estimation was too small).
               Increase the counter anyhow (will lead to over-counting)
               */
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

template <HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int ATTEMPT, char HashType>
__global__ void
hypre_spgemm_symbolic( HYPRE_Int  M, /* HYPRE_Int K, HYPRE_Int N, */
                       HYPRE_Int *rind,
                       HYPRE_Int *ia,
                       HYPRE_Int *ja,
                       HYPRE_Int *ib,
                       HYPRE_Int *jb,
                       HYPRE_Int *ig,
                       HYPRE_Int *jg,
                       HYPRE_Int *rc,
                       HYPRE_Int *rf)
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

   for (HYPRE_Int i = grid_group_id; i < M; i += grid_num_groups)
   {
      HYPRE_Int ii, jsum;
      hypre_int failed = 0;

      if (ATTEMPT == 2)
      {
         if (warp_lane_id == 0)
         {
            ii = read_only_load(&rind[i]);
         }
         ii = __shfl_sync(HYPRE_WARP_FULL_MASK, ii, 0);
      }
      else
      {
         ii = i;
      }

      /* start/end position of global memory hash table */
      HYPRE_Int istart_g = 0, iend_g = 0, ghash_size = 0;

      if (ig)
      {
         if (warp_lane_id < 2)
         {
            istart_g = read_only_load(ig + grid_group_id + warp_lane_id);
         }
         iend_g   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_g, 1);
         istart_g = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_g, 0);

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
#pragma unroll
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
      {
         group_s_HashKeys[k] = -1;
      }

      group_sync<GROUP_SIZE>();

      /* work with two hash tables */
      jsum = hypre_spgemm_compute_row_symbl<HashType>(ii, warp_lane_id, ia, ja, ib, jb,
                                                      SHMEM_HASH_SIZE, group_s_HashKeys,
                                                      ghash_size, jg + istart_g, failed);

#if defined(HYPRE_DEBUG)
      if (ATTEMPT == 2)
      {
         hypre_device_assert(failed == 0);
      }
#endif

      /* num of nonzeros of this row (an upper bound) */
      jsum = group_reduce_sum<HYPRE_Int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(jsum);

      /* if this row failed */
      if (ATTEMPT == 1)
      {
         failed = group_reduce_sum<HYPRE_Int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(failed);
      }

      if (lane_id == 0)
      {
         rc[ii] = jsum;

         if (ATTEMPT == 1)
         {
            rf[ii] = failed > 0;
         }
      }
   }
}

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE, HYPRE_Int ATTEMPT>
void
hypre_spgemm_rownnz_attempt(HYPRE_Int  m,
                            HYPRE_Int *rf_ind, /* ATTEMPT = 2, input: row indices (length of m) */
                            HYPRE_Int  k,
                            HYPRE_Int  n,
                            HYPRE_Int *d_ia,
                            HYPRE_Int *d_ja,
                            HYPRE_Int *d_ib,
                            HYPRE_Int *d_jb,
                            HYPRE_Int  in_rc,
                            HYPRE_Int *d_rc, /* ATTEMPT = 1,  input: rownnz est (if in_rc), output: rownnz bound;
                                                ATTEMPT = 2,  input: rownnz bound, output: rownnz (exact) */
                            HYPRE_Int *d_rf  /* ATTEMPT = 1, output: if symbolic mult. failed  */ )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_CUDA)
   const HYPRE_Int num_warps_per_block   = 16;
   const HYPRE_Int BDIMX                 =  2;
#elif defined(HYPRE_USING_HIP)
   const HYPRE_Int num_warps_per_block   = 16;
   const HYPRE_Int BDIMX                 =  4;
#endif
   const HYPRE_Int BDIMY                 = GROUP_SIZE / BDIMX;
   const HYPRE_Int num_threads_per_block = num_warps_per_block * HYPRE_WARP_SIZE;
   const HYPRE_Int num_groups_per_block  = num_threads_per_block / GROUP_SIZE;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
   hypre_assert(bDim.x * bDim.y == GROUP_SIZE);
   // total number of threads used: a group works on a row
   HYPRE_Int num_threads = hypre_min((size_t) m * GROUP_SIZE, HYPRE_MAX_NUM_WARPS * HYPRE_WARP_SIZE);
   // grid dimension (number of blocks)
   dim3 gDim( (num_threads + num_threads_per_block - 1) / num_threads_per_block );
   // number of active groups
   HYPRE_Int num_act_groups = hypre_min(bDim.z * gDim.x, m);

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/
   HYPRE_Int *d_ghash_i = NULL, *d_ghash_j = NULL;
   HYPRE_Int  ghash_size = 0;

   if (in_rc)
   {
      hypre_SpGemmCreateGlobalHashTable(m, rf_ind, num_act_groups, d_rc, SHMEM_HASH_SIZE,
                                        &d_ghash_i, &d_ghash_j, NULL, &ghash_size);

      hypre_printf("%s[%d]: ghash size %d\n", __func__, __LINE__, ghash_size);
   }

   /*
   if (in_rc)
   {
      int *result = HYPRE_THRUST_CALL(max_element, d_rc, d_rc + m);
      int *result2 = HYPRE_THRUST_CALL(min_element, d_rc, d_rc + m);
      int t, t2;
      hypre_TMemcpy(&t, result, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(&t2, result2, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      printf("min = %d, max = %d\n", t2, t);
   }
   */

   printf("%s[%d]: SHMEM_HASH_SIZE %d, ghash %p\n", __func__, __LINE__, SHMEM_HASH_SIZE, d_ghash_i);

   /* ---------------------------------------------------------------------------
    * symbolic multiplication:
    * On output, it provides an upper bound of nnz in rows of C
    * ---------------------------------------------------------------------------*/
   if (hash_type == 'L')
   {
      HYPRE_CUDA_LAUNCH( (hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, ATTEMPT, 'L'>), gDim, bDim,
                          m, rf_ind, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
   else if (hash_type == 'Q')
   {
      HYPRE_CUDA_LAUNCH( (hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, ATTEMPT, 'Q'>), gDim, bDim,
                         m, rf_ind, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
   else if (hash_type == 'D')
   {
      HYPRE_CUDA_LAUNCH( (hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, ATTEMPT, 'D'>), gDim, bDim,
                         m, rf_ind, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }

   hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif
}

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
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
                                       HYPRE_Int *d_rf )
{
#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmRowUpperbound");
#endif

   hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, 1> (m, NULL, k, n, d_ia, d_ja, d_ib, d_jb, in_rc, d_rc, d_rf);

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

//template HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound<HYPRE_SPGEMM_SYMBL_HASH_SIZE, HYPRE_WARP_SIZE>(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc, HYPRE_Int *d_rf);

template HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound<2*HYPRE_SPGEMM_SYMBL_HASH_SIZE, 2*HYPRE_WARP_SIZE>(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc, HYPRE_Int *d_rf);

//template HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound<4*HYPRE_SPGEMM_SYMBL_HASH_SIZE, 4*HYPRE_WARP_SIZE>(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc, HYPRE_Int *d_rf);


template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
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
   hypre_GpuProfilingPushRange("CSRSpGemmRowNNZ");
#endif

   /* a binary array to indicate if row nnz counting is failed for a row */
   HYPRE_Int *d_rf  = d_rc + m;

   hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, 1> (m, NULL, k, n, d_ia, d_ja, d_ib, d_jb, in_rc, d_rc, d_rf);

   /* row nnz is exact if no row failed */
   HYPRE_Int num_failed_rows = hypreDevice_IntegerReduceSum(m, d_rf);

   if (num_failed_rows)
   {
      //hypre_printf("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__, num_failed_rows, num_failed_rows / (m + 0.0) );

      HYPRE_Int *rf_ind = hypre_TAlloc(HYPRE_Int, num_failed_rows, HYPRE_MEMORY_DEVICE);

      HYPRE_Int *new_end =
      HYPRE_THRUST_CALL( copy_if,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(m),
                         d_rf,
                         rf_ind,
                         thrust::identity<HYPRE_Int>() );

      hypre_assert(new_end - rf_ind == num_failed_rows);

      hypre_spgemm_rownnz_attempt<SHMEM_HASH_SIZE, GROUP_SIZE, 2> (num_failed_rows, rf_ind, k, n, d_ia, d_ja, d_ib, d_jb, 1, d_rc, NULL);

      hypre_TFree(rf_ind, HYPRE_MEMORY_DEVICE);
   }

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_CSRSpGemmRownnz<HYPRE_SPGEMM_SYMBL_HASH_SIZE, HYPRE_WARP_SIZE>(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc);

template HYPRE_Int hypreDevice_CSRSpGemmRownnz<2*HYPRE_SPGEMM_SYMBL_HASH_SIZE, 2*HYPRE_WARP_SIZE>(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc);

//template HYPRE_Int hypreDevice_CSRSpGemmRownnz<HYPRE_SPGEMM_SYMBL_HASH_SIZE, 4*HYPRE_WARP_SIZE>(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc);

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

