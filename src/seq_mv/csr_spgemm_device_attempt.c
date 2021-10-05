/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
           Perform SpMM with Row Nnz Estimation
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
#include "seq_mv.h"
#include "csr_spgemm_device.h"

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Numerical Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

template <char HashType>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_attempt( HYPRE_Int      HashSize,      /* capacity of the hash table */
                         volatile HYPRE_Int     *HashKeys,      /* assumed to be initialized as all -1's */
                         volatile HYPRE_Complex *HashVals,      /* assumed to be initialized as all 0's */
                                  HYPRE_Int      key,           /* assumed to be nonnegative */
                                  HYPRE_Complex  val,
                                  HYPRE_Int     &count,         /* increase by 1 if is a new entry */
                                  hypre_int      warp_failed )
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

      if (old == -1 || old == key)
      {
         /* new insertion, increase counter */
         if (old == -1)
         {
            count++;
         }

         /* this slot was open or contained 'key', update value */
         if (!warp_failed)
         {
            atomicAdd((HYPRE_Complex*)(HashVals+j), val);
         }

         return j;
      }
   }

   return -1;
}

template <HYPRE_Int ATTEMPT, char HashType>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_compute_row_attempt( HYPRE_Int      rowi,
                         volatile HYPRE_Int      lane_id,
                                  HYPRE_Int     *ia,
                                  HYPRE_Int     *ja,
                                  HYPRE_Complex *aa,
                                  HYPRE_Int     *ib,
                                  HYPRE_Int     *jb,
                                  HYPRE_Complex *ab,
                                  HYPRE_Int      s_HashSize,
                         volatile HYPRE_Int     *s_HashKeys,
                         volatile HYPRE_Complex *s_HashVals,
                                  HYPRE_Int      g_HashSize,
                                  HYPRE_Int     *g_HashKeys,
                                  HYPRE_Complex *g_HashVals,
                                  hypre_int     &failed )
{
   /* load the start and end position of row i of A */
   HYPRE_Int j = 0;

   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }
   const HYPRE_Int istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   const HYPRE_Int iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);

   HYPRE_Int num_new_insert = 0;
   hypre_int warp_failed    = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart; i < iend; i += blockDim.y)
   {
      HYPRE_Int     colA = -1;
      HYPRE_Complex valA = 0.0;

      if (threadIdx.x == 0 && i + threadIdx.y < iend)
      {
         colA = read_only_load(ja + i + threadIdx.y);
         valA = read_only_load(aa + i + threadIdx.y);
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(lane_id);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      const HYPRE_Int     rowB = __shfl_sync(HYPRE_WARP_FULL_MASK, colA, 0, blockDim.x);
      const HYPRE_Complex mult = __shfl_sync(HYPRE_WARP_FULL_MASK, valA, 0, blockDim.x);
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
            const HYPRE_Int     k_idx = read_only_load(jb + k);
            const HYPRE_Complex k_val = read_only_load(ab + k) * mult;
            /* first try to insert into shared memory hash table */
            HYPRE_Int pos = hypre_spgemm_hash_insert_attempt<HashType>
               (s_HashSize, s_HashKeys, s_HashVals, k_idx, k_val, num_new_insert, warp_failed);

            if (-1 == pos)
            {
               pos = hypre_spgemm_hash_insert_attempt<HashType>
                     (g_HashSize, g_HashKeys, g_HashVals, k_idx, k_val, num_new_insert, warp_failed);
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

         if (ATTEMPT == 1)
         {
            if (!warp_failed)
            {
               warp_failed = warp_allreduce_sum(failed);
            }
         }
      }
   }

   return num_new_insert;
}

template <HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_HASH_SIZE>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_copy_from_hash_into_C_row( HYPRE_Int      lane_id,
                               volatile HYPRE_Int     *s_HashKeys,
                               volatile HYPRE_Complex *s_HashVals,
                                        HYPRE_Int      ghash_size,
                                        HYPRE_Int     *jg_start,
                                        HYPRE_Complex *ag_start,
                                        HYPRE_Int     *jc_start,
                                        HYPRE_Complex *ac_start )
{
   HYPRE_Int j = 0;

   /* copy shared memory hash table into C */
#pragma unroll
   for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += HYPRE_WARP_SIZE)
   {
      HYPRE_Int key, sum, pos;
      key = s_HashKeys[k];
      HYPRE_Int in = key != -1;
      pos = warp_prefix_sum(lane_id, in, sum);
      if (key != -1)
      {
         jc_start[j + pos] = key;
         ac_start[j + pos] = s_HashVals[k];
      }
      j += sum;
   }

   /* copy global memory hash table into C */
#pragma unroll
   for (HYPRE_Int k = 0; k < ghash_size; k += HYPRE_WARP_SIZE)
   {
      HYPRE_Int key = -1, sum, pos;
      if (k + lane_id < ghash_size)
      {
         key = jg_start[k + lane_id];
      }
      HYPRE_Int in = key != -1;
      pos = warp_prefix_sum(lane_id, in, sum);
      if (key != -1)
      {
         jc_start[j + pos] = key;
         ac_start[j + pos] = ag_start[k + lane_id];
      }
      j += sum;
   }

   return j;
}

template <HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int ATTEMPT, char HashType>
__global__ void
hypre_spgemm_attempt( HYPRE_Int      M, /* HYPRE_Int K, HYPRE_Int N, */
                      HYPRE_Int     *rind,
                      HYPRE_Int     *ia,
                      HYPRE_Int     *ja,
                      HYPRE_Complex *aa,
                      HYPRE_Int     *ib,
                      HYPRE_Int     *jb,
                      HYPRE_Complex *ab,
                      HYPRE_Int     *ig,
                      HYPRE_Int     *jg,
                      HYPRE_Complex *ag,
                      HYPRE_Int     *ic,
                      HYPRE_Int     *jc,
                      HYPRE_Complex *ac,
                      HYPRE_Int     *rc,
                      HYPRE_Int     *rf )
{
   volatile const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   /* warp id inside the block */
   volatile const HYPRE_Int warp_id = get_warp_id();
   /* warp id in the grid */
   volatile const HYPRE_Int grid_warp_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
   /* lane id inside the warp */
   volatile HYPRE_Int lane_id = get_lane_id();
   /* shared memory hash table */
   __shared__ volatile HYPRE_Int s_HashKeys[NUM_WARPS_PER_BLOCK * SHMEM_HASH_SIZE];
   __shared__ volatile HYPRE_Complex s_HashVals[NUM_WARPS_PER_BLOCK * SHMEM_HASH_SIZE];
   /* shared memory hash table for this warp */
   volatile HYPRE_Int *warp_s_HashKeys = s_HashKeys + warp_id * SHMEM_HASH_SIZE;
   volatile HYPRE_Complex *warp_s_HashVals = s_HashVals + warp_id * SHMEM_HASH_SIZE;

   hypre_device_assert(blockDim.z              == NUM_WARPS_PER_BLOCK);
   hypre_device_assert(blockDim.x * blockDim.y == HYPRE_WARP_SIZE);
   hypre_device_assert(NUM_WARPS_PER_BLOCK     <= HYPRE_WARP_SIZE);

   for (HYPRE_Int i = grid_warp_id; i < M; i += num_warps)
   {
      HYPRE_Int ii, jsum;
      hypre_int failed = 0;

      if (ATTEMPT == 2)
      {
         if (lane_id == 0)
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
         if (lane_id < 2)
         {
            istart_g = read_only_load(ig + grid_warp_id + lane_id);
         }
         iend_g   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_g, 1);
         istart_g = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_g, 0);

         /* size of global hash table allocated for this row
           (must be power of 2) */
         ghash_size = iend_g - istart_g;

         /* initialize warp's global memory hash table */
#pragma unroll
         for (HYPRE_Int k = lane_id; k < ghash_size; k += HYPRE_WARP_SIZE)
         {
            jg[istart_g + k] = -1;
            ag[istart_g + k] = 0.0;
         }
      }

      /* initialize warp's shared memory hash table */
#pragma unroll
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += HYPRE_WARP_SIZE)
      {
         warp_s_HashKeys[k] = -1;
         warp_s_HashVals[k] = 0.0;
      }

      __syncwarp();

      /* work with two hash tables */
      jsum = hypre_spgemm_compute_row_attempt<ATTEMPT, HashType>(ii, lane_id, ia, ja, aa, ib, jb, ab,
                                                                 SHMEM_HASH_SIZE, warp_s_HashKeys, warp_s_HashVals,
                                                                 ghash_size, jg + istart_g, ag + istart_g, failed);

      /* num of inserts in this row (an upper bound) */
      jsum = warp_allreduce_sum(jsum);

      /* if this row failed */
      if (ATTEMPT == 1)
      {
         failed = warp_allreduce_sum(failed);
      }
#if defined(HYPRE_DEBUG)
      else if (ATTEMPT == 2)
      {
         hypre_device_assert(failed == 0);
      }
#endif

      HYPRE_Int istart_c = 0, iend_c = 0;

      if (!failed)
      {
         if (lane_id < 2)
         {
            istart_c = read_only_load(ic + i + lane_id);
         }
         iend_c   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 1);
         istart_c = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 0);

         failed = (iend_c - istart_c) < jsum;
      }

#if defined(HYPRE_DEBUG)
      if (ATTEMPT == 2)
      {
         hypre_device_assert(failed == 0);
      }
#endif

      if (lane_id == 0)
      {
         rc[ii] = jsum;

         if (ATTEMPT == 1)
         {
            rf[ii] = failed > 0;
         }
      }

      if (!failed)
      {
         HYPRE_Int j = hypre_spgemm_copy_from_hash_into_C_row<NUM_WARPS_PER_BLOCK, SHMEM_HASH_SIZE>
            (lane_id, warp_s_HashKeys, warp_s_HashVals, ghash_size, jg + istart_g,
             ag + istart_g, jc + istart_c, ac + istart_c);

#if defined(HYPRE_DEBUG)
         hypre_device_assert(istart_c + j <= iend_c);
#endif
      }
   } // for (i=...)
}

template <HYPRE_Int NUM_WARPS_PER_BLOCK>
__global__ void
hypre_spgemm_copy_into_C( HYPRE_Int      M,
                          HYPRE_Int     *rf,
                          HYPRE_Int     *ic1,
                          HYPRE_Int     *jc1,
                          HYPRE_Complex *ac1,
                          HYPRE_Int     *jc2,
                          HYPRE_Complex *ac2,
                          HYPRE_Int     *ic,
                          HYPRE_Int     *jc,
                          HYPRE_Complex *ac)
{
   const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   /* warp id inside the block */
   const HYPRE_Int warp_id = get_warp_id();
   /* warp id in the grid */
   volatile const HYPRE_Int grid_warp_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
   /* lane id inside the warp */
   volatile const HYPRE_Int lane_id = get_lane_id();

   hypre_device_assert(blockDim.x * blockDim.y == HYPRE_WARP_SIZE);

   for (HYPRE_Int i = grid_warp_id; i < M; i += num_warps)
   {
      HYPRE_Int irf = 0, istart_c = 0, iend_c = 0, istart_x = 0;

      if (lane_id == 0)
      {
         istart_x = read_only_load(ic1 + i);
         irf      = read_only_load(rf  + i);
      }

      irf      = __shfl_sync(HYPRE_WARP_FULL_MASK, irf,      0);
      istart_x = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_x, 0);

      if (lane_id < 2)
      {
         istart_c = read_only_load(ic + i + lane_id);
      }

      iend_c   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 1);
      istart_c = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 0);

      if (irf)
      {
         for (HYPRE_Int k = istart_c + lane_id; k < iend_c; k += HYPRE_WARP_SIZE)
         {
            jc[k] = jc2[k + istart_x - istart_c];
            ac[k] = ac2[k + istart_x - istart_c];
         }
      }
      else
      {
         for (HYPRE_Int k = istart_c + lane_id; k < iend_c; k += HYPRE_WARP_SIZE)
         {
            jc[k] = jc1[k + istart_x - istart_c];
            ac[k] = ac1[k + istart_x - istart_c];
         }
      }
   }
}

/* SpGeMM with Rownnz Estimates */
template <HYPRE_Int shmem_hash_size, HYPRE_Int ATTEMPT>
HYPRE_Int
hypre_spgemm_numerical_with_rowest( HYPRE_Int       m,
                                    HYPRE_Int      *rf_ind,
                                    HYPRE_Int       k,
                                    HYPRE_Int       n,
                                    HYPRE_Int      *d_ia,
                                    HYPRE_Int      *d_ja,
                                    HYPRE_Complex  *d_a,
                                    HYPRE_Int      *d_ib,
                                    HYPRE_Int      *d_jb,
                                    HYPRE_Complex  *d_b,
                                    HYPRE_Int      *d_rc,
                                    HYPRE_Int      *d_rf,
                                    HYPRE_Int     **d_ic_out,
                                    HYPRE_Int     **d_jc_out,
                                    HYPRE_Complex **d_c_out )
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_CUDA)
   const HYPRE_Int num_warps_per_block = 16;
#elif defined(HYPRE_USING_HIP)
   const HYPRE_Int num_warps_per_block = 16;
#endif
   const HYPRE_Int BDIMX               = 2;
   const HYPRE_Int BDIMY               = HYPRE_WARP_SIZE / BDIMX;

   /* CUDA kernel configurations */
   dim3 bDim(BDIMX, BDIMY, num_warps_per_block);
   hypre_assert(bDim.x * bDim.y == HYPRE_WARP_SIZE);
   // for cases where one WARP works on a row
   HYPRE_Int num_warps = hypre_min(m, HYPRE_MAX_NUM_WARPS);
   dim3 gDim( (num_warps + bDim.z - 1) / bDim.z );
   // number of active warps
   HYPRE_Int num_act_warps = hypre_min(bDim.z * gDim.x, m);

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   /* ---------------------------------------------------------------------------
    * global memory hash table
    * ---------------------------------------------------------------------------*/
   HYPRE_Int     *d_ghash_i = NULL;
   HYPRE_Int     *d_ghash_j = NULL;
   HYPRE_Complex *d_ghash_a = NULL;

   hypre_SpGemmCreateGlobalHashTable(m, rf_ind, num_act_warps, d_rc, shmem_hash_size, &d_ghash_i,
                                     &d_ghash_j, &d_ghash_a, NULL);

   /* allocate tmp C */
   *d_ic_out = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);

   hypre_assert(shmem_hash_size == HYPRE_SPGEMM_NUMER_HASH_SIZE);

   /* in ATTEMPT 1, allocate ija with slightly larger size than in d_rc (raise to power to 2), type == 2
    * in ATTEMPT 2, allocate ija with size in d_rc, type == 1 */
   hypre_create_ija(3 - ATTEMPT, m, rf_ind, d_rc, *d_ic_out, d_jc_out, d_c_out, NULL);

   if (hash_type == 'L')
   {
      HYPRE_CUDA_LAUNCH ( (hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, ATTEMPT, 'L'>), gDim, bDim,
                          m, rf_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ghash_i, d_ghash_j, d_ghash_a,
                          *d_ic_out, *d_jc_out, *d_c_out, d_rc, d_rf );
   }
   else if (hash_type == 'Q')
   {
      HYPRE_CUDA_LAUNCH ( (hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, ATTEMPT, 'Q'>), gDim, bDim,
                          m, rf_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ghash_i, d_ghash_j, d_ghash_a,
                          *d_ic_out, *d_jc_out, *d_c_out, d_rc, d_rf );
   }
   else if (hash_type == 'D')
   {
      HYPRE_CUDA_LAUNCH ( (hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, ATTEMPT, 'D'>), gDim, bDim,
                          m, rf_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ghash_i, d_ghash_j, d_ghash_a,
                          *d_ic_out, *d_jc_out, *d_c_out, d_rc, d_rf );
   }
   else
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_a, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzEstimate( HYPRE_Int       m,
                                              HYPRE_Int       k,
                                              HYPRE_Int       n,
                                              HYPRE_Int      *d_ia,
                                              HYPRE_Int      *d_ja,
                                              HYPRE_Complex  *d_a,
                                              HYPRE_Int      *d_ib,
                                              HYPRE_Int      *d_jb,
                                              HYPRE_Complex  *d_b,
                                              HYPRE_Int      *d_rc,
                                              HYPRE_Int     **d_ic_out,
                                              HYPRE_Int     **d_jc_out,
                                              HYPRE_Complex **d_c_out,
                                              HYPRE_Int      *nnzC_out )
{
   const HYPRE_Int shmem_hash_size = HYPRE_SPGEMM_NUMER_HASH_SIZE;

   /* a binary array to indicate if row nnz counting is failed for a row */
   HYPRE_Int     *d_rf = d_rc + m;
   /* temporary C */
   HYPRE_Int     *d_ic1 = NULL, *d_jc1 = NULL, *d_ic2 = NULL, *d_jc2 = NULL;
   HYPRE_Complex *d_c1 = NULL, *d_c2 = NULL;

   /* attempt 1 */
   hypre_spgemm_numerical_with_rowest<shmem_hash_size, 1>
      (m, NULL, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_rf, &d_ic1, &d_jc1, &d_c1);

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

      /* attempt 2 */
      hypre_spgemm_numerical_with_rowest<shmem_hash_size, 2>
         (num_failed_rows, rf_ind, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, NULL, &d_ic2, &d_jc2, &d_c2);

      HYPRE_THRUST_CALL( scatter,
                         d_ic2,
                         d_ic2 + num_failed_rows,
                         rf_ind,
                         d_ic1 );

      hypre_TFree(rf_ind, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_ic2,  HYPRE_MEMORY_DEVICE);
   }

   /* allocate final C and copy results */
   HYPRE_Int     *d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_jc;
   HYPRE_Complex *d_c;
   HYPRE_Int      nnzC;

   /* d_rc has (exact) row nnz now */
   hypre_create_ija(1, m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC);

   const HYPRE_Int num_warps_per_block = 16;
   dim3 bDim(4, 8, num_warps_per_block);
   dim3 gDim((m + bDim.z - 1) / bDim.z);

   HYPRE_CUDA_LAUNCH( (hypre_spgemm_copy_into_C<num_warps_per_block>), gDim, bDim,
                       m, d_rf, d_ic1, d_jc1, d_c1, d_jc2, d_c2, d_ic, d_jc, d_c );

   hypre_TFree(d_ic1, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jc1, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_c1,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_jc2, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_c2,  HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   return hypre_error_flag;
}
#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */
