/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*- - - - - - - - - - - - - - - - - - - - - - - - - -
 *- - - - - - - - - - - - - - - - - - - - - - - - - -
                Symbolic Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - -
 *- - - - - - - - - - - - - - - - - - - - - - - - - -
 */
template <char HashType>
static __device__ __forceinline__
HYPRE_Int
hash_insert_symbl(HYPRE_Int            HashSize,      /* capacity of the hash table */
                  volatile HYPRE_Int  *HashKeys,      /* assumed to be initialized as all -1's */
                  HYPRE_Int            key,           /* assumed to be nonnegative */
                  HYPRE_Int           &count          /* increase by 1 if is a new entry */)
{
#pragma unroll
   for (HYPRE_Int i = 0; i < HashSize; i++)
   {
      HYPRE_Int j;
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
csr_spmm_compute_row_symbl(HYPRE_Int   rowi,                HYPRE_Int  lane_id,
                           HYPRE_Int  *ia,                  HYPRE_Int *ja,
                           HYPRE_Int  *ib,                  HYPRE_Int *jb,
                           HYPRE_Int   s_HashSize, volatile HYPRE_Int *s_HashKeys,
                           HYPRE_Int   g_HashSize,          HYPRE_Int *g_HashKeys,
                           char &failed)
{
   /* load the start and end position of row i of A */
   HYPRE_Int j = -1;
   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
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
      //const HYPRE_Int ymask = get_mask<4>(lane_id);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      const HYPRE_Int rowB = __shfl_sync(HYPRE_WARP_FULL_MASK, colA, 0, blockDim.x);
      /* open this row of B, collectively */
      HYPRE_Int tmp = -1;
      if (rowB != -1 && threadIdx.x < 2)
      {
         tmp = read_only_load(ib+rowB+threadIdx.x);
      }
      const HYPRE_Int rowB_start = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 0, blockDim.x);
      const HYPRE_Int rowB_end   = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 1, blockDim.x);

      for (HYPRE_Int k = rowB_start; k < rowB_end; k += blockDim.x)
      {
         if (k + threadIdx.x < rowB_end)
         {
            const HYPRE_Int k_idx = read_only_load(jb + k + threadIdx.x);
            /* first try to insert into shared memory hash table */
            HYPRE_Int pos = hash_insert_symbl<HashType>(s_HashSize, s_HashKeys, k_idx, num_new_insert);
            if (-1 == pos)
            {
               pos = hash_insert_symbl<HashType>(g_HashSize, g_HashKeys, k_idx, num_new_insert);
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

template <HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int ATTEMPT, char HashType>
__global__
void csr_spmm_symbolic(HYPRE_Int  M, /* HYPRE_Int K, HYPRE_Int N, */
                       HYPRE_Int *ia, HYPRE_Int *ja,
                       HYPRE_Int *ib, HYPRE_Int *jb,
                       HYPRE_Int *ig, HYPRE_Int *jg,
                       HYPRE_Int *rc, HYPRE_Int *rf)
{
   volatile const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   /* warp id inside the block */
   volatile const HYPRE_Int warp_id = get_warp_id();
   /* warp id in the grid */
   volatile const HYPRE_Int grid_warp_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
   /* lane id inside the warp */
   volatile HYPRE_Int lane_id = get_lane_id();
   /* shared memory hash table */
   __shared__ volatile HYPRE_Int  s_HashKeys[NUM_WARPS_PER_BLOCK * SHMEM_HASH_SIZE];
   /* shared memory hash table for this warp */
   volatile HYPRE_Int *warp_s_HashKeys = s_HashKeys + warp_id * SHMEM_HASH_SIZE;

   char failed = 0;

   hypre_device_assert(blockDim.z              == NUM_WARPS_PER_BLOCK);
   hypre_device_assert(blockDim.x * blockDim.y == HYPRE_WARP_SIZE);
   hypre_device_assert(NUM_WARPS_PER_BLOCK <= HYPRE_WARP_SIZE);

   for (HYPRE_Int i = grid_warp_id; i < M; i += num_warps)
   {
      HYPRE_Int j;

      if (ATTEMPT == 2)
      {
         if (lane_id == 0)
         {
            j = rf[i];
         }
         j = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
         if (j == 0)
         {
            continue;
         }
      }

      /* start/end position of global memory hash table */
      HYPRE_Int istart_g, iend_g, ghash_size;
      if (lane_id < 2)
      {
         j = read_only_load(ig + grid_warp_id + lane_id);
      }
      istart_g = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
      iend_g   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);

      /* size of global hash table allocated for this row
         (must be power of 2 and >= the actual size of the row of C) */
      ghash_size = iend_g - istart_g;

      /* initialize warp's shared and global memory hash table */
#pragma unrolll
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += HYPRE_WARP_SIZE)
      {
         warp_s_HashKeys[k] = -1;
      }
#pragma unrolll
      for (HYPRE_Int k = lane_id; k < ghash_size; k += HYPRE_WARP_SIZE)
      {
         jg[istart_g+k] = -1;
      }
      __syncwarp();

      /* work with two hash tables */
      j = csr_spmm_compute_row_symbl<HashType>(i, lane_id, ia, ja, ib, jb,
                                               SHMEM_HASH_SIZE, warp_s_HashKeys,
                                               ghash_size, jg + istart_g, failed);

#if defined(HYPRE_DEBUG)
      if (ATTEMPT == 2)
      {
         hypre_device_assert(failed == 0);
      }
#endif

      /* num of nonzeros of this row (an upper bound) */
      j = warp_reduce_sum(j);

      /* if this row failed */
      if (ATTEMPT == 1)
      {
         failed = warp_reduce_sum(failed);
      }

      if (lane_id == 0)
      {
         rc[i] = j;
         if (ATTEMPT == 1)
         {
            rf[i] = failed > 0;
         }
#ifdef HYPRE_DEBUG
         else
         {
            rf[i] = failed > 0;
         }
#endif
      }
   }
}

template <HYPRE_Int ATTEMPT>
void gpu_csr_spmm_rownnz_attempt(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                 HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb,
                                 HYPRE_Int *d_rc, HYPRE_Int *d_rf)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] -= hypre_MPI_Wtime();
#endif

   const HYPRE_Int num_warps_per_block =  20;
   const HYPRE_Int shmem_hash_size     = 256;//512;
   const HYPRE_Int BDIMX               =   2;
   const HYPRE_Int BDIMY               =  16;

   /* CUDA kernel configurations */
   dim3 bDim(BDIMX, BDIMY, num_warps_per_block);
   hypre_assert(bDim.x * bDim.y == HYPRE_WARP_SIZE);
   // for cases where one WARP works on a row
   HYPRE_Int num_warps = min(m, HYPRE_MAX_NUM_WARPS);
   dim3 gDim( (num_warps + bDim.z - 1) / bDim.z );
   // number of active warps
   HYPRE_Int num_act_warps = min(bDim.z * gDim.x, m);

   char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/
   HYPRE_Int  *d_ghash_i, *d_ghash_j, ghash_size, *d_act;
   if (ATTEMPT == 1)
   {
      d_act = NULL; /* all rows are active */
   }
   else
   {
      d_act = d_rf;
   }
   csr_spmm_create_hash_table(m, d_rc, d_act, shmem_hash_size, num_act_warps,
                              &d_ghash_i, &d_ghash_j, NULL, &ghash_size);

   /* ---------------------------------------------------------------------------
    * symbolic multiplication:
    * On output, it provides an upper bound of nnz in rows of C
    * ---------------------------------------------------------------------------*/
   if (hash_type == 'L')
   {
      HYPRE_CUDA_LAUNCH( (csr_spmm_symbolic<num_warps_per_block, shmem_hash_size, ATTEMPT, 'L'>), gDim, bDim,
                         m, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
   else if (hash_type == 'Q')
   {
      HYPRE_CUDA_LAUNCH( (csr_spmm_symbolic<num_warps_per_block, shmem_hash_size, ATTEMPT, 'Q'>), gDim, bDim,
                         m, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
   else if (hash_type == 'D')
   {
      HYPRE_CUDA_LAUNCH( (csr_spmm_symbolic<num_warps_per_block, shmem_hash_size, ATTEMPT, 'D'>), gDim, bDim,
                         m, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
   else
   {
      printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_SYMBOLIC] += hypre_MPI_Wtime();
#endif
}

HYPRE_Int
hypreDevice_CSRSpGemmRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                      HYPRE_Int *d_ia, HYPRE_Int *d_ja,
                                      HYPRE_Int *d_ib, HYPRE_Int *d_jb,
                                      HYPRE_Int *d_rc, HYPRE_Int *d_rf)
{
   gpu_csr_spmm_rownnz_attempt<1> (m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);

   return hypre_error_flag;
}

HYPRE_Int
hypreDevice_CSRSpGemmRownnz(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                            HYPRE_Int *d_ia, HYPRE_Int *d_ja,
                            HYPRE_Int *d_ib, HYPRE_Int *d_jb,
                            HYPRE_Int *d_rc)
{
   /* a binary array to indicate if row nnz counting is failed for a row */
   HYPRE_Int *d_rf = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);

   gpu_csr_spmm_rownnz_attempt<1> (m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);

   /* row nnz is exact if no row failed */
   HYPRE_Int rownnz_exact = hypreDevice_IntegerReduceSum(m, d_rf);

   printf("^^^^num of failed rows                                    %d (%.2f)\n", rownnz_exact, rownnz_exact/(m+0.0));

   if (rownnz_exact != 0)
   {
      gpu_csr_spmm_rownnz_attempt<2> (m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);

      hypre_assert(hypreDevice_IntegerReduceSum(m, d_rf) == 0);
   }

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */
