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

#define HYPRE_SPGEMM_NUMER_HASH_SIZE 128

#if defined(HYPRE_USING_GPU)

template <char HashType>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_attempt( HYPRE_Int      HashSize,      /* capacity of the hash table */
                                  #ifdef HYPRE_USING_SYCL
                                  HYPRE_Int     *HashKeys,      /* shared, assumed to be initialized as all -1's */
                                  HYPRE_Complex *HashVals,      /* shared, assumed to be initialized as all 0's */
                                  #else
                                  volatile HYPRE_Int     *HashKeys,      /* shared, assumed to be initialized as all -1's */
                                  volatile HYPRE_Complex *HashVals,      /* shared, assumed to be initialized as all 0's */
                                  #endif
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
      HYPRE_Int old = atomicCAS((HYPRE_Int*)(HashKeys + j), -1, key);
#ifdef HYPRE_USING_SYCL
      if (old == 1 /* true */)
      {
#else
      if (old == -1 || old == key)
      {
#endif
         /* new insertion, increase counter */
#ifdef HYPRE_USING_SYCL
         if (old == 0 /* false */)
         {
#else
         if (old == -1)
         {
#endif
            count++;
         }
         /* this slot was open or contained 'key', update value */
         if (!warp_failed)
         {
            atomicAdd((HYPRE_Complex*)(HashVals + j), val);
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
                                  #ifdef HYPRE_USING_SYCL
                                  HYPRE_Int     *s_HashKeys,
                                  HYPRE_Complex *s_HashVals,
                                  #else
                                  volatile HYPRE_Int     *s_HashKeys,
                                  volatile HYPRE_Complex *s_HashVals,
                                  #endif
                                  HYPRE_Int      g_HashSize,
                                  HYPRE_Int     *g_HashKeys,
                                  HYPRE_Complex *g_HashVals,
                                  hypre_int     &failed
#ifdef HYPRE_USING_SYCL
                                  , sycl::nd_item<3>& item
#endif
                                )
{
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int threadIdx_x = item.get_local_id(2);
   HYPRE_Int threadIdx_y = item.get_local_id(1);
   HYPRE_Int blockDim_y = item.get_local_range(1);
   HYPRE_Int blockDim_x = item.get_local_range(2);
#else
   HYPRE_Int threadIdx_x = threadIdx.x;
   HYPRE_Int threadIdx_y = threadIdx.y;
   HYPRE_Int blockDim_y = blockDim.y;
   HYPRE_Int blockDim_x = blockDim.x;
#endif

   /* load the start and end position of row i of A */
   HYPRE_Int j = 0;

   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }
   HYPRE_Int num_new_insert = 0;
   hypre_int warp_failed    = 0;

#ifdef HYPRE_USING_SYCL
   const HYPRE_Int istart = SG.shuffle(j, 0);
   const HYPRE_Int iend   = SG.shuffle(j, 1);
#else
   const HYPRE_Int istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   const HYPRE_Int iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);
#endif

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart; i < iend; i += blockDim_y)
   {
      HYPRE_Int     colA = -1;
      HYPRE_Complex valA = 0.0;

      if (threadIdx_x == 0 && i + threadIdx_y < iend)
      {
         colA = read_only_load(ja + i + threadIdx_y);
         valA = read_only_load(aa + i + threadIdx_y);
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(lane_id);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      HYPRE_Int tmp = 0;

#ifdef HYPRE_USING_SYCL
      /* threads in the same ygroup work on one row together */
      const HYPRE_Int     rowB = SG.shuffle(colA, 0); //, blockDim_x);
      const HYPRE_Complex mult = SG.shuffle(valA, 0); //, blockDim_x);
      /* open this row of B, collectively */
      if (rowB != -1 && threadIdx_x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx_x);
      }
      const HYPRE_Int rowB_start = SG.shuffle(tmp, 0); //, blockDim_x);
      const HYPRE_Int rowB_end   = SG.shuffle(tmp, 1); //, blockDim_x);

      for (HYPRE_Int k = rowB_start + threadIdx_x; sycl::any_of_group(SG, k < rowB_end);
           k += blockDim_x)
#else
      /* threads in the same ygroup work on one row together */
      const HYPRE_Int     rowB = __shfl_sync(HYPRE_WARP_FULL_MASK, colA, 0, blockDim_x);
      const HYPRE_Complex mult = __shfl_sync(HYPRE_WARP_FULL_MASK, valA, 0, blockDim_x);
      /* open this row of B, collectively */
      if (rowB != -1 && threadIdx_x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx_x);
      }
      const HYPRE_Int rowB_start = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 0, blockDim_x);
      const HYPRE_Int rowB_end   = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 1, blockDim_x);

      for (HYPRE_Int k = rowB_start + threadIdx_x; __any_sync(HYPRE_WARP_FULL_MASK, k < rowB_end);
           k += blockDim_x)
#endif
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
#ifdef HYPRE_USING_SYCL
               warp_failed = warp_allreduce_sum(failed, SG);
#else
               warp_failed = warp_allreduce_sum(failed);
#endif
            }
         }
      }
   }

   return num_new_insert;
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
                      HYPRE_Int     *js,
                      HYPRE_Complex *as,
                      HYPRE_Int     *ig,
                      HYPRE_Int     *jg,
                      HYPRE_Complex *ag,
                      HYPRE_Int     *rc,
                      HYPRE_Int     *rf
#ifdef HYPRE_USING_SYCL
                      , sycl::nd_item<3>& item,
                      HYPRE_Int* s_HashKeys,    /* shared */
                      HYPRE_Complex* s_HashVals /* shared */
#endif
                    )
{
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int warp_size = SG.get_local_range().get(0);
   HYPRE_Int blockDim_z = item.get_local_range(0);
   HYPRE_Int blockDim_y = item.get_local_range(1);
   HYPRE_Int blockDim_x = item.get_local_range(2);

   const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * item.get_group_range(2);
   /* warp id inside the block */
   const HYPRE_Int warp_id = item.get_local_id(0);
   /* warp id in the grid */
   const HYPRE_Int grid_warp_id = item.get_group(2) * NUM_WARPS_PER_BLOCK + warp_id;
   /* lane id inside the warp */
   HYPRE_Int lane_id = get_lane_id(item);
   /* shared memory hash table */

   /* shared memory hash table for this warp */
   HYPRE_Int *warp_s_HashKeys = s_HashKeys + warp_id * SHMEM_HASH_SIZE;
   HYPRE_Complex *warp_s_HashVals = s_HashVals + warp_id * SHMEM_HASH_SIZE;
#else
   HYPRE_Int warp_size  = HYPRE_WARP_SIZE;
   HYPRE_Int blockDim_z = blockDim.z;
   HYPRE_Int blockDim_y = blockDim.y;
   HYPRE_Int blockDim_x = blockDim.x;

   volatile const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   /* warp id inside the block */
   volatile const HYPRE_Int warp_id = get_warp_id();
   /* warp id in the grid */
   volatile const HYPRE_Int grid_warp_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
   /* lane id inside the warp */
   volatile HYPRE_Int lane_id = get_lane_id();
   /* shared memory hash table */
#if 1
   __shared__ volatile HYPRE_Int s_HashKeys[NUM_WARPS_PER_BLOCK * SHMEM_HASH_SIZE];
   __shared__ volatile HYPRE_Complex s_HashVals[NUM_WARPS_PER_BLOCK * SHMEM_HASH_SIZE];
#else
   extern __shared__ volatile HYPRE_Int shared_mem[];
   volatile HYPRE_Int *s_HashKeys = shared_mem;
   volatile HYPRE_Complex *s_HashVals = (volatile HYPRE_Complex *) &s_HashKeys[NUM_WARPS_PER_BLOCK *
                                                                                                   SHMEM_HASH_SIZE];
#endif // #if 1

   /* shared memory hash table for this warp */
   volatile HYPRE_Int *warp_s_HashKeys = s_HashKeys + warp_id * SHMEM_HASH_SIZE;
   volatile HYPRE_Complex *warp_s_HashVals = s_HashVals + warp_id * SHMEM_HASH_SIZE;
#endif

   hypre_device_assert(blockDim_z              == NUM_WARPS_PER_BLOCK);
   hypre_device_assert(blockDim_x * blockDim_y == warp_size);
   hypre_device_assert(NUM_WARPS_PER_BLOCK     <= warp_size);

   /* a warp working on the ith row */
   for (HYPRE_Int i = grid_warp_id; i < M; i += num_warps)
   {
      /* start/end position of global memory hash table */
      HYPRE_Int j = 0, ii;
      hypre_int failed = 0;

      if (ATTEMPT == 2)
      {
         if (lane_id == 0)
         {
            j = rind[i];
         }
#ifdef HYPRE_USING_SYCL
         ii = SG.shuffle(j, 0);
#else
         ii = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
#endif
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
            j = read_only_load(ig + ii + lane_id);
         }

#ifdef HYPRE_USING_SYCL
         istart_g = SG.shuffle(j, 0);
         iend_g   = SG.shuffle(j, 1);
#else
         istart_g = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
         iend_g   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);
#endif

         /* size of global hash table allocated for this row
           (must be power of 2 and >= the actual size of the row of C - shmem hash size) */
         ghash_size = iend_g - istart_g;

         /* initialize warp's global memory hash table */
#pragma unroll
         for (HYPRE_Int k = lane_id; k < ghash_size; k += warp_size)
         {
            jg[istart_g + k] = -1;
            ag[istart_g + k] = 0.0;
         }
      }

      /* initialize warp's shared memory hash table */
#pragma unroll
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += warp_size)
      {
         warp_s_HashKeys[k] = -1;
         warp_s_HashVals[k] = 0.0;
      }

#ifdef HYPRE_USING_SYCL
      SG.barrier();
#else
      __syncwarp();
#endif

      /* work with two hash tables */
      j = hypre_spgemm_compute_row_attempt<ATTEMPT, HashType>(ii, lane_id, ia, ja, aa, ib, jb, ab,
                                                              SHMEM_HASH_SIZE, warp_s_HashKeys, warp_s_HashVals,
                                                              ghash_size, jg + istart_g, ag + istart_g, failed
#ifdef HYPRE_USING_SYCL
                                                              , item
#endif
                                                             );

#if defined(HYPRE_DEBUG)
      if (ATTEMPT == 2)
      {
         hypre_device_assert(failed == 0);
      }
#endif

      /* num of inserts in this row (an upper bound) */
#ifdef HYPRE_USING_SYCL
      j = sycl::reduce_over_group(SG, j, std::plus<>());
#else
      j = warp_reduce_sum(j);
#endif

      /* if this row failed */
      if (ATTEMPT == 1)
      {
#ifdef HYPRE_USING_SYCL
         failed = warp_allreduce_sum(failed, SG);
#else
         failed = warp_allreduce_sum(failed);
#endif
      }

      if (lane_id == 0)
      {
         rc[ii] = j;

         if (ATTEMPT == 1)
         {
            rf[ii] = failed > 0;
         }
      }

      if (!failed)
      {
#pragma unroll
         for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += warp_size)
         {
            js[ii * SHMEM_HASH_SIZE + k] = warp_s_HashKeys[k];
            as[ii * SHMEM_HASH_SIZE + k] = warp_s_HashVals[k];
         }
      }
   } // for (i=...)
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
                                        HYPRE_Complex *ac_start
#ifdef HYPRE_USING_SYCL
                                        , sycl::nd_item<3>& item
#endif
                                      )
{
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int warp_size = SG.get_local_range().get(0);
#else
   HYPRE_Int warp_size = HYPRE_WARP_SIZE;
#endif

   HYPRE_Int j = 0;

   /* copy shared memory hash table into C */
#pragma unroll
   for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += warp_size)
   {
      HYPRE_Int key, sum, pos;
      key = s_HashKeys[k];
      HYPRE_Int in = key != -1;
#ifdef HYPRE_USING_SYCL
      pos = warp_prefix_sum(lane_id, in, sum, item);
#else
      pos = warp_prefix_sum(lane_id, in, sum);
#endif
      if (key != -1)
      {
         jc_start[j + pos] = key;
         ac_start[j + pos] = s_HashVals[k];
      }
      j += sum;
   }

   /* copy global memory hash table into C */
#pragma unroll
   for (HYPRE_Int k = 0; k < ghash_size; k += warp_size)
   {
      HYPRE_Int key = -1, sum, pos;
      if (k + lane_id < ghash_size)
      {
         key = jg_start[k + lane_id];
      }
      HYPRE_Int in = key != -1;
#ifdef HYPRE_USING_SYCL
      pos = warp_prefix_sum(lane_id, in, sum, item);
#else
      pos = warp_prefix_sum(lane_id, in, sum);
#endif
      if (key != -1)
      {
         jc_start[j + pos] = key;
         ac_start[j + pos] = ag_start[k + lane_id];
      }
      j += sum;
   }

   return j;
}

template <HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_HASH_SIZE>
__global__ void
hypre_spgemm_copy_from_hash_into_C(
#ifdef HYPRE_USING_SYCL
   sycl::nd_item<3>& item,
#endif
   HYPRE_Int      M,
   HYPRE_Int     *rf,
   HYPRE_Int     *js,
   HYPRE_Complex *as,
   HYPRE_Int     *ig1,
   HYPRE_Int     *jg1,
   HYPRE_Complex *ag1,
   HYPRE_Int     *ig2,
   HYPRE_Int     *jg2,
   HYPRE_Complex *ag2,
   HYPRE_Int     *ic,
   HYPRE_Int     *jc,
   HYPRE_Complex *ac )
{
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int warp_size = SG.get_local_range().get(0);
   HYPRE_Int blockDim_y = item.get_local_range(1);
   HYPRE_Int blockDim_x = item.get_local_range(2);

   const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * item.get_group_range(2);
   /* warp id inside the block */
   const HYPRE_Int warp_id = item.get_local_id(0);
   /* warp id in the grid */
   volatile const HYPRE_Int grid_warp_id = item.get_group(2) * NUM_WARPS_PER_BLOCK + warp_id;
   /* lane id inside the warp */
   volatile const HYPRE_Int lane_id = get_lane_id(item);
#else
   HYPRE_Int warp_size  = HYPRE_WARP_SIZE;
   HYPRE_Int blockDim_y = blockDim.y;
   HYPRE_Int blockDim_x = blockDim.x;

   const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   /* warp id inside the block */
   const HYPRE_Int warp_id = get_warp_id();
   /* warp id in the grid */
   volatile const HYPRE_Int grid_warp_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
   /* lane id inside the warp */
   volatile const HYPRE_Int lane_id = get_lane_id();
#endif

   hypre_device_assert(blockDim_x * blockDim_y == warp_size);

   for (HYPRE_Int i = grid_warp_id; i < M; i += num_warps)
   {
      HYPRE_Int j = 0, irf = 0;
      HYPRE_Int istart_c = 0, istart_g = 0, iend_g = 0, *ig = NULL, *jg = NULL;
      HYPRE_Complex *ag = NULL;

      if (lane_id == 0)
      {
         istart_c = read_only_load(ic + i);
         irf = read_only_load(rf + i);
      }
#ifdef HYPRE_USING_SYCL
      istart_c  = SG.shuffle(istart_c, 0);
      irf = SG.shuffle(irf, 0);
#else
      istart_c  = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 0);
      irf = __shfl_sync(HYPRE_WARP_FULL_MASK, irf, 0);
#endif

      if (irf)
      {
         ig = ig2;  jg = jg2;  ag = ag2;
      }
      else
      {
         ig = ig1;  jg = jg1;  ag = ag1;
      }

      if (ig)
      {
         if (lane_id < 2)
         {
            j = read_only_load(ig + i + lane_id);
         }
#ifdef HYPRE_USING_SYCL
         istart_g = SG.shuffle(j, 0);
         iend_g   = SG.shuffle(j, 1);
#else
         istart_g = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
         iend_g   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);
#endif
      }

#ifdef HYPRE_DEBUG
      j =
#endif
         hypre_spgemm_copy_from_hash_into_C_row<NUM_WARPS_PER_BLOCK, SHMEM_HASH_SIZE>
         (lane_id, js + i * SHMEM_HASH_SIZE, as + i * SHMEM_HASH_SIZE, iend_g - istart_g,
          jg + istart_g, ag + istart_g, jc + istart_c, ac + istart_c
#ifdef HYPRE_USING_SYCL
          , item
#endif
         );

#ifdef HYPRE_DEBUG
      hypre_device_assert(istart_c + j == ic[i + 1]);
#endif
   }
}

/* SpGeMM with Rownnz Estimates */
template <HYPRE_Int shmem_hash_size, char hash_type>
HYPRE_Int
hypre_spgemm_numerical_with_rowest( HYPRE_Int       m,
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
                                    HYPRE_Complex **d_c_out,
                                    HYPRE_Int      *nnzC_out )
{
   const HYPRE_Int num_warps_per_block = 16;
   const HYPRE_Int BDIMX               = 2;
   const HYPRE_Int BDIMY               = HYPRE_WARP_SIZE / BDIMX;
   const size_t    shash_size          = shmem_hash_size * (size_t) m;

#if 0
   const size_t    shmem_size          = num_warps_per_block * shmem_hash_size * (sizeof(
                                                                                     HYPRE_Complex) + sizeof(HYPRE_Int));
   const HYPRE_Int shmem_maxbytes      = 65536;
   hypre_assert(shmem_size <= shmem_maxbytes);
   /* CUDA V100 */
   if (shmem_maxbytes > 49152)
   {
      HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, 1, hash_type>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_maxbytes) );

      HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, 2, hash_type>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_maxbytes) );
   }
#endif

#ifdef HYPRE_USING_SYCL
   /* SYCL kernel configurations */
   sycl::range<3> bDim(num_warps_per_block, BDIMY, BDIMX);
   hypre_assert(bDim[2] * bDim[1] == HYPRE_WARP_SIZE);
#else
   /* CUDA kernel configurations */
   dim3 bDim(BDIMX, BDIMY, num_warps_per_block);
   hypre_assert(bDim.x * bDim.y == HYPRE_WARP_SIZE);
#endif
   /* ---------------------------------------------------------------------------
    * global memory hash table
    * ---------------------------------------------------------------------------*/
   HYPRE_Int     *d_ghash1_i = NULL;
   HYPRE_Int     *d_ghash1_j = NULL;
   HYPRE_Complex *d_ghash1_a = NULL;
   HYPRE_Int     *d_ghash2_i = NULL;
   HYPRE_Int     *d_ghash2_j = NULL;
   HYPRE_Complex *d_ghash2_a = NULL;
   HYPRE_Int     *d_js       = hypre_TAlloc(HYPRE_Int,     shash_size, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *d_as       = hypre_TAlloc(HYPRE_Complex, shash_size, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *rf_ind     = NULL;

   /* ---------------------------------------------------------------------------
    * 1st multiplication attempt:
    * ---------------------------------------------------------------------------*/
   {
      hypre_SpGemmCreateGlobalHashTable(m, NULL, m, d_rc, shmem_hash_size, &d_ghash1_i,
                                        &d_ghash1_j, &d_ghash1_a, NULL, 2);
#ifdef HYPRE_USING_SYCL
      // for cases where one WARP works on a row
      sycl::range<3> gDim( 1, 1, (m + bDim[0] - 1) / bDim[0] );

      hypre_HandleComputeStream(hypre_handle())->submit([&] (sycl::handler & cgh)
      {

         sycl::range<1> shared_range(num_warps_per_block * shmem_hash_size);
         sycl::accessor<HYPRE_Int, 1, sycl::access_mode::read_write,
              sycl::target::local> s_HashKeys_acc(shared_range, cgh);
         sycl::accessor<HYPRE_Complex, 1, sycl::access_mode::read_write,
              sycl::target::local> s_HashVals_acc(shared_range, cgh);

         cgh.parallel_for(
            sycl::nd_range<3>(gDim * bDim,
                              bDim), [ = ] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_WARP_SIZE)]]
         {
            hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, 1, hash_type>(
               m, NULL, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash1_i, d_ghash1_j, d_ghash1_a,
               d_rc, d_rf,
               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer() );
         });
      });
#else
      // for cases where one WARP works on a row
      dim3 gDim( (m + bDim.z - 1) / bDim.z );

      HYPRE_GPU_LAUNCH ( (hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, 1, hash_type>),
                         gDim, bDim, /* shmem_size, */
                         m, NULL, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash1_i, d_ghash1_j, d_ghash1_a,
                         d_rc, d_rf );
#endif
   }

   HYPRE_Int num_failed_rows = hypreDevice_IntegerReduceSum(m, d_rf);

   /* ---------------------------------------------------------------------------
    * 2nd multiplication attempt:
    * ---------------------------------------------------------------------------*/
   if (num_failed_rows)
   {
      //hypre_printf("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__, num_failed_rows, num_failed_rows / (m + 0.0) );

      rf_ind = hypre_TAlloc(HYPRE_Int, num_failed_rows, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_USING_SYCL
      HYPRE_Int *new_end = hypreSycl_copy_if( oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                                              oneapi::dpl::counting_iterator<HYPRE_Int>(m),
                                              d_rf,
                                              rf_ind,
                                              oneapi::dpl::identity() );
#else
      HYPRE_Int *new_end = HYPRE_THRUST_CALL( copy_if,
                                              thrust::make_counting_iterator(0),
                                              thrust::make_counting_iterator(m),
                                              d_rf,
                                              rf_ind,
                                              thrust::identity<HYPRE_Int>() );
#endif

      hypre_assert(new_end - rf_ind == num_failed_rows);

      /* ---------------------------------------------------------------------------
       * build a second hash table for long rows
       * ---------------------------------------------------------------------------*/
      hypre_SpGemmCreateGlobalHashTable(num_failed_rows, rf_ind, m, d_rc, shmem_hash_size,
                                        &d_ghash2_i, &d_ghash2_j, &d_ghash2_a, NULL, 2);
#ifdef HYPRE_USING_SYCL
      // for cases where one WARP works on a row
      sycl::range<3> gDim( 1, 1, (num_failed_rows + bDim[0] - 1) / bDim[0] );

      hypre_HandleComputeStream(hypre_handle())->submit([&] (sycl::handler & cgh)
      {

         sycl::range<1> shared_range(num_warps_per_block * shmem_hash_size);
         sycl::accessor<HYPRE_Int, 1, sycl::access_mode::read_write,
              sycl::target::local> s_HashKeys_acc(shared_range, cgh);
         sycl::accessor<HYPRE_Complex, 1, sycl::access_mode::read_write,
              sycl::target::local> s_HashVals_acc(shared_range, cgh);

         cgh.parallel_for(
            sycl::nd_range<3>(gDim * bDim,
                              bDim), [ = ] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_WARP_SIZE)]]
         {
            hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, 2, hash_type>(
               num_failed_rows, rf_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash2_i, d_ghash2_j,
               d_ghash2_a,
               d_rc, NULL,
               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer() );
         });
      });
#else
      // for cases where one WARP works on a row
      dim3 gDim( (num_failed_rows + bDim.z - 1) / bDim.z );

      HYPRE_GPU_LAUNCH ( (hypre_spgemm_attempt<num_warps_per_block, shmem_hash_size, 2, hash_type>),
                         gDim, bDim, /* shmem_size, */
                         num_failed_rows, rf_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash2_i, d_ghash2_j,
                         d_ghash2_a,
                         d_rc, NULL );
#endif
   }

   HYPRE_Int     *d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_jc;
   HYPRE_Complex *d_c;
   HYPRE_Int      nnzC;

   hypre_create_ija(m, d_rc, d_ic, &d_jc, &d_c, &nnzC);

   /* ---------------------------------------------------------------------------
    * copy from hash tables to C
    * ---------------------------------------------------------------------------*/
   {
      // for cases where one WARP works on a row
#ifdef HYPRE_USING_SYCL
      sycl::range<3> gDim( 1, 1, (m + bDim[0] - 1) / bDim[0] );
#else
      dim3 gDim( (m + bDim.z - 1) / bDim.z );
#endif
      HYPRE_GPU_LAUNCH( (hypre_spgemm_copy_from_hash_into_C<num_warps_per_block, shmem_hash_size>), gDim,
                        bDim,
                        m, d_rf,
                        d_js, d_as,
                        d_ghash1_i, d_ghash1_j, d_ghash1_a,
                        d_ghash2_i, d_ghash2_j, d_ghash2_a,
                        d_ic, d_jc, d_c );
   }

   hypre_TFree(d_ghash1_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash1_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash1_a, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash2_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash2_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash2_a, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_js,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_as,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(rf_ind,     HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

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
                                              HYPRE_Int      *nnzC )
{
   const HYPRE_Int shmem_hash_size = HYPRE_SPGEMM_NUMER_HASH_SIZE;
   const char      hash_type       = hypre_HandleSpgemmHashType(hypre_handle());

   /* a binary array to indicate if row nnz counting is failed for a row */
   HYPRE_Int *d_rf = d_rc + m;

   if (hash_type == 'L')
   {
      hypre_spgemm_numerical_with_rowest<shmem_hash_size, 'L'>
      (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_rf, d_ic_out, d_jc_out, d_c_out, nnzC);
   }
   else if (hash_type == 'Q')
   {
      hypre_spgemm_numerical_with_rowest<shmem_hash_size, 'Q'>
      (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_rf, d_ic_out, d_jc_out, d_c_out, nnzC);
   }
   else if (hash_type == 'D')
   {
      hypre_spgemm_numerical_with_rowest<shmem_hash_size, 'D'>
      (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_rf, d_ic_out, d_jc_out, d_c_out, nnzC);
   }
   else
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   return hypre_error_flag;
}

#endif // HYPRE_USING_GPU
