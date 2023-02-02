/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
           Perform SpGEMM with Row Nnz Upper Bound
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
#include "seq_mv.h"
#include "csr_spgemm_device.h"

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Numerical Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/* HashKeys: assumed to be initialized as all -1's
 * HashVals: assumed to be initialized as all 0's
 * Key:      assumed to be nonnegative
 */
template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, HYPRE_Int UNROLL_FACTOR>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_numer( volatile HYPRE_Int     *HashKeys,
                                volatile HYPRE_Complex *HashVals,
                                HYPRE_Int               key,
                                HYPRE_Complex           val )
{
   HYPRE_Int j = 0;

#pragma unroll UNROLL_FACTOR
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
      HYPRE_Int old = atomicCAS((HYPRE_Int *)(HashKeys + j), -1, key);

      if (old == -1 || old == key)
      {
         /* this slot was open or contained 'key', update value */
         atomicAdd((HYPRE_Complex*)(HashVals + j), val);
         return j;
      }
   }

   return -1;
}

template <char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_numer( HYPRE_Int               HashSize,
                                volatile HYPRE_Int     *HashKeys,
                                volatile HYPRE_Complex *HashVals,
                                HYPRE_Int               key,
                                HYPRE_Complex           val )
{
   HYPRE_Int j = 0;

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
      HYPRE_Int old = atomicCAS((HYPRE_Int *)(HashKeys + j), -1, key);

      if (old == -1 || old == key)
      {
         /* this slot was open or contained 'key', update value */
         atomicAdd((HYPRE_Complex*)(HashVals + j), val);
         return j;
      }
   }

   return -1;
}

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, HYPRE_Int GROUP_SIZE, bool HAS_GHASH, bool IA1, HYPRE_Int UNROLL_FACTOR>
static __device__ __forceinline__
void
hypre_spgemm_compute_row_numer( HYPRE_Int               istart_a,
                                HYPRE_Int               iend_a,
                                HYPRE_Int               istart_c,
#if defined(HYPRE_DEBUG)
                                HYPRE_Int               iend_c,
#endif
                                const    HYPRE_Int     *ja,
                                const    HYPRE_Complex *aa,
                                const    HYPRE_Int     *ib,
                                const    HYPRE_Int     *jb,
                                const    HYPRE_Complex *ab,
                                HYPRE_Int              *jc,
                                HYPRE_Complex          *ac,
                                volatile HYPRE_Int     *s_HashKeys,
                                volatile HYPRE_Complex *s_HashVals,
                                HYPRE_Int               g_HashSize,
                                HYPRE_Int              *g_HashKeys,
                                HYPRE_Complex          *g_HashVals )
{
   /* load column idx and values of row of A */
   for (HYPRE_Int i = istart_a + threadIdx.y; __any_sync(HYPRE_WARP_FULL_MASK, i < iend_a);
        i += blockDim.y)
   {
      HYPRE_Int     rowB = -1;
      HYPRE_Complex mult = 0.0;

      if (threadIdx.x == 0 && i < iend_a)
      {
         rowB = read_only_load(ja + i);
         mult = aa ? read_only_load(aa + i) : 1.0;
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(...);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      rowB = __shfl_sync(HYPRE_WARP_FULL_MASK, rowB, 0, blockDim.x);
      mult = __shfl_sync(HYPRE_WARP_FULL_MASK, mult, 0, blockDim.x);
      /* open this row of B, collectively */
      HYPRE_Int tmp = 0;
      if (rowB != -1 && threadIdx.x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx.x);
      }
      const HYPRE_Int rowB_start = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 0, blockDim.x);
      const HYPRE_Int rowB_end   = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 1, blockDim.x);

#if defined(HYPRE_DEBUG)
      if (IA1) { hypre_device_assert(rowB == -1 || rowB_end - rowB_start == iend_c - istart_c); }
#endif

      for (HYPRE_Int k = rowB_start + threadIdx.x; __any_sync(HYPRE_WARP_FULL_MASK, k < rowB_end);
           k += blockDim.x)
      {
         if (k < rowB_end)
         {
            const HYPRE_Int     k_idx = read_only_load(jb + k);
            const HYPRE_Complex k_val = (ab ? read_only_load(ab + k) : 1.0) * mult;

            if (IA1)
            {
               jc[istart_c + k - rowB_start] = k_idx;
               ac[istart_c + k - rowB_start] = k_val;
            }
            else
            {
               /* first try to insert into shared memory hash table */
               HYPRE_Int pos = hypre_spgemm_hash_insert_numer<SHMEM_HASH_SIZE, HASHTYPE, UNROLL_FACTOR>
                               (s_HashKeys, s_HashVals, k_idx, k_val);

               if (HAS_GHASH && -1 == pos)
               {
                  pos = hypre_spgemm_hash_insert_numer<HASHTYPE>
                        (g_HashSize, g_HashKeys, g_HashVals, k_idx, k_val);
               }

               hypre_device_assert(pos != -1);
            }
         }
      }
   }
}

template <HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE, bool HAS_GHASH, HYPRE_Int UNROLL_FACTOR>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_copy_from_hash_into_C_row( HYPRE_Int               lane_id,
                                        volatile HYPRE_Int     *s_HashKeys,
                                        volatile HYPRE_Complex *s_HashVals,
                                        HYPRE_Int               ghash_size,
                                        HYPRE_Int              *jg_start,
                                        HYPRE_Complex          *ag_start,
                                        HYPRE_Int              *jc_start,
                                        HYPRE_Complex          *ac_start )
{
   HYPRE_Int j = 0;
   const HYPRE_Int STEP_SIZE = hypre_min(GROUP_SIZE, HYPRE_WARP_SIZE);

   /* copy shared memory hash table into C */
   //#pragma unroll UNROLL_FACTOR

   if (__any_sync(HYPRE_WARP_FULL_MASK, s_HashKeys != NULL))
   {
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += STEP_SIZE)
      {
         HYPRE_Int sum;
         HYPRE_Int key = s_HashKeys ? s_HashKeys[k] : -1;
         HYPRE_Int pos = group_prefix_sum<HYPRE_Int, STEP_SIZE>(lane_id, (HYPRE_Int) (key != -1), sum);

         if (key != -1)
         {
            jc_start[j + pos] = key;
            ac_start[j + pos] = s_HashVals[k];
         }
         j += sum;
      }
   }

   if (HAS_GHASH)
   {
      /* copy global memory hash table into C */
      for (HYPRE_Int k = lane_id; __any_sync(HYPRE_WARP_FULL_MASK, k < ghash_size); k += STEP_SIZE)
      {
         HYPRE_Int sum;
         HYPRE_Int key = k < ghash_size ? jg_start[k] : -1;
         HYPRE_Int pos = group_prefix_sum<HYPRE_Int, STEP_SIZE>(lane_id, (HYPRE_Int) (key != -1), sum);

         if (key != -1)
         {
            jc_start[j + pos] = key;
            ac_start[j + pos] = ag_start[k];
         }
         j += sum;
      }
   }

   return j;
}

template <HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE, bool HAS_RIND,
          bool FAILED_SYMBL, char HASHTYPE, bool HAS_GHASH>
__global__ void
hypre_spgemm_numeric( hypre_DeviceItem                       &item,
                      const HYPRE_Int                   M,
                      const HYPRE_Int*     __restrict__ rind,
                      const HYPRE_Int*     __restrict__ ia,
                      const HYPRE_Int*     __restrict__ ja,
                      const HYPRE_Complex* __restrict__ aa,
                      const HYPRE_Int*     __restrict__ ib,
                      const HYPRE_Int*     __restrict__ jb,
                      const HYPRE_Complex* __restrict__ ab,
                      const HYPRE_Int*     __restrict__ ic,
                      HYPRE_Int*           __restrict__ jc,
                      HYPRE_Complex*       __restrict__ ac,
                      HYPRE_Int*           __restrict__ rc,
                      const HYPRE_Int*     __restrict__ ig,
                      HYPRE_Int*           __restrict__ jg,
                      HYPRE_Complex*       __restrict__ ag )
{
   /* number of groups in the grid */
   volatile const HYPRE_Int grid_num_groups = get_num_groups() * gridDim.x;
   /* group id inside the block */
   volatile const HYPRE_Int group_id = get_group_id();
   /* group id in the grid */
   volatile const HYPRE_Int grid_group_id = blockIdx.x * get_num_groups() + group_id;
   /* lane id inside the group */
   volatile const HYPRE_Int lane_id = get_group_lane_id(item);
   /* shared memory hash table */
#if defined(HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   extern __shared__ volatile HYPRE_Int shared_mem[];
   volatile HYPRE_Int     *s_HashKeys = shared_mem;
   volatile HYPRE_Complex *s_HashVals = (volatile HYPRE_Complex *)
                                        &s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
#else
   __shared__ volatile HYPRE_Int     s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
   __shared__ volatile HYPRE_Complex s_HashVals[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
#endif
   /* shared memory hash table for this group */
   volatile HYPRE_Int     *group_s_HashKeys = s_HashKeys + group_id * SHMEM_HASH_SIZE;
   volatile HYPRE_Complex *group_s_HashVals = s_HashVals + group_id * SHMEM_HASH_SIZE;

   const HYPRE_Int UNROLL_FACTOR = hypre_min(HYPRE_SPGEMM_NUMER_UNROLL, SHMEM_HASH_SIZE);

   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (HYPRE_Int i = grid_group_id; __any_sync(HYPRE_WARP_FULL_MASK, i < M); i += grid_num_groups)
   {
      HYPRE_Int ii = -1;

      if (HAS_RIND)
      {
         group_read<GROUP_SIZE>(item, rind + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M, ii);
      }
      else
      {
         ii = i;
      }

      /* start/end position of global memory hash table */
      HYPRE_Int istart_g = 0, iend_g = 0, ghash_size = 0;

      if (HAS_GHASH)
      {
         group_read<GROUP_SIZE>(item, ig + grid_group_id, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M,
                                istart_g, iend_g);

         /* size of global hash table allocated for this row
            (must be power of 2 and >= the actual size of the row of C) */
         ghash_size = iend_g - istart_g;

         /* initialize group's global memory hash table */
         for (HYPRE_Int k = lane_id; k < ghash_size; k += GROUP_SIZE)
         {
            jg[istart_g + k] = -1;
            ag[istart_g + k] = 0.0;
         }
      }

      /* initialize group's shared memory hash table */
      if (GROUP_SIZE >= HYPRE_WARP_SIZE || i < M)
      {
#pragma unroll UNROLL_FACTOR
         for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
         {
            group_s_HashKeys[k] = -1;
            group_s_HashVals[k] = 0.0;
         }
      }

      group_sync<GROUP_SIZE>();

      /* start/end position of row of A */
      HYPRE_Int istart_a = 0, iend_a = 0;

      /* load the start and end position of row ii of A */
      group_read<GROUP_SIZE>(item, ia + ii, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M, istart_a, iend_a);

      /* start/end position of row of C */
      HYPRE_Int istart_c = 0;
#if defined(HYPRE_DEBUG)
      HYPRE_Int iend_c = 0;
      group_read<GROUP_SIZE>(item, ic + ii, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M, istart_c, iend_c);
#else
      group_read<GROUP_SIZE>(item, ic + ii, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M, istart_c);
#endif

      /* work with two hash tables */
      if (iend_a == istart_a + 1)
      {
         hypre_spgemm_compute_row_numer<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, true, UNROLL_FACTOR>
         (istart_a, iend_a, istart_c,
#if defined(HYPRE_DEBUG)
          iend_c,
#endif
          ja, aa, ib, jb, ab, jc, ac, group_s_HashKeys, group_s_HashVals, ghash_size,
          jg + istart_g, ag + istart_g);
      }
      else
      {
         hypre_spgemm_compute_row_numer<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, false, UNROLL_FACTOR>
         (istart_a, iend_a, istart_c,
#if defined(HYPRE_DEBUG)
          iend_c,
#endif
          ja, aa, ib, jb, ab, jc, ac, group_s_HashKeys, group_s_HashVals, ghash_size,
          jg + istart_g, ag + istart_g);
      }

      if (GROUP_SIZE > HYPRE_WARP_SIZE)
      {
         group_sync<GROUP_SIZE>();
      }

      HYPRE_Int jsum = 0;

      /* the first warp of the group copies results into the final C
       * if GROUP_SIZE < WARP_SIZE, the whole group copies */
      if (get_warp_in_group_id<GROUP_SIZE>(item) == 0)
      {
         jsum = hypre_spgemm_copy_from_hash_into_C_row<GROUP_SIZE, SHMEM_HASH_SIZE, HAS_GHASH, UNROLL_FACTOR>
                (lane_id,
                 (iend_a > istart_a + 1) && (GROUP_SIZE >= HYPRE_WARP_SIZE || i < M) ? group_s_HashKeys : NULL,
                 group_s_HashVals,
                 ghash_size,
                 jg + istart_g, ag + istart_g,
                 jc + istart_c, ac + istart_c);

#if defined(HYPRE_DEBUG)
         if (iend_a != istart_a + 1)
         {
            if (FAILED_SYMBL)
            {
               hypre_device_assert(istart_c + jsum <= iend_c);
            }
            else
            {
               hypre_device_assert(istart_c + jsum == iend_c);
            }
         }
#endif
      }

      if (GROUP_SIZE > HYPRE_WARP_SIZE)
      {
         group_sync<GROUP_SIZE>();
      }

      if (FAILED_SYMBL)
      {
         /* when symb mult was failed, save (exact) row nnz */
         if ((GROUP_SIZE >= HYPRE_WARP_SIZE || i < M) && lane_id == 0)
         {
            rc[ii] = jsum;
         }
      }
   }
}

/* SpGeMM with Rownnz/Upper bound */
template <HYPRE_Int BIN, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE, bool HAS_RIND>
HYPRE_Int
hypre_spgemm_numerical_with_rownnz( HYPRE_Int      m,
                                    HYPRE_Int     *row_ind,
                                    HYPRE_Int      k,
                                    HYPRE_Int      n,
                                    bool           need_ghash,
                                    HYPRE_Int      exact_rownnz,
                                    HYPRE_Int     *d_ia,
                                    HYPRE_Int     *d_ja,
                                    HYPRE_Complex *d_a,
                                    HYPRE_Int     *d_ib,
                                    HYPRE_Int     *d_jb,
                                    HYPRE_Complex *d_b,
                                    HYPRE_Int     *d_rc,
                                    HYPRE_Int     *d_ic,
                                    HYPRE_Int     *d_jc,
                                    HYPRE_Complex *d_c )
{
   const HYPRE_Int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
#if defined(HYPRE_USING_CUDA)
   const HYPRE_Int BDIMX                = hypre_min(4, GROUP_SIZE);
#elif defined(HYPRE_USING_HIP)
   const HYPRE_Int BDIMX                = hypre_min(4, GROUP_SIZE);
#endif
   const HYPRE_Int BDIMY                = GROUP_SIZE / BDIMX;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
   hypre_assert(bDim.x * bDim.y == GROUP_SIZE);
   // grid dimension (number of blocks)
   const HYPRE_Int num_blocks = hypre_min( hypre_HandleSpgemmBlockNumDim(hypre_handle())[1][BIN],
                                           (HYPRE_Int) ((m + bDim.z - 1) / bDim.z) );
   dim3 gDim( num_blocks );
   // number of active groups
   HYPRE_Int num_act_groups = hypre_min((HYPRE_Int) (bDim.z * gDim.x), m);

   const char HASH_TYPE = HYPRE_SPGEMM_HASH_TYPE;

   if (HASH_TYPE != 'L' && HASH_TYPE != 'Q' && HASH_TYPE != 'D')
   {
      hypre_error_w_msg(1, "Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
   }

   /* ---------------------------------------------------------------------------
    * build hash table
    * ---------------------------------------------------------------------------*/
   HYPRE_Int     *d_ghash_i = NULL;
   HYPRE_Int     *d_ghash_j = NULL;
   HYPRE_Complex *d_ghash_a = NULL;
   HYPRE_Int      ghash_size = 0;

   if (need_ghash)
   {
      hypre_SpGemmCreateGlobalHashTable(m, row_ind, num_act_groups, d_rc, SHMEM_HASH_SIZE,
                                        &d_ghash_i, &d_ghash_j, &d_ghash_a, &ghash_size);
   }

#ifdef HYPRE_SPGEMM_PRINTF
   HYPRE_SPGEMM_PRINT("%s[%d], BIN[%d]: m %d k %d n %d, HASH %c, SHMEM_HASH_SIZE %d, GROUP_SIZE %d, "
                      "exact_rownnz %d, need_ghash %d, ghash %p size %d\n",
                      __FILE__, __LINE__, BIN, m, k, n,
                      HASH_TYPE, SHMEM_HASH_SIZE, GROUP_SIZE, exact_rownnz, need_ghash, d_ghash_i, ghash_size);
   HYPRE_SPGEMM_PRINT("kernel spec [%d %d %d] x [%d %d %d]\n", gDim.x, gDim.y, gDim.z, bDim.x, bDim.y,
                      bDim.z);
#endif

#if defined(HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   const size_t shmem_bytes = num_groups_per_block * SHMEM_HASH_SIZE * (sizeof(HYPRE_Int) + sizeof(
                                                                           HYPRE_Complex));
#else
   const size_t shmem_bytes = 0;
#endif

   /* ---------------------------------------------------------------------------
    * numerical multiplication:
    * ---------------------------------------------------------------------------*/
   hypre_assert(HAS_RIND == (row_ind != NULL) );

   /* <NUM_GROUPS_PER_BLOCK, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, FAILED_SYMBL, HASHTYPE, HAS_GHASH> */

   if (exact_rownnz)
   {
      if (ghash_size)
      {
         HYPRE_GPU_LAUNCH2 (
            (hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, false, HASH_TYPE, true>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, NULL, d_ghash_i, d_ghash_j,
            d_ghash_a );
      }
      else
      {
         HYPRE_GPU_LAUNCH2 (
            (hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, false, HASH_TYPE, false>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, NULL, d_ghash_i, d_ghash_j,
            d_ghash_a );
      }
   }
   else
   {
      if (ghash_size)
      {
         HYPRE_GPU_LAUNCH2 (
            (hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, true, HASH_TYPE, true>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_rc, d_ghash_i, d_ghash_j,
            d_ghash_a );
      }
      else
      {
         HYPRE_GPU_LAUNCH2 (
            (hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, true, HASH_TYPE, false>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_rc, d_ghash_i, d_ghash_j,
            d_ghash_a );
      }
   }

   hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_a, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

template <HYPRE_Int GROUP_SIZE>
__global__ void
hypre_spgemm_copy_from_Cext_into_C( hypre_DeviceItem    &item,
                                    HYPRE_Int      M,
                                    HYPRE_Int     *ix,
                                    HYPRE_Int     *jx,
                                    HYPRE_Complex *ax,
                                    HYPRE_Int     *ic,
                                    HYPRE_Int     *jc,
                                    HYPRE_Complex *ac )
{
   /* number of groups in the grid */
   const HYPRE_Int grid_num_groups = get_num_groups() * gridDim.x;
   /* group id inside the block */
   const HYPRE_Int group_id = get_group_id();
   /* group id in the grid */
   const HYPRE_Int grid_group_id = blockIdx.x * get_num_groups() + group_id;
   /* lane id inside the group */
   const HYPRE_Int lane_id = get_group_lane_id(item);

   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (HYPRE_Int i = grid_group_id; __any_sync(HYPRE_WARP_FULL_MASK, i < M); i += grid_num_groups)
   {
      HYPRE_Int istart_c = 0, iend_c = 0, istart_x = 0;

      /* start/end position in C and X */
      group_read<GROUP_SIZE>(item, ic + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M, istart_c, iend_c);
#if defined(HYPRE_DEBUG)
      HYPRE_Int iend_x = 0;
      group_read<GROUP_SIZE>(item, ix + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M, istart_x, iend_x);
      hypre_device_assert(iend_c - istart_c <= iend_x - istart_x);
#else
      group_read<GROUP_SIZE>(item, ix + i, GROUP_SIZE >= HYPRE_WARP_SIZE || i < M, istart_x);
#endif
      const HYPRE_Int p = istart_x - istart_c;
      for (HYPRE_Int k = istart_c + lane_id; k < iend_c; k += GROUP_SIZE)
      {
         jc[k] = jx[k + p];
         ac[k] = ax[k + p];
      }
   }
}

template <HYPRE_Int GROUP_SIZE>
HYPRE_Int
hypreDevice_CSRSpGemmNumerPostCopy( HYPRE_Int       m,
                                    HYPRE_Int      *d_rc,
                                    HYPRE_Int      *nnzC,
                                    HYPRE_Int     **d_ic,
                                    HYPRE_Int     **d_jc,
                                    HYPRE_Complex **d_c)
{
   HYPRE_Int nnzC_new = hypreDevice_IntegerReduceSum(m, d_rc);

   hypre_assert(nnzC_new <= *nnzC && nnzC_new >= 0);

   if (nnzC_new < *nnzC)
   {
      HYPRE_Int     *d_ic_new = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *d_jc_new;
      HYPRE_Complex *d_c_new;

      /* alloc final C */
      hypre_create_ija(m, NULL, d_rc, d_ic_new, &d_jc_new, &d_c_new, &nnzC_new);

#ifdef HYPRE_SPGEMM_PRINTF
      HYPRE_SPGEMM_PRINT("%s[%d]: Post Copy: new nnzC %d\n", __FILE__, __LINE__, nnzC_new);
#endif

      /* copy to the final C */
      const HYPRE_Int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
      dim3 bDim(GROUP_SIZE, 1, num_groups_per_block);
      dim3 gDim( (m + bDim.z - 1) / bDim.z );

      HYPRE_GPU_LAUNCH( (hypre_spgemm_copy_from_Cext_into_C<GROUP_SIZE>), gDim, bDim,
                        m, *d_ic, *d_jc, *d_c, d_ic_new, d_jc_new, d_c_new );

      hypre_TFree(*d_ic, HYPRE_MEMORY_DEVICE);
      hypre_TFree(*d_jc, HYPRE_MEMORY_DEVICE);
      hypre_TFree(*d_c,  HYPRE_MEMORY_DEVICE);

      *d_ic = d_ic_new;
      *d_jc = d_jc_new;
      *d_c  = d_c_new;
      *nnzC = nnzC_new;
   }

   return hypre_error_flag;
}

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int hypre_spgemm_numerical_max_num_blocks( HYPRE_Int  multiProcessorCount,
                                                 HYPRE_Int *num_blocks_ptr,
                                                 HYPRE_Int *block_size_ptr )
{
   const char HASH_TYPE = HYPRE_SPGEMM_HASH_TYPE;
   const HYPRE_Int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
   const HYPRE_Int block_size = num_groups_per_block * GROUP_SIZE;
   hypre_int numBlocksPerSm = 0;
#if defined(HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   const hypre_int shmem_bytes = num_groups_per_block * SHMEM_HASH_SIZE *
                                 (sizeof(HYPRE_Int) + sizeof(HYPRE_Complex));
   hypre_int dynamic_shmem_size = shmem_bytes;
#else
   hypre_int dynamic_shmem_size = 0;
#endif

#if defined(HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
#if defined(HYPRE_USING_CUDA)
   /* with CUDA, to use > 48K shared memory, must use dynamic and must opt-in. BIN = 10 requires 96K */
   const hypre_int max_shmem_optin = hypre_HandleDeviceMaxShmemPerBlock(hypre_handle())[1];
   if (dynamic_shmem_size <= max_shmem_optin)
   {
      HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, false>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, true,  HASH_TYPE, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      HYPRE_CUDA_CALL( cudaFuncSetAttribute(
                          hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, true, HASH_TYPE, false>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );

      /*
         HYPRE_CUDA_CALL( cudaFuncSetAttribute(
         hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, false, false, HASH_TYPE, false>,
         cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_size) );
      */
   }
#endif
#endif

#if defined(HYPRE_USING_CUDA)
   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
      block_size, dynamic_shmem_size);
#endif

#if defined(HYPRE_USING_HIP)
   hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
      block_size, dynamic_shmem_size);
#endif

   *num_blocks_ptr = multiProcessorCount * numBlocksPerSm;
   *block_size_ptr = block_size;

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */
