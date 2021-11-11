/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
           Perform SpMM with Row Nnz Upper Bound
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
#include "seq_mv.h"
#include "csr_spgemm_device.h"

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Numerical Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_numer( char           HashType,
                                HYPRE_Int      HashSize,      /* capacity of the hash table */
                       volatile HYPRE_Int     *HashKeys,      /* assumed to be initialized as all -1's */
                       volatile HYPRE_Complex *HashVals,      /* assumed to be initialized as all 0's */
                                HYPRE_Int      key,           /* assumed to be nonnegative */
                                HYPRE_Complex  val )
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
         j = HashFunc(HashType, HashSize, key, i, j);
      }

      /* try to insert key+1 into slot j */
      HYPRE_Int old = atomicCAS((HYPRE_Int *)(HashKeys+j), -1, key);

      if (old == -1 || old == key)
      {
         /* this slot was open or contained 'key', update value */
         atomicAdd((HYPRE_Complex*)(HashVals+j), val);
         return j;
      }
   }

   return -1;
}

static __device__ __forceinline__
void
hypre_spgemm_compute_row_numer( char           HashType,
                                HYPRE_Int      rowi,
                                HYPRE_Int      warp_lane_id,
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
                                HYPRE_Complex *g_HashVals)
{
   /* load the start and end position of row i of A */
   HYPRE_Int i = 0;

   if (warp_lane_id < 2)
   {
      i = read_only_load(ia + rowi + warp_lane_id);
   }
   const HYPRE_Int istart = __shfl_sync(HYPRE_WARP_FULL_MASK, i, 0);
   const HYPRE_Int iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, i, 1);

   /* load column idx and values of row i of A */
   for (i = istart; i < iend; i += blockDim.y)
   {
      HYPRE_Int     colA = -1;
      HYPRE_Complex valA = 0.0;

      if (threadIdx.x == 0 && i + threadIdx.y < iend)
      {
         colA = read_only_load(ja + i + threadIdx.y);
         valA = read_only_load(aa + i + threadIdx.y);
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(warp_lane_id);
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
            HYPRE_Int pos = hypre_spgemm_hash_insert_numer
               (HashType, s_HashSize, s_HashKeys, s_HashVals, k_idx, k_val);

            if (-1 == pos)
            {
               pos = hypre_spgemm_hash_insert_numer
                     (HashType, g_HashSize, g_HashKeys, g_HashVals, k_idx, k_val);
            }

            hypre_device_assert(pos != -1);
         }
      }
   }
}

template <HYPRE_Int SHMEM_HASH_SIZE>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_copy_from_hash_into_C_row( HYPRE_Int      lane_id,
                               volatile HYPRE_Int     *s_HashKeys,
                               volatile HYPRE_Complex *s_HashVals,
                                        HYPRE_Int      ghash_size,
                                        HYPRE_Int     *jg_start,
                                        HYPRE_Complex *ag_start,
                                        HYPRE_Int     *jc_start,
                                        HYPRE_Complex *ac_start)
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

template <HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE>
__global__ void
hypre_spgemm_numeric( HYPRE_Int      FAILED_SYMBL,
                      char           HashType,
                      HYPRE_Int      M, /* HYPRE_Int K, HYPRE_Int N, */
                      HYPRE_Int     *ia,
                      HYPRE_Int     *ja,
                      HYPRE_Complex *aa,
                      HYPRE_Int     *ib,
                      HYPRE_Int     *jb,
                      HYPRE_Complex *ab,
                      HYPRE_Int     *ic,
                      HYPRE_Int     *jc,
                      HYPRE_Complex *ac,
                      HYPRE_Int     *rc,
                      HYPRE_Int     *ig,
                      HYPRE_Int     *jg,
                      HYPRE_Complex *ag)
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
   __shared__ volatile HYPRE_Int     s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
   __shared__ volatile HYPRE_Complex s_HashVals[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
   /* shared memory hash table for this group */
   volatile HYPRE_Int     *group_s_HashKeys = s_HashKeys + group_id * SHMEM_HASH_SIZE;
   volatile HYPRE_Complex *group_s_HashVals = s_HashVals + group_id * SHMEM_HASH_SIZE;

   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (HYPRE_Int i = grid_group_id; i < M; i += grid_num_groups)
   {
      /* start/end position of global memory hash table */
      HYPRE_Int istart_g = 0, iend_g = 0, ghash_size = 0, jsum = 0;

      if (ig)
      {
         if (warp_lane_id < 2)
         {
            istart_g = read_only_load(ig + grid_group_id + warp_lane_id);
         }
         iend_g   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_g, 1);
         istart_g = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_g, 0);

         /* size of global hash table allocated for this row
            (must be power of 2 and >= the actual size of the row of C) */
         ghash_size = iend_g - istart_g;

         /* initialize group's global memory hash table */
#pragma unroll
         for (HYPRE_Int k = lane_id; k < ghash_size; k += GROUP_SIZE)
         {
            jg[istart_g + k] = -1;
            ag[istart_g + k] = 0.0;
         }
      }

      /* initialize group's shared memory hash table */
#pragma unroll
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
      {
         group_s_HashKeys[k] = -1;
         group_s_HashVals[k] = 0.0;
      }

      group_sync<GROUP_SIZE>();

      /* work with two hash tables */
      hypre_spgemm_compute_row_numer(HashType, i, warp_lane_id, ia, ja, aa, ib, jb, ab,
                                     SHMEM_HASH_SIZE, group_s_HashKeys,
                                     group_s_HashVals,
                                     ghash_size, jg + istart_g, ag + istart_g);

      if (GROUP_SIZE != HYPRE_WARP_SIZE)
      {
         group_sync<GROUP_SIZE>();
      }

      /* copy results into the final C */
      if (get_warp_in_group_id<GROUP_SIZE>() == 0)
      {
         HYPRE_Int istart_c;
#ifdef HYPRE_DEBUG
         HYPRE_Int iend_c;
         if (warp_lane_id < 2)
         {
            istart_c = read_only_load(ic + i + warp_lane_id);
         }
         iend_c   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 1);
         istart_c = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 0);
#else
         if (warp_lane_id < 1)
         {
            istart_c = read_only_load(ic + i);
         }
         istart_c = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 0);
#endif

         jsum = hypre_spgemm_copy_from_hash_into_C_row<SHMEM_HASH_SIZE>
            (warp_lane_id, group_s_HashKeys, group_s_HashVals, ghash_size, jg + istart_g,
             ag + istart_g, jc + istart_c, ac + istart_c);

#if defined(HYPRE_DEBUG)
         if (FAILED_SYMBL)
         {
            hypre_device_assert(istart_c + jsum <= iend_c);
         }
         else
         {
            hypre_device_assert(istart_c + jsum == iend_c);
         }
#endif
      }

      if (GROUP_SIZE != HYPRE_WARP_SIZE)
      {
         group_sync<GROUP_SIZE>();
      }

      if (FAILED_SYMBL)
      {
         /* when symb mult was failed, save (exact) row nnz into rc */
         if (lane_id == 0)
         {
            rc[i] = jsum;
         }
      }
   }
}

template <HYPRE_Int GROUP_SIZE>
__global__ void
hypre_spgemm_copy_from_Cext_into_C( HYPRE_Int      M,
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
   const HYPRE_Int lane_id = get_lane_id();
   /* lane id inside the warp */
   const HYPRE_Int warp_lane_id = get_warp_lane_id();

   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);

   for (HYPRE_Int i = grid_group_id; i < M; i += grid_num_groups)
   {
      HYPRE_Int istart_c, iend_c, istart_x;
#if defined(HYPRE_DEBUG)
      HYPRE_Int iend_x;
#endif
      /* start/end position in C and X*/
      if (warp_lane_id < 2)
      {
         istart_c = read_only_load(ic + i + warp_lane_id);
         istart_x = read_only_load(ix + i + warp_lane_id);
      }
      iend_c   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 1);
      istart_c = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_c, 0);
#if defined(HYPRE_DEBUG)
      iend_x   = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_x, 1);
#endif
      istart_x = __shfl_sync(HYPRE_WARP_FULL_MASK, istart_x, 0);
#if defined(HYPRE_DEBUG)
      hypre_device_assert(iend_c - istart_c <= iend_x - istart_x);
#endif

      const HYPRE_Int p = istart_x - istart_c;
      for (HYPRE_Int k = istart_c + lane_id; k < iend_c; k += GROUP_SIZE)
      {
         jc[k] = jx[k + p];
         ac[k] = ax[k + p];
      }
   }
}

/* SpGeMM with Rownnz/Upper bound */
template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int
hypre_spgemm_numerical_with_rownnz( HYPRE_Int       m,
                                    HYPRE_Int       k,
                                    HYPRE_Int       n,
                                    HYPRE_Int       EXACT_ROWNNZ,
                                    char            hash_type,
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
                                    HYPRE_Int      *nnzC)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] -= hypre_MPI_Wtime();
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

   /* ---------------------------------------------------------------------------
    * build hash table
    * ---------------------------------------------------------------------------*/
   HYPRE_Int     *d_ghash_i = NULL;
   HYPRE_Int     *d_ghash_j = NULL;
   HYPRE_Complex *d_ghash_a = NULL;
   HYPRE_Int      ghash_size = 0;

   /* RL Note: even with exact rownnz, still may need global hash, since shared hash has different size from symbol. */
   hypre_SpGemmCreateGlobalHashTable(m, NULL, num_act_groups, d_rc, SHMEM_HASH_SIZE,
                                     &d_ghash_i, &d_ghash_j, &d_ghash_a, &ghash_size);

   printf("%s[%d]: SHMEM_HASH_SIZE %d, ghash %p\n", __func__, __LINE__, SHMEM_HASH_SIZE, d_ghash_i);
   hypre_printf("%s[%d]: ghash size %d\n", __func__, __LINE__, ghash_size);

   /* ---------------------------------------------------------------------------
    * numerical multiplication:
    * ---------------------------------------------------------------------------*/
   /* if rc contains exact rownnz: can allocate the final C=(ic,jc,c) directly;
      if rc contains upper bound : it is a temporary space that is more than enough to store C */
   HYPRE_Int     *d_ic = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_jc;
   HYPRE_Complex *d_c;
   HYPRE_Int      nnzC_nume;

   hypre_create_ija(1, SHMEM_HASH_SIZE, m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC_nume);

   printf("%s[%d] nnzC_nume %d\n", __func__, __LINE__, nnzC_nume);

   HYPRE_CUDA_LAUNCH ( (hypre_spgemm_numeric<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE>),
                        gDim, bDim, /* shmem_size, */
                        !EXACT_ROWNNZ, hash_type,
                        m, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_rc,
                        d_ghash_i, d_ghash_j, d_ghash_a );

   /* post-processing */
   if (!EXACT_ROWNNZ)
   {
      HYPRE_Int nnzC_nume_new = hypreDevice_IntegerReduceSum(m, d_rc);

      hypre_assert(nnzC_nume_new <= nnzC_nume);

      if (nnzC_nume_new < nnzC_nume)
      {
         HYPRE_Int     *d_ic_new = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
         HYPRE_Int     *d_jc_new;
         HYPRE_Complex *d_c_new;
         HYPRE_Int      tmp;

         /* alloc final C */
         hypre_create_ija(1, SHMEM_HASH_SIZE, m, NULL, d_rc, d_ic_new, &d_jc_new, &d_c_new, &tmp);
         hypre_assert(tmp == nnzC_nume_new);

         printf("%s[%d]: nnzC_nume %d\n", __func__, __LINE__, nnzC_nume_new);

         /* copy to the final C */
         dim3 gDim( (m + bDim.z - 1) / bDim.z );
         HYPRE_CUDA_LAUNCH( (hypre_spgemm_copy_from_Cext_into_C<GROUP_SIZE>), gDim, bDim,
                            m, d_ic, d_jc, d_c, d_ic_new, d_jc_new, d_c_new );

         hypre_TFree(d_ic, HYPRE_MEMORY_DEVICE);
         hypre_TFree(d_jc, HYPRE_MEMORY_DEVICE);
         hypre_TFree(d_c,  HYPRE_MEMORY_DEVICE);

         d_ic = d_ic_new;
         d_jc = d_jc_new;
         d_c  = d_c_new;
         nnzC_nume = nnzC_nume_new;
      }
   }

   hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_a, HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC     = nnzC_nume;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
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
                                                HYPRE_Int      *d_rc,         /* input: nnz (upper bound) of each row */
                                                HYPRE_Int       exact_rownnz, /* if d_rc is exact       */
                                                HYPRE_Int     **d_ic_out,
                                                HYPRE_Int     **d_jc_out,
                                                HYPRE_Complex **d_c_out,
                                                HYPRE_Int      *nnzC )
{
#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPushRange("CSRSpGemmNumerB");
#endif

   const char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
   {
      hypre_printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
      exit(0);
   }

   //HYPRE_Int max_rc = HYPRE_THRUST_CALL(reduce, d_rc, d_rc + m, 0, thrust::maximum<HYPRE_Int>());
   //hypre_printf("max_rc numerical %d\n", max_rc);

   hypre_spgemm_numerical_with_rownnz<SHMEM_HASH_SIZE, GROUP_SIZE>
      (m, k, n, exact_rownnz, hash_type, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_ic_out, d_jc_out, d_c_out, nnzC);

#ifdef HYPRE_SPGEMM_NVTX
   hypre_GpuProfilingPopRange();
#endif

   return hypre_error_flag;
}

template HYPRE_Int hypreDevice_CSRSpGemmNumerWithRownnzUpperbound<HYPRE_SPGEMM_NUMER_HASH_SIZE, HYPRE_WARP_SIZE>(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int exact_rownnz, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

#endif /* HYPRE_USING_CUDA  || defined(HYPRE_USING_HIP) */

