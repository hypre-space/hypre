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

#if defined(HYPRE_USING_SYCL)

template <typename T>
using relaxed_atomic_ref =
  sycl::ONEAPI::atomic_ref< T, sycl::ONEAPI::memory_order::relaxed,
			    sycl::ONEAPI::memory_scope::device,
			    sycl::access::address_space::local_space>;

template <char HashType, HYPRE_Int attempt>
static __inline__ __attribute__((always_inline))
HYPRE_Int
hash_insert_attempt(
    HYPRE_Int HashSize,           /* capacity of the hash table */
    volatile HYPRE_Int *HashKeys, /* assumed to be initialized as all -1's */
    volatile HYPRE_Complex *HashVals, /* assumed to be initialized as all 0's */
    HYPRE_Int key,                    /* assumed to be nonnegative */
    HYPRE_Complex val, HYPRE_Int &count, /* increase by 1 if is a new entry */
    char failed, volatile char *warp_failed)
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
      sycl::atomic<HYPRE_Int, sycl::access::address_space::local_space>
        obj(sycl::multi_ptr<HYPRE_Int, sycl::access::address_space::local_space>((HYPRE_Int*)(HashKeys + j)));
      HYPRE_Int old = obj.compare_exchange_strong(-1, key, sycl::memory_order::relaxed,
                                                  sycl::memory_order::relaxed);
      // enable the following for SYCL 2020:
      //relaxed_atomic_ref<HYPRE_Int*>((HYPRE_Int *)(HashKeys + j)).compare_exchange_strong(-1, key);

      if (old == -1)
      {
         /* new insertion, increase counter */
         count++;
         /* this slot was open, insert value */
         if (attempt == 2 || failed == 0 || *warp_failed == 0)
         {
	   sycl::atomic<HYPRE_Complex, sycl::access::address_space::local_space>
	     obj(sycl::multi_ptr<HYPRE_Complex, sycl::access::address_space::local_space>((HYPRE_Complex*)(HashVals + j)));
	   sycl::atomic_fetch_add(obj, val, sycl::memory_order::relaxed);
	   // enable the following for SYCL 2020:
	   // relaxed_atomic_ref<HYPRE_Complex*>( HashVals + j ).fetch_add( val );
         }
         return j;
      }

      if (old == key)
      {
         /* this slot contains 'key', update value */
         if (attempt == 2 || failed == 0 || *warp_failed == 0)
         {
	   sycl::atomic<HYPRE_Complex, sycl::access::address_space::local_space>
	     obj(sycl::multi_ptr<HYPRE_Complex, sycl::access::address_space::local_space>((HYPRE_Complex*)(HashVals + j)));
	   sycl::atomic_fetch_add(obj, val, sycl::memory_order::relaxed);
	   // enable the following for SYCL 2020:
	   //relaxed_atomic_ref<HYPRE_Complex *>((HYPRE_Complex *)(HashVals + j)).fetch_add(val);
         }
         return j;
      }
   }

   return -1;
}

template <HYPRE_Int attempt, char HashType>
static __inline__ __attribute__((always_inline))
HYPRE_Int
csr_spmm_compute_row_attempt(
    HYPRE_Int rowi, volatile HYPRE_Int lane_id, HYPRE_Int *ia, HYPRE_Int *ja,
    HYPRE_Complex *aa, HYPRE_Int *ib, HYPRE_Int *jb, HYPRE_Complex *ab,
    HYPRE_Int s_HashSize, volatile HYPRE_Int *s_HashKeys,
    volatile HYPRE_Complex *s_HashVals, HYPRE_Int g_HashSize,
    HYPRE_Int *g_HashKeys, HYPRE_Complex *g_HashVals, char &failed,
    volatile char *warp_s_failed, sycl::nd_item<3>& item)
{
   /* load the start and end position of row i of A */
   HYPRE_Int j = -1;
   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }

   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int blockDim_x = item.get_local_range().get(2);
   HYPRE_Int blockDim_y = item.get_local_range().get(1);
   HYPRE_Int blockDim_z = item.get_local_range().get(0);
   HYPRE_Int threadIdx_x = item.get_local_id(2);
   HYPRE_Int threadIdx_y = item.get_local_id(1);

   const HYPRE_Int istart = SG.shuffle(j, 0);
   const HYPRE_Int iend = SG.shuffle(j, 1);

   HYPRE_Int num_new_insert = 0;

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

      /* threads in the same ygroup work on one row together */
      const HYPRE_Int rowB = SG.shuffle(colA, 0);
      const HYPRE_Complex mult = SG.shuffle(valA, 0);
      /* open this row of B, collectively */
      HYPRE_Int tmp = -1;
      if (rowB != -1 && threadIdx_x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx_x);
      }

      const HYPRE_Int rowB_start = SG.shuffle(tmp, 0);
      const HYPRE_Int rowB_end = SG.shuffle(tmp, 1);

      for (HYPRE_Int k = rowB_start; k < rowB_end; k += blockDim_x)
      {
         if (k + threadIdx_x < rowB_end)
         {
            const HYPRE_Int k_idx = read_only_load(jb + k + threadIdx_x);
            const HYPRE_Complex k_val = read_only_load(ab + k + threadIdx_x) * mult;
            /* first try to insert into shared memory hash table */
            HYPRE_Int pos = hash_insert_attempt<HashType, attempt>
               (s_HashSize, s_HashKeys, s_HashVals, k_idx, k_val, num_new_insert,
                failed, warp_s_failed);

            if (-1 == pos)
            {
               pos = hash_insert_attempt<HashType, attempt>
                     (g_HashSize, g_HashKeys, g_HashVals, k_idx, k_val, num_new_insert,
                      failed, warp_s_failed);
            }
            /* if failed again, both hash tables must have been full
               (hash table size estimation was too small).
               Increase the counter anyhow (will lead to over-counting)
               */
            if (pos == -1)
            {
               num_new_insert ++;
               failed = 1;
               if (attempt == 1)
               {
                  *warp_s_failed = 1;
               }
            }
         }
      }
   }

   return num_new_insert;
}

template <HYPRE_Int NUM_SUBGROUPS_PER_WG, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int attempt, char HashType>
void
csr_spmm_attempt(HYPRE_Int  M, /* HYPRE_Int K, HYPRE_Int N, */
                 HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Complex *aa,
                 HYPRE_Int *ib, HYPRE_Int *jb, HYPRE_Complex *ab,
                                HYPRE_Int *js, HYPRE_Complex *as,
                 HYPRE_Int *ig, HYPRE_Int *jg, HYPRE_Complex *ag,
                 HYPRE_Int *rc, HYPRE_Int *rg,
		 sycl::nd_item<3>& item,
                 volatile HYPRE_Int *s_HashKeys, // shared
                 volatile HYPRE_Complex *s_HashVals, // shared
		 volatile char *s_failed) // shared
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);
   HYPRE_Int blockDim_x = item.get_local_range().get(2);
   HYPRE_Int blockDim_y = item.get_local_range().get(1);
   HYPRE_Int blockDim_z = item.get_local_range().get(0);

   volatile const HYPRE_Int num_subgroups =
       NUM_SUBGROUPS_PER_WG * item.get_group_range(2);
   /* subgroup id inside the WG */
   volatile const HYPRE_Int subgroup_id = SG.get_group_linear_id();
   /* lane id inside the subgroup */
   volatile HYPRE_Int lane_id = SG.get_local_linear_id();

   /* shared memory hash table for this warp */
   volatile HYPRE_Int  *warp_s_HashKeys = s_HashKeys + subgroup_id * SHMEM_HASH_SIZE;
   volatile HYPRE_Complex *warp_s_HashVals = s_HashVals + subgroup_id * SHMEM_HASH_SIZE;
   /* shared memory failed flag for warps */
   volatile char *warp_s_failed = s_failed + subgroup_id;

#ifdef HYPRE_DEBUG
   assert(blockDim_z              == NUM_SUBGROUPS_PER_WG);
   assert(blockDim_x * blockDim_y == sub_group_size);
   assert(NUM_SUBGROUPS_PER_WG <= sub_group_size);
#endif

   for (HYPRE_Int i = item.get_group(2) * NUM_SUBGROUPS_PER_WG + subgroup_id;
        i < M; i += num_subgroups)
   {
      /* start/end position of global memory hash table */
      HYPRE_Int j = -1, istart_g, iend_g, ghash_size;
      char failed = 0;

      if (lane_id < 2)
      {
         j = read_only_load(ig + i + lane_id);
      }

      istart_g = SG.shuffle(j, 0);
      iend_g = SG.shuffle(j, 1);

      /* size of global hash table allocated for this row (must be power of 2) */
      ghash_size = iend_g - istart_g;

      if (attempt == 2)
      {
         if (ghash_size == 0)
         {
            continue;
         }
      }

      /* initialize warp's shared failed flag */
      if (attempt == 1 && lane_id == 0)
      {
         *warp_s_failed = 0;
      }
      /* initialize warp's shared and global memory hash table */
#pragma unrolll
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += sub_group_size)
      {
         warp_s_HashKeys[k] = -1;
         warp_s_HashVals[k] = 0.0;
      }
#pragma unrolll
      for (HYPRE_Int k = lane_id; k < ghash_size; k += sub_group_size)
      {
         jg[istart_g+k] = -1;
         ag[istart_g+k] = 0.0;
      }

      SG.barrier();

      /* work with two hash tables */
      j = csr_spmm_compute_row_attempt<attempt, HashType>(
          i, lane_id, ia, ja, aa, ib, jb, ab, SHMEM_HASH_SIZE, warp_s_HashKeys,
          warp_s_HashVals, ghash_size, jg + istart_g, ag + istart_g, failed,
          warp_s_failed, item);

#ifdef HYPRE_DEBUG
      if (attempt == 2)
      {
         assert(failed == 0);
      }
#endif

      /* num of inserts in this row (an upper bound) */
      j = warp_reduce_sum(j, item);

      if (attempt == 1)
      {
	failed = warp_allreduce_sum(failed, item);
      }

      if (attempt == 1 && failed)
      {
         if (lane_id == 0)
         {
            rg[i] = next_power_of_2(j - SHMEM_HASH_SIZE);
         }
      }
      else
      {
         if (lane_id == 0)
         {
            rc[i] = j;
            if (attempt == 1)
            {
               rg[i] = 0;
            }
         }
#pragma unroll
         for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += sub_group_size)
         {
            js[i*SHMEM_HASH_SIZE + k] = warp_s_HashKeys[k];
            as[i*SHMEM_HASH_SIZE + k] = warp_s_HashVals[k];
         }
      }
   } // for (i=...)
}

template <HYPRE_Int NUM_SUBGROUPS_PER_WG, HYPRE_Int SHMEM_HASH_SIZE>
static __inline__ __attribute__((always_inline))
HYPRE_Int
copy_from_hash_into_C_row(
    HYPRE_Int lane_id, volatile HYPRE_Int *s_HashKeys,
    volatile HYPRE_Complex *s_HashVals, HYPRE_Int ghash_size,
    HYPRE_Int *jg_start, HYPRE_Complex *ag_start, HYPRE_Int *jc_start,
    HYPRE_Complex *ac_start, sycl::nd_item<3>& item)
{
   HYPRE_Int j = 0;
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);

   /* copy shared memory hash table into C */
#pragma unrolll
   for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += sub_group_size)
   {
      HYPRE_Int key, sum, pos;
      key = s_HashKeys[k];
      HYPRE_Int in = key != -1;
      pos = warp_prefix_sum(lane_id, in, sum, item);
      if (key != -1)
      {
         jc_start[j + pos] = key;
         ac_start[j + pos] = s_HashVals[k];
      }
      j += sum;
   }

   /* copy global memory hash table into C */
#pragma unrolll
   for (HYPRE_Int k = 0; k < ghash_size; k += sub_group_size)
   {
      HYPRE_Int key = -1, sum, pos;
      if (k + lane_id < ghash_size)
      {
         key = jg_start[k + lane_id];
      }
      HYPRE_Int in = key != -1;
      pos = warp_prefix_sum(lane_id, in, sum, item);
      if (key != -1)
      {
         jc_start[j + pos] = key;
         ac_start[j + pos] = ag_start[k + lane_id];
      }
      j += sum;
   }

   return j;
}

template <HYPRE_Int NUM_SUBGROUPS_PER_WG, HYPRE_Int SHMEM_HASH_SIZE>
void
copy_from_hash_into_C(sycl::nd_item<3>& item,
		      HYPRE_Int  M,   HYPRE_Int *js,  HYPRE_Complex *as,
                      HYPRE_Int *ig1, HYPRE_Int *jg1, HYPRE_Complex *ag1,
                      HYPRE_Int *ig2, HYPRE_Int *jg2, HYPRE_Complex *ag2,
                      HYPRE_Int *ic, HYPRE_Int *jc, HYPRE_Complex *ac)
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);
   HYPRE_Int blockDim_x = item.get_local_range().get(2);
   HYPRE_Int blockDim_y = item.get_local_range().get(1);

   const HYPRE_Int num_subgroups =
       NUM_SUBGROUPS_PER_WG * item.get_group_range(2);
   /* subgroup id inside the WG */
   const HYPRE_Int subgroup_id = SG.get_group_linear_id();
   /* lane id inside the subgroup */
   volatile const HYPRE_Int lane_id = SG.get_local_linear_id();

#ifdef HYPRE_DEBUG
   assert(blockDim_x * blockDim_y == sub_group_size);
#endif

   for (HYPRE_Int i = item.get_group(2) * NUM_SUBGROUPS_PER_WG + subgroup_id;
        i < M; i += num_subgroups)
   {
      HYPRE_Int kc, kg1, kg2;

      /* start/end position in C */
      if (lane_id < 2)
      {
         kc  = read_only_load(ic  + i + lane_id);
         kg1 = read_only_load(ig1 + i + lane_id);
         kg2 = read_only_load(ig2 + i + lane_id);
      }

      HYPRE_Int istart_c = SG.shuffle(kc, 0);
#ifdef HYPRE_DEBUG
      HYPRE_Int iend_c    = SG.shuffle(kc, 1);
#endif

      HYPRE_Int istart_g1 = SG.shuffle(kg1, 0);
      HYPRE_Int iend_g1 = SG.shuffle(kg1, 1);
      HYPRE_Int istart_g2 = SG.shuffle(kg2, 0);
      HYPRE_Int iend_g2 = SG.shuffle(kg2, 1);
      HYPRE_Int g1_size = iend_g1 - istart_g1;
      HYPRE_Int g2_size = iend_g2 - istart_g2;

#ifdef HYPRE_DEBUG
      HYPRE_Int j;
#endif

      if (g2_size == 0)
      {
#ifdef HYPRE_DEBUG
         j =
#endif
         copy_from_hash_into_C_row<NUM_SUBGROUPS_PER_WG, SHMEM_HASH_SIZE>
         (lane_id, js + i * SHMEM_HASH_SIZE, as + i * SHMEM_HASH_SIZE, g1_size, jg1 + istart_g1,
	  ag1 + istart_g1, jc + istart_c, ac + istart_c, item);
      }
      else
      {
#ifdef HYPRE_DEBUG
         j =
#endif
         copy_from_hash_into_C_row<NUM_SUBGROUPS_PER_WG, SHMEM_HASH_SIZE>
         (lane_id, js + i * SHMEM_HASH_SIZE, as + i * SHMEM_HASH_SIZE, g2_size, jg2 + istart_g2,
	  ag2 + istart_g2, jc + istart_c, ac + istart_c, item);
      }
#ifdef HYPRE_DEBUG
      assert(istart_c + j == iend_c);
#endif
   }
}

/* SpGeMM with Rownnz Estimates */
HYPRE_Int
hypreDevice_CSRSpGemmWithRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                        HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                                        HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b,
                                        HYPRE_Int *d_rc,
                                        HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out,
                                        HYPRE_Int *nnzC)
{
   const HYPRE_Int num_subgroups_per_WG =  20;
   const HYPRE_Int shmem_hash_size     = 128;
   const HYPRE_Int BDIMX               =   2;
   const HYPRE_Int BDIMY               =  16;

   /* SYCL kernel configurations */
   sycl::range<3> bDim(num_subgroups_per_WG, BDIMY, BDIMX);
   hypre_assert(bDim[2] * bDim[1] == HYPRE_SUBGROUP_SIZE);

   // for cases where one SUBGROUP works on a row
   sycl::range<3> gDim(1, 1, (m + bDim[0] - 1) / bDim[0]);

   char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   /* ---------------------------------------------------------------------------
    * build hash table
    * ---------------------------------------------------------------------------*/
   HYPRE_Int  *d_ghash_i, *d_ghash_j, ghash_size;
   HYPRE_Complex *d_ghash_a;
   csr_spmm_create_hash_table(m, d_rc, nullptr, shmem_hash_size, m,
                              &d_ghash_i, &d_ghash_j, &d_ghash_a, &ghash_size);

   size_t m_ul = m;

   HYPRE_Int     *d_ic       = hypre_TAlloc(HYPRE_Int,     m+1,                  HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_ghash2_i = hypre_TAlloc(HYPRE_Int,     m+1,                  HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *d_js       = hypre_TAlloc(HYPRE_Int,     shmem_hash_size*m_ul, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *d_as       = hypre_TAlloc(HYPRE_Complex, shmem_hash_size*m_ul, HYPRE_MEMORY_DEVICE);

   /* ---------------------------------------------------------------------------
    * 1st multiplication attempt:
    * ---------------------------------------------------------------------------*/
   hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) {
       sycl::range<1> shared_range(num_subgroups_per_WG * shmem_hash_size);
       sycl::accessor<HYPRE_Int, 1, sycl::access_mode::read_write,
                      sycl::target::local> s_HashKeys_acc(shared_range, cgh);
       sycl::accessor<HYPRE_Complex, 1, sycl::access_mode::read_write,
                      sycl::target::local> s_HashVals_acc(shared_range, cgh);
       sycl::accessor<char, 1, sycl::access_mode::read_write,
                      sycl::target::local> s_failed_acc(sycl::range<1>(num_subgroups_per_WG), cgh);

       if (hash_type == 'L')
       {
         cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                          [=] (sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                            csr_spmm_attempt<num_subgroups_per_WG, shmem_hash_size, 1, 'L'>(
                              m, /*k, n,*/ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash_i, d_ghash_j, d_ghash_a,
                              d_ic + 1, d_ghash2_i + 1,
                              item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer(), s_failed_acc.get_pointer());
                          });
       }
       else if (hash_type == 'Q')
       {
         cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                          [=] (sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                            csr_spmm_attempt<num_subgroups_per_WG, shmem_hash_size, 1, 'Q'>(
                              m, /*k, n,*/ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash_i, d_ghash_j, d_ghash_a,
                              d_ic + 1, d_ghash2_i + 1,
                              item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer(), s_failed_acc.get_pointer());
                          });
       }
       else if (hash_type == 'D')
       {
         cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                          [=] (sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                            csr_spmm_attempt<num_subgroups_per_WG, shmem_hash_size, 1, 'D'>(
                              m, /*k, n,*/ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash_i, d_ghash_j, d_ghash_a,
                              d_ic + 1, d_ghash2_i + 1,
                              item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer(), s_failed_acc.get_pointer());
                          });
       }
       else
       {
         printf("Unrecognized hash type ... [L, Q, D]\n");
         exit(0);
       }
     });

   /* ---------------------------------------------------------------------------
    * build a secondary hash table for long rows
    * ---------------------------------------------------------------------------*/
   HYPRE_Int ghash2_size, *d_ghash2_j;
   HYPRE_Complex *d_ghash2_a;

   csr_spmm_create_ija(m, d_ghash2_i, &d_ghash2_j, &d_ghash2_a, &ghash2_size);

   /* ---------------------------------------------------------------------------
    * 2nd multiplication attempt:
    * ---------------------------------------------------------------------------*/
   hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) {
       sycl::range<1> shared_range(num_subgroups_per_WG * shmem_hash_size);
       sycl::accessor<HYPRE_Int, 1, sycl::access_mode::read_write,
                      sycl::target::local> s_HashKeys_acc(shared_range, cgh);
       sycl::accessor<HYPRE_Complex, 1, sycl::access_mode::read_write,
                      sycl::target::local> s_HashVals_acc(shared_range, cgh);
       sycl::accessor<char, 1, sycl::access_mode::read_write,
                      sycl::target::local> s_failed_acc(sycl::range<1>(num_subgroups_per_WG), cgh);

       if (ghash2_size > 0)
       {
         if (hash_type == 'L')
         {
           cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                            [=] (sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                              csr_spmm_attempt<num_subgroups_per_WG, shmem_hash_size, 2, 'L'>(
                                m, /*k, n,*/ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash2_i, d_ghash2_j, d_ghash2_a,
                                d_ic + 1, nullptr,
                                item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer(), s_failed_acc.get_pointer());
                            });
         }
         else if (hash_type == 'Q')
         {
           cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                            [=] (sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                              csr_spmm_attempt<num_subgroups_per_WG, shmem_hash_size, 2, 'Q'>(
                                m, /*k, n,*/ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash2_i, d_ghash2_j, d_ghash2_a,
                                d_ic + 1, nullptr,
                                item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer(), s_failed_acc.get_pointer());
                            });
         }
         else if (hash_type == 'D')
         {
           cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                            [=] (sycl::nd_item<3> item) [[cl::intel_reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                              csr_spmm_attempt<num_subgroups_per_WG, shmem_hash_size, 2, 'D'>(
                                m, /*k, n,*/ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_js, d_as, d_ghash2_i, d_ghash2_j, d_ghash2_a,
                                d_ic + 1, nullptr,
                                item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer(), s_failed_acc.get_pointer());
                            });
         }
         else
         {
           printf("Unrecognized hash type ... [L, Q, D]\n");
           exit(0);
         }
       }
     });

   HYPRE_Int nnzC_gpu, *d_jc;
   HYPRE_Complex *d_c;
   csr_spmm_create_ija(m, d_ic, &d_jc, &d_c, &nnzC_gpu);

   HYPRE_SYCL_3D_LAUNCH( (copy_from_hash_into_C<num_subgroups_per_WG, shmem_hash_size>),
			 gDim, bDim,
			 m, d_js, d_as, d_ghash_i, d_ghash_j, d_ghash_a, d_ghash2_i, d_ghash2_j, d_ghash2_a,
			 d_ic, d_jc, d_c);

   hypre_TFree(d_ghash_i,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_a,  HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash2_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash2_j, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash2_a, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_js,       HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_as,       HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC     = nnzC_gpu;

   return hypre_error_flag;
}

#endif /* HYPRE_USING_SYCL */
