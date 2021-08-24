/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.hpp"
#include <cmath>
#include <algorithm>

#if defined(HYPRE_USING_SYCL)

template <typename T>
using relaxed_atomic_ref =
  sycl::ONEAPI::atomic_ref< T, sycl::ONEAPI::memory_order::relaxed,
			    sycl::ONEAPI::memory_scope::device,
			    sycl::access::address_space::local_space>;

/*- - - - - - - - - - - - - - - - - - - - - - - - - -
 *- - - - - - - - - - - - - - - - - - - - - - - - - -
                Symbolic Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - -
 *- - - - - - - - - - - - - - - - - - - - - - - - - -
 */
template <char HashType>
static __inline__ __attribute__((always_inline))
HYPRE_Int hash_insert_symbl(
    HYPRE_Int HashSize,           /* capacity of the hash table */
    volatile HYPRE_Int *HashKeys, /* assumed to be initialized as all -1's */
    HYPRE_Int key,                /* assumed to be nonnegative */
    HYPRE_Int &count /* increase by 1 if is a new entry */)
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
static __inline__ __attribute__((always_inline))
HYPRE_Int csr_spmm_compute_row_symbl(
    HYPRE_Int rowi, HYPRE_Int lane_id, HYPRE_Int *ia, HYPRE_Int *ja,
    HYPRE_Int *ib, HYPRE_Int *jb, HYPRE_Int s_HashSize,
    volatile HYPRE_Int *s_HashKeys, HYPRE_Int g_HashSize, HYPRE_Int *g_HashKeys,
    char &failed, sycl::nd_item<3>& item)
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int threadIdx_x = item.get_local_id(2);
   HYPRE_Int threadIdx_y = item.get_local_id(1);
   HYPRE_Int blockDim_y = item.get_local_range().get(1);
   HYPRE_Int blockDim_x = item.get_local_range().get(2);

   /* load the start and end position of row i of A */
   HYPRE_Int j = -1;
   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }
   const HYPRE_Int istart = SG.shuffle(j, 0);
   const HYPRE_Int iend   = SG.shuffle(j, 1);

   HYPRE_Int num_new_insert = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart; i < iend; i += blockDim_y)
   {
      HYPRE_Int colA = -1;

      if (threadIdx_x == 0 && i + threadIdx_y < iend)
      {
	colA = read_only_load(ja + i + threadIdx_y);
      }

      /* work-items in the same ygroup work on one row together */
      const HYPRE_Int rowB = SG.shuffle(colA, 0, blockDim_x);
      /* open this row of B, collectively */
      HYPRE_Int tmp = -1;
      if (rowB != -1 && threadIdx_x < 2)
      {
	tmp = read_only_load(ib+rowB+threadIdx_x);
      }
      const HYPRE_Int rowB_start = SG.shuffle(tmp, 0, blockDim_x);
      const HYPRE_Int rowB_end   = SG.shuffle(tmp, 1, blockDim_x);

      for (HYPRE_Int k = rowB_start; k < rowB_end; k += blockDim_x)
      {
         if (k + threadIdx_x < rowB_end)
         {
            const HYPRE_Int k_idx = read_only_load(jb + k + threadIdx_x);
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

template <HYPRE_Int NUM_SUBGROUPS_PER_WG, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int ATTEMPT, char HashType>
void csr_spmm_symbolic(HYPRE_Int  M, /* HYPRE_Int K, HYPRE_Int N, */
                       HYPRE_Int *ia, HYPRE_Int *ja,
                       HYPRE_Int *ib, HYPRE_Int *jb,
                       HYPRE_Int *ig, HYPRE_Int *jg,
                       HYPRE_Int *rc, HYPRE_Int *rf,
		       sycl::nd_item<3>& item,
                       volatile HYPRE_Int *s_HashKeys) // shared
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);
   HYPRE_Int blockDim_x = item.get_local_range().get(2);
   HYPRE_Int blockDim_y = item.get_local_range().get(1);
   HYPRE_Int blockDim_z = item.get_local_range().get(0);

   volatile const HYPRE_Int num_subgroups = NUM_SUBGROUPS_PER_WG *
     (item.get_group_range(0) * item.get_group_range(1) * item.get_group_range(2));
   /* warp id inside the WG */
   volatile const HYPRE_Int subgroup_id = SG.get_group_linear_id();
   /* warp id in the grid */
   volatile const HYPRE_Int grid_subgroup_id =
       item.get_group(2) * NUM_SUBGROUPS_PER_WG + subgroup_id;
   /* lane id inside the subgroup */
   volatile HYPRE_Int lane_id = SG.get_local_linear_id();
   /* shared memory hash table for this subgroup */
   volatile HYPRE_Int *warp_s_HashKeys = s_HashKeys + subgroup_id * SHMEM_HASH_SIZE;

   char failed = 0;

#ifdef HYPRE_DEBUG
   assert(blockDim_z              == NUM_SUBGROUPS_PER_WG);
   assert(blockDim_x * blockDim_y == sub_group_size);
   assert(NUM_SUBGROUPS_PER_WG <= sub_group_size);
#endif

   for (HYPRE_Int i = grid_subgroup_id; i < M; i += num_subgroups)
   {
      HYPRE_Int j;

      if (ATTEMPT == 2)
      {
         if (lane_id == 0)
         {
            j = rf[i];
         }
         j = SG.shuffle(j, 0);
         if (j == 0)
         {
            continue;
         }
      }

      /* start/end position of global memory hash table */
      HYPRE_Int istart_g, iend_g, ghash_size;
      if (lane_id < 2)
      {
         j = read_only_load(ig + grid_subgroup_id + lane_id);
      }
      istart_g = SG.shuffle(j, 0);
      iend_g   = SG.shuffle(j, 1);

      /* size of global hash table allocated for this row
         (must be power of 2 and >= the actual size of the row of C) */
      ghash_size = iend_g - istart_g;

      /* initialize warp's shared and global memory hash table */
#pragma unrolll
      for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += sub_group_size)
      {
         warp_s_HashKeys[k] = -1;
      }
#pragma unrolll
      for (HYPRE_Int k = lane_id; k < ghash_size; k += sub_group_size)
      {
         jg[istart_g+k] = -1;
      }

      SG.barrier();

      /* work with two hash tables */
      j = csr_spmm_compute_row_symbl<HashType>(
          i, lane_id, ia, ja, ib, jb, SHMEM_HASH_SIZE, warp_s_HashKeys,
          ghash_size, jg + istart_g, failed, item);

#ifdef HYPRE_DEBUG
      if (ATTEMPT == 2)
      {
         assert(failed == 0);
      }
#endif

      /* num of nonzeros of this row (an upper bound) */
      j = warp_reduce_sum(j, item);

      /* if this row failed */
      if (ATTEMPT == 1)
      {
	failed = warp_reduce_sum(failed, item);
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

   const HYPRE_Int num_subgroups_per_WG =  20;
   const HYPRE_Int shmem_hash_size     = 256;//512;
   const HYPRE_Int BDIMX               =   2;
   const HYPRE_Int BDIMY               =  16;

   /* SYCL kernel configurations */
   sycl::range<3> bDim(num_subgroups_per_WG, BDIMY, BDIMX);
   hypre_assert(bDim[2] * bDim[1] == HYPRE_SUBGROUP_SIZE);

   // for cases where one SUB-GROUP works on a row
   HYPRE_Int num_subgroups = min(m, HYPRE_MAX_NUM_SUBGROUPS);
   sycl::range<3> gDim(1, 1, (num_subgroups + bDim[0] - 1) / bDim[0]);
   // number of active warps
   HYPRE_Int num_act_warps = std::min<unsigned int>(bDim[0] * gDim[2], m);

   char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/
   HYPRE_Int  *d_ghash_i, *d_ghash_j, ghash_size, *d_act;
   if (ATTEMPT == 1)
   {
      d_act = nullptr; /* all rows are active */
   }
   else
   {
      d_act = d_rf;
   }
   csr_spmm_create_hash_table(m, d_rc, d_act, shmem_hash_size, num_act_warps,
                              &d_ghash_i, &d_ghash_j, nullptr, &ghash_size);

   /* ---------------------------------------------------------------------------
    * symbolic multiplication:
    * On output, it provides an upper bound of nnz in rows of C
    * ---------------------------------------------------------------------------*/
   hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) {
       sycl::range<1> shared_range(num_subgroups_per_WG * shmem_hash_size);
       sycl::accessor<HYPRE_Int, 1, sycl::access_mode::read_write,
                      sycl::target::local> s_HashKeys_acc(shared_range, cgh);

       if (hash_type == 'L')
       {
         cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                          [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                            csr_spmm_symbolic<num_subgroups_per_WG, shmem_hash_size, ATTEMPT, 'L'>(
                              m, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf,
                              item, s_HashKeys_acc.get_pointer());
                          });
       }
       else if (hash_type == 'Q')
       {
         cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                          [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                            csr_spmm_symbolic<num_subgroups_per_WG, shmem_hash_size, ATTEMPT, 'Q'>(
                              m, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf,
                              item, s_HashKeys_acc.get_pointer());
                          });
       }
       else if (hash_type == 'D')
       {
         cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                          [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                            csr_spmm_symbolic<num_subgroups_per_WG, shmem_hash_size, ATTEMPT, 'D'>(
                              m, /*k, n,*/ d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf,
                              item, s_HashKeys_acc.get_pointer());
                          });
       }
       else
       {
         printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
         exit(0);
       }
     });

   hypre_TFree(d_ghash_i, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_ghash_j, HYPRE_MEMORY_DEVICE);
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

#ifdef HYPRE_DEBUG
      hypre_assert(hypreDevice_IntegerReduceSum(m, d_rf) == 0);
#endif
   }

   hypre_TFree(d_rf, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* HYPRE_USING_SYCL */
