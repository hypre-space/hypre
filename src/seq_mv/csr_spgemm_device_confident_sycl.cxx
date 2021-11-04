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
#include "csr_spgemm_device.hpp"
#include <cmath>

#include <algorithm>

#if defined(HYPRE_USING_SYCL)

template <typename T>
using relaxed_atomic_ref =
  sycl::ext::oneapi::atomic_ref< T, sycl::ext::oneapi::memory_order::relaxed,
                            sycl::ext::oneapi::memory_scope::device,
                            sycl::access::address_space::local_space>;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Numerical Multiplication
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

template <char HashType, HYPRE_Int FAILED_SYMBL>
static __inline__ __attribute__((always_inline))
HYPRE_Int
hash_insert_numer(
    HYPRE_Int HashSize,           /* capacity of the hash table */
    volatile HYPRE_Int *HashKeys, /* assumed to be initialized as all -1's */
    volatile HYPRE_Complex *HashVals, /* assumed to be initialized as all 0's */
    HYPRE_Int key,                    /* assumed to be nonnegative */
    HYPRE_Complex val, HYPRE_Int &count)
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

      if (old == -1 || old == key)
      {
         if (FAILED_SYMBL)
         {
            if (old == -1)
            {
               count++;
            }
         }
         /* this slot was open or contained 'key', update value */
         sycl::atomic<HYPRE_Complex, sycl::access::address_space::local_space>
           obj(sycl::multi_ptr<HYPRE_Complex, sycl::access::address_space::local_space>((HYPRE_Complex*)(HashVals + j)));
         sycl::atomic_fetch_add(obj, val, sycl::memory_order::relaxed);
         // enable the following for SYCL 2020:
         // relaxed_atomic_ref<HYPRE_Complex*>( HashVals + j ).fetch_add( val );

         return j;
      }
   }

   return -1;
}

template <HYPRE_Int FAILED_SYMBL, char HashType>
static __inline__ __attribute__((always_inline))
HYPRE_Int
csr_spmm_compute_row_numer(
    HYPRE_Int rowi, HYPRE_Int lane_id, HYPRE_Int *ia, HYPRE_Int *ja,
    HYPRE_Complex *aa, HYPRE_Int *ib, HYPRE_Int *jb, HYPRE_Complex *ab,
    HYPRE_Int s_HashSize, volatile HYPRE_Int *s_HashKeys,
    volatile HYPRE_Complex *s_HashVals, HYPRE_Int g_HashSize,
    HYPRE_Int *g_HashKeys, HYPRE_Complex *g_HashVals, sycl::nd_item<3>& item)
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int threadIdx_x = item.get_local_id(2);
   HYPRE_Int threadIdx_y = item.get_local_id(1);
   HYPRE_Int blockDim_y = item.get_local_range().get(1);
   HYPRE_Int blockDim_x = item.get_local_range().get(2);

   /* load the start and end position of row i of A */
   HYPRE_Int i;
   if (lane_id < 2)
   {
      i = read_only_load(ia + rowi + lane_id);
   }

   const HYPRE_Int istart = SG.shuffle(i, 0);
   const HYPRE_Int iend = SG.shuffle(i, 1);

   HYPRE_Int num_new_insert = 0;

   /* load column idx and values of row i of A */
   for (i = istart; i < iend; i += blockDim_y)
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
            HYPRE_Int pos = hash_insert_numer<HashType, FAILED_SYMBL>
               (s_HashSize, s_HashKeys, s_HashVals, k_idx, k_val, num_new_insert);
            if (-1 == pos)
            {
               pos = hash_insert_numer<HashType, FAILED_SYMBL>
                     (g_HashSize, g_HashKeys, g_HashVals, k_idx, k_val, num_new_insert);
            }
#ifdef HYPRE_DEBUG
            assert(pos != -1);
#endif
         }
      }
   }

   return num_new_insert;
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
   HYPRE_Int sub_group_size = item.get_sub_group().get_local_range().get(0);

   HYPRE_Int j = 0;

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

template <HYPRE_Int NUM_SUBGROUPS_PER_WG, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int FAILED_SYMBL, char HashType>
void
csr_spmm_numeric(HYPRE_Int  M, /* HYPRE_Int K, HYPRE_Int N, */
                 HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Complex *aa,
                 HYPRE_Int *ib, HYPRE_Int *jb, HYPRE_Complex *ab,
                 HYPRE_Int *ic, HYPRE_Int *jc, HYPRE_Complex *ac,
                 HYPRE_Int *rc,
                 HYPRE_Int *ig, HYPRE_Int *jg, HYPRE_Complex *ag,
                 sycl::nd_item<3>& item,
                 volatile HYPRE_Int *s_HashKeys, // shared
                 volatile HYPRE_Complex *s_HashVals) // shared
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);

   /* total number of sub-groups in global iteration space (no_of_sub_groups_per_WG * total_no_of_WGs) */
   volatile const HYPRE_Int num_subgroups = NUM_SUBGROUPS_PER_WG *
     (item.get_group_range(0) * item.get_group_range(1) * item.get_group_range(2));
   /* subgroup id inside the WG */
   volatile const HYPRE_Int subgroup_id = SG.get_group_linear_id();
   /* subgroup id in the grid */
   volatile const HYPRE_Int grid_subgroup_id =
       item.get_group(2) * NUM_SUBGROUPS_PER_WG + subgroup_id;
   /* lane id inside the sub-group */
   volatile HYPRE_Int lane_id = SG.get_local_linear_id();

   /* shared memory hash table for this warp */
   volatile HYPRE_Int  *warp_s_HashKeys = s_HashKeys + subgroup_id * SHMEM_HASH_SIZE;
   volatile HYPRE_Complex *warp_s_HashVals = s_HashVals + subgroup_id * SHMEM_HASH_SIZE;

#ifdef HYPRE_DEBUG
   assert(item.get_local_range().get(0) == NUM_SUBGROUPS_PER_WG);
   assert(item.get_local_range().get(2) * item.get_local_range().get(1) ==
          sub_group_size);
#endif

   /* a subgroup working on the ith row */
   for (HYPRE_Int i = grid_subgroup_id; i < M; i += num_subgroups)
   {
      /* start/end position of global memory hash table */
      HYPRE_Int j = -1, istart_g, iend_g, ghash_size, jsum;

      if (lane_id < 2)
      {
         j = read_only_load(ig + grid_subgroup_id + lane_id);
      }

      istart_g = SG.shuffle(j, 0);
      iend_g = SG.shuffle(j, 1);

      /* size of global hash table allocated for this row
         (must be power of 2 and >= the actual size of the row of C) */
      ghash_size = iend_g - istart_g;

      /* initialize subgroups's shared and global memory hash table */
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

      /* work with two hash tables. jsum is the (exact) nnz for row i */
      jsum = csr_spmm_compute_row_numer<FAILED_SYMBL, HashType>(
          i, lane_id, ia, ja, aa, ib, jb, ab, SHMEM_HASH_SIZE, warp_s_HashKeys,
          warp_s_HashVals, ghash_size, jg + istart_g, ag + istart_g, item);

      if (FAILED_SYMBL)
      {
         /* in the case when symb mult was failed, save row nnz into rc */
         /* num of nonzeros of this row of C (exact) */
        jsum = warp_reduce_sum(jsum, item);
         if (lane_id == 0)
         {
            rc[i] = jsum;
         }
      }

      /* copy results into the final C */
      /* start/end position in C */
#ifdef HYPRE_DEBUG
      if (lane_id < 2)
      {
         j = read_only_load(ic + i + lane_id);
      }

      HYPRE_Int istart_c = SG.shuffle(j, 0);
      HYPRE_Int iend_c = SG.shuffle(j, 1);
#else
      if (lane_id < 1)
      {
         j = read_only_load(ic + i);
      }
      HYPRE_Int istart_c = SG.shuffle(j, 0);
#endif

      j = copy_from_hash_into_C_row<NUM_SUBGROUPS_PER_WG, SHMEM_HASH_SIZE>
             (lane_id, warp_s_HashKeys, warp_s_HashVals, ghash_size, jg + istart_g,
              ag + istart_g, jc + istart_c, ac + istart_c, item);

#ifdef HYPRE_DEBUG
      if (FAILED_SYMBL)
      {
         assert(istart_c + j <= iend_c);
      }
      else
      {
         assert(istart_c + j == iend_c);
      }
#endif
   }
}

template <HYPRE_Int NUM_SUBGROUPS_PER_WG>
void
copy_from_Cext_into_C(sycl::nd_item<3>& item,
                      HYPRE_Int  M,
                      HYPRE_Int *ix, HYPRE_Int *jx, HYPRE_Complex *ax,
                      HYPRE_Int *ic, HYPRE_Int *jc, HYPRE_Complex *ac)
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);

   const HYPRE_Int num_subgroups = NUM_SUBGROUPS_PER_WG *
     (item.get_group_range(0) * item.get_group_range(1) * item.get_group_range(2));
   /* subgroup id inside the WG */
   const HYPRE_Int subgroup_id = SG.get_group_linear_id();
   /* lane id inside the subgroup */
   volatile const HYPRE_Int lane_id = SG.get_local_linear_id();

#ifdef HYPRE_DEBUG
   assert(item.get_local_range().get(2) * item.get_local_range().get(1) ==
          sub_group_size);
#endif

   for (HYPRE_Int i = item.get_group(2) * NUM_SUBGROUPS_PER_WG + subgroup_id;
        i < M; i += num_subgroups)
   {
      HYPRE_Int kc, kx;

      /* start/end position in C and X*/
      if (lane_id < 2)
      {
         kc = read_only_load(ic + i + lane_id);
         kx = read_only_load(ix + i + lane_id);
      }

      HYPRE_Int istart_c = SG.shuffle(kc, 0);
      HYPRE_Int iend_c = SG.shuffle(kc, 1);
      HYPRE_Int istart_x = SG.shuffle(kx, 0);

#ifdef HYPRE_DEBUG
      HYPRE_Int iend_x = SG.shuffle(kx, 1);
      assert(iend_c - istart_c <= iend_x - istart_x);
#endif

      HYPRE_Int p = istart_x - istart_c;
      for (HYPRE_Int k = istart_c + lane_id; k < iend_c; k += sub_group_size)
      {
         jc[k] = jx[k+p];
         ac[k] = ax[k+p];
      }
   }
}

/* SpGeMM with Rownnz Upper bound */
HYPRE_Int
hypreDevice_CSRSpGemmWithRownnzUpperbound(HYPRE_Int   m,        HYPRE_Int   k,        HYPRE_Int       n,
                                          HYPRE_Int  *d_ia,     HYPRE_Int  *d_ja,     HYPRE_Complex  *d_a,
                                          HYPRE_Int  *d_ib,     HYPRE_Int  *d_jb,     HYPRE_Complex  *d_b,
                                          HYPRE_Int  *d_rc,     HYPRE_Int   exact_rownnz,
                                          HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out,
                                          HYPRE_Int *nnzC)
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] -= hypre_MPI_Wtime();
#endif

  const HYPRE_Int num_subgroups_per_WG =  20;
  const HYPRE_Int shmem_hash_size     = 128;
  const HYPRE_Int BDIMX               =   2;
  const HYPRE_Int BDIMY               =  16;

  /* SYCL kernel configurations */
  sycl::range<3> bDim(num_subgroups_per_WG, BDIMY, BDIMX);
  hypre_assert(bDim[2] * bDim[1] == HYPRE_SUBGROUP_SIZE);

  // for cases where one SUB-GROUP works on a row
  HYPRE_Int num_subgroups = std::min(m, HYPRE_MAX_NUM_SUBGROUPS);
  sycl::range<3> gDim(1, 1, (num_subgroups + bDim[0] - 1) / bDim[0]);
  // number of active subgroups
  HYPRE_Int num_act_subgroups = std::min<unsigned int>(bDim[0] * gDim[2], m);

  char hash_type = hypre_HandleSpgemmHashType(hypre_handle());

  /* ---------------------------------------------------------------------------
   * build hash table
   * ---------------------------------------------------------------------------*/
  HYPRE_Int  *d_ghash_i, *d_ghash_j, ghash_size;
  HYPRE_Complex *d_ghash_a;
  csr_spmm_create_hash_table(m, d_rc, nullptr, shmem_hash_size, num_act_subgroups,
                             &d_ghash_i, &d_ghash_j, &d_ghash_a, &ghash_size);

  /* ---------------------------------------------------------------------------
   * numerical multiplication:
   * ---------------------------------------------------------------------------*/
  HYPRE_Int *d_ic, *d_jc, nnzC_nume, *d_ic_new = nullptr, *d_jc_new = nullptr, nnzC_nume_new = -1;
  HYPRE_Complex *d_c, *d_c_new = nullptr;

  /* if rc contains exact_rownnz: can allocate the final C directly;
     if rc contains upper bound : it is a temporary space that is more than enough to store C */
  csr_spmm_create_ija(m, d_rc, &d_ic, &d_jc, &d_c, &nnzC_nume);

  if (!exact_rownnz)
  {
    d_ic_new = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
  }

  if (hash_type != 'L' && hash_type != 'Q' && hash_type != 'D')
  {
    printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
    exit(0);
  }

  hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) {
      sycl::range<1> shared_range(num_subgroups_per_WG * shmem_hash_size);

      sycl::accessor<HYPRE_Int, 1, sycl::access_mode::read_write,
                     sycl::target::local> s_HashKeys_acc(shared_range, cgh);
      sycl::accessor<HYPRE_Complex, 1, sycl::access_mode::read_write,
                     sycl::target::local> s_HashVals_acc(shared_range, cgh);

      if (exact_rownnz)
      {
        if (hash_type == 'L')
        {
          cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                           [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                             csr_spmm_numeric<num_subgroups_per_WG, shmem_hash_size, 0, 'L'>(
                               m, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_ic_new + 1,
                               d_ghash_i, d_ghash_j, d_ghash_a,
                               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer());
                           });
        }
        else if (hash_type == 'Q')
        {
          cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                           [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                             csr_spmm_numeric<num_subgroups_per_WG, shmem_hash_size, 0, 'Q'>(
                               m, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_ic_new + 1,
                               d_ghash_i, d_ghash_j, d_ghash_a,
                               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer());
                           });
        }
        else if (hash_type == 'D')
        {
          cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                           [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                             csr_spmm_numeric<num_subgroups_per_WG, shmem_hash_size, 0, 'D'>(
                               m, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_ic_new + 1,
                               d_ghash_i, d_ghash_j, d_ghash_a,
                               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer());
                           });
        }
      }
      else
      {
        if (hash_type == 'L')
        {
          cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                           [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                             csr_spmm_numeric<num_subgroups_per_WG, shmem_hash_size, 1, 'L'>(
                               m, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_ic_new + 1,
                               d_ghash_i, d_ghash_j, d_ghash_a,
                               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer());
                           });
        }
        else if (hash_type == 'Q')
        {
          cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                           [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                             csr_spmm_numeric<num_subgroups_per_WG, shmem_hash_size, 1, 'Q'>(
                               m, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_ic_new + 1,
                               d_ghash_i, d_ghash_j, d_ghash_a,
                               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer());
                           });
        }
        else if (hash_type == 'D')
        {
          cgh.parallel_for(sycl::nd_range<3>(gDim*bDim, bDim),
                           [=] (sycl::nd_item<3> item) [[intel::reqd_sub_group_size(HYPRE_SUBGROUP_SIZE)]] {
                             csr_spmm_numeric<num_subgroups_per_WG, shmem_hash_size, 1, 'D'>(
                               m, /* k, n, */ d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_ic, d_jc, d_c, d_ic_new + 1,
                               d_ghash_i, d_ghash_j, d_ghash_a,
                               item, s_HashKeys_acc.get_pointer(), s_HashVals_acc.get_pointer());
                           });
        }
      }
    });

  if (!exact_rownnz)
  {
    /* alloc final C */
    csr_spmm_create_ija(m, d_ic_new, &d_jc_new, &d_c_new, &nnzC_nume_new);

    hypre_assert(nnzC_nume_new <= nnzC_nume);

    if (nnzC_nume_new < nnzC_nume)
    {
      /* copy to the final C */
      sycl::range<3> gDim(1, 1, (m + bDim[0] - 1) / bDim[0]);
      HYPRE_SYCL_3D_LAUNCH( (copy_from_Cext_into_C<num_subgroups_per_WG>),
                            gDim, bDim,
                            m, d_ic, d_jc, d_c, d_ic_new, d_jc_new, d_c_new );

      hypre_TFree(d_ic, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_jc, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_c,  HYPRE_MEMORY_DEVICE);

      d_ic = d_ic_new;
      d_jc = d_jc_new;
      d_c = d_c_new;
      nnzC_nume = nnzC_nume_new;
    }
    else
    {
      hypre_TFree(d_ic_new, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_jc_new, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_c_new,  HYPRE_MEMORY_DEVICE);
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
  hypre_HandleSyclComputeQueue(hypre_handle())->wait_and_throw();
  hypre_profile_times[HYPRE_TIMER_ID_SPMM_NUMERIC] += hypre_MPI_Wtime();
#endif

  return hypre_error_flag;
}

#endif /* HYPRE_USING_SYCL */
