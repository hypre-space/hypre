/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_SYCL)

template <typename T>
using relaxed_atomic_ref =
  sycl::ONEAPI::atomic_ref< T, sycl::ONEAPI::memory_order::relaxed,
                                sycl::ONEAPI::memory_scope::device,
                                sycl::access::address_space::global_space>;

/* assume d_i is of length (m+1) and contains the "sizes" in d_i[1], ..., d_i[m]
   the value of d_i[0] is not assumed
*/
void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz)
{
  sycl::queue* q = hypre_HandleSyclComputeQueue(hypre_handle());
  q->memset(d_i, 0, sizeof(HYPRE_Int)).wait();
  /* make ghash pointers by prefix scan */
  HYPRE_ONEDPL_CALL(inclusive_scan, d_i, d_i + m + 1, d_i);
  /* total size */
  q->memcpy(nnz, d_i + m, sizeof(HYPRE_Int)).wait();
  if (d_j)
  {
    *d_j = hypre_TAlloc(HYPRE_Int, *nnz, HYPRE_MEMORY_DEVICE);
  }
  if (d_a)
  {
    *d_a = hypre_TAlloc(HYPRE_Complex, *nnz, HYPRE_MEMORY_DEVICE);
  }
}

/* assume d_c is of length m and contains the "sizes" */
void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_c, HYPRE_Int **d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz)
{
  sycl::queue* q = hypre_HandleSyclComputeQueue(hypre_handle());

  *d_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
  q->memset(*d_i, 0, sizeof(HYPRE_Int)).wait();
  /* make ghash pointers by prefix scan */
  HYPRE_ONEDPL_CALL(inclusive_scan, d_c, d_c + m, *d_i + 1);
  /* total size */
  q->memcpy(nnz, (*d_i) + m, sizeof(HYPRE_Int)).wait();
  if (d_j)
  {
    *d_j = hypre_TAlloc(HYPRE_Int, *nnz, HYPRE_MEMORY_DEVICE);
  }
  if (d_a)
  {
    *d_a = hypre_TAlloc(HYPRE_Complex, *nnz, HYPRE_MEMORY_DEVICE);
  }
}


void csr_spmm_get_ghash_size(sycl::nd_item<3>& item, HYPRE_Int n, HYPRE_Int *rc, HYPRE_Int *rf, HYPRE_Int *rg, HYPRE_Int SHMEM_HASH_SIZE)
{
#ifdef HYPRE_DEBUG
  assert(item.get_local_range(2) * item.get_local_range(1) == item.get_sub_group().get_local_range().get(0));
#endif

  const HYPRE_Int global_thread_id = item.get_global_linear_id();
  const HYPRE_Int total_num_threads = item.get_global_range(2) * item.get_global_range(1) *
    item.get_global_range(0);

  for (HYPRE_Int i = global_thread_id; i < n; i+= total_num_threads)
  {
    HYPRE_Int j = (!rf || rf[i]) ? next_power_of_2(rc[i] - SHMEM_HASH_SIZE) : 0;
    rg[i] = j;
  }
}


void csr_spmm_get_ghash_size(sycl::nd_item<3>& item, HYPRE_Int n, HYPRE_Int num_ghash, HYPRE_Int *rc, HYPRE_Int *rf, HYPRE_Int *rg, HYPRE_Int SHMEM_HASH_SIZE)
{
#ifdef HYPRE_DEBUG
  assert(item.get_local_range(2) * item.get_local_range(1) == item.get_sub_group().get_local_range().get(0));
#endif

  const HYPRE_Int global_thread_id = item.get_global_linear_id();
  const HYPRE_Int total_num_threads = item.get_global_range(2) * item.get_global_range(1) *
    item.get_global_range(0);

  for (HYPRE_Int i = global_thread_id; i < n; i+= total_num_threads)
  {
    HYPRE_Int j = (!rf || rf[i]) ? next_power_of_2(rc[i] - SHMEM_HASH_SIZE) : 0;
    if (j)
    {
      relaxed_atomic_ref<HYPRE_Int>( rg[i % num_ghash] ).fetch_max(j);
    }
  }
}

HYPRE_Int
csr_spmm_create_hash_table(HYPRE_Int m, HYPRE_Int *d_rc, HYPRE_Int *d_rf, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int num_ghash,
                           HYPRE_Int **d_ghash_i, HYPRE_Int **d_ghash_j, HYPRE_Complex **d_ghash_a, HYPRE_Int *ghash_size)
{
  const HYPRE_Int num_subgroups_per_WG =  20;
  const HYPRE_Int BDIMX               =   4;
  const HYPRE_Int BDIMY               =   8;

  sycl::range<3> bDim(num_subgroups_per_WG, BDIMY, BDIMX);
  sycl::range<3> gDim(1, 1, (m + bDim[0] * HYPRE_WARP_SIZE - 1) / (bDim[0] * HYPRE_SUBGROUP_SIZE));

  hypre_assert(num_ghash <= m);

  *d_ghash_i = hypre_TAlloc(HYPRE_Int, num_ghash + 1, HYPRE_MEMORY_DEVICE);

  if (num_ghash == m)
  {
    HYPRE_SYCL_3D_LAUNCH( csr_spmm_get_ghash_size, gDim, bDim, m, d_rc, d_rf, (*d_ghash_i) + 1, SHMEM_HASH_SIZE );
  }
  else
  {
    sycl::queue* q = hypre_HandleSyclComputeQueue(hypre_handle());
    q->memset(*d_ghash_i, 0, (num_ghash + 1) * sizeof(HYPRE_Int)).wait();
    HYPRE_SYCL_3D_LAUNCH( csr_spmm_get_ghash_size, gDim, bDim, m, num_ghash, d_rc, d_rf, (*d_ghash_i) + 1, SHMEM_HASH_SIZE );
  }

  csr_spmm_create_ija(num_ghash, *d_ghash_i, d_ghash_j, d_ghash_a, ghash_size);

  return hypre_error_flag;
}

#endif /* HYPRE_USING_SYCL */
