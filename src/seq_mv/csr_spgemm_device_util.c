/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/* assume d_i is of length (m+1) and contains the "sizes" in d_i[1], ..., d_i[m]
   the value of d_i[0] is not assumed
 */
void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz)
{
   hypre_Memset(d_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   /* make ghash pointers by prefix scan */
   HYPRE_THRUST_CALL(inclusive_scan, d_i, d_i + m + 1, d_i);
   /* total size */
   hypre_TMemcpy(nnz, d_i + m, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
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
   *d_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
   hypre_Memset(*d_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   /* make ghash pointers by prefix scan */
   HYPRE_THRUST_CALL(inclusive_scan, d_c, d_c + m, *d_i + 1);

   /* total size */
   hypre_TMemcpy(nnz, (*d_i) + m, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   if (d_j)
   {
      *d_j = hypre_TAlloc(HYPRE_Int, *nnz, HYPRE_MEMORY_DEVICE);
   }
   if (d_a)
   {
      *d_a = hypre_TAlloc(HYPRE_Complex, *nnz, HYPRE_MEMORY_DEVICE);
   }
}

__global__
void csr_spmm_get_ghash_size(HYPRE_Int n, HYPRE_Int *rc, HYPRE_Int *rf, HYPRE_Int *rg, HYPRE_Int SHMEM_HASH_SIZE)
{
   hypre_device_assert(blockDim.x * blockDim.y == HYPRE_WARP_SIZE);

   const HYPRE_Int global_thread_id  = blockIdx.x * get_block_size() + get_thread_id();
   const HYPRE_Int total_num_threads = gridDim.x  * get_block_size();

   for (HYPRE_Int i = global_thread_id; i < n; i+= total_num_threads)
   {
      HYPRE_Int j = (!rf || rf[i]) ? next_power_of_2(rc[i] - SHMEM_HASH_SIZE) : 0;
      rg[i] = j;
   }
}

__global__
void csr_spmm_get_ghash_size(HYPRE_Int n, HYPRE_Int num_ghash, HYPRE_Int *rc, HYPRE_Int *rf, HYPRE_Int *rg, HYPRE_Int SHMEM_HASH_SIZE)
{
   hypre_device_assert(blockDim.x * blockDim.y == HYPRE_WARP_SIZE);

   const HYPRE_Int global_thread_id  = blockIdx.x * get_block_size() + get_thread_id();
   const HYPRE_Int total_num_threads = gridDim.x  * get_block_size();

   for (HYPRE_Int i = global_thread_id; i < n; i+= total_num_threads)
   {
      HYPRE_Int j = (!rf || rf[i]) ? next_power_of_2(rc[i] - SHMEM_HASH_SIZE) : 0;
      if (j)
      {
         atomicMax(&rg[i % num_ghash], j);
      }
   }
}

HYPRE_Int
csr_spmm_create_hash_table(HYPRE_Int m, HYPRE_Int *d_rc, HYPRE_Int *d_rf, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int num_ghash,
                           HYPRE_Int **d_ghash_i, HYPRE_Int **d_ghash_j, HYPRE_Complex **d_ghash_a, HYPRE_Int *ghash_size)
{
   const HYPRE_Int num_warps_per_block =  20;
   const HYPRE_Int BDIMX               =   4;
   const HYPRE_Int BDIMY               =   8;

   dim3 bDim(BDIMX, BDIMY, num_warps_per_block);
   dim3 gDim( (m + bDim.z * HYPRE_WARP_SIZE - 1) / (bDim.z * HYPRE_WARP_SIZE) );

   hypre_assert(num_ghash <= m);

   *d_ghash_i = hypre_TAlloc(HYPRE_Int, num_ghash + 1, HYPRE_MEMORY_DEVICE);

   if (num_ghash == m)
   {
      HYPRE_CUDA_LAUNCH( csr_spmm_get_ghash_size, gDim, bDim, m, d_rc, d_rf, (*d_ghash_i) + 1, SHMEM_HASH_SIZE );
   }
   else
   {
      hypre_Memset(*d_ghash_i, 0, (num_ghash + 1) * sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
      HYPRE_CUDA_LAUNCH( csr_spmm_get_ghash_size, gDim, bDim, m, num_ghash, d_rc, d_rf, (*d_ghash_i) + 1, SHMEM_HASH_SIZE );
   }

   csr_spmm_create_ija(num_ghash, *d_ghash_i, d_ghash_j, d_ghash_a, ghash_size);

   return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP) */
