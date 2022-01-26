/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/* assume d_c is of length m and contains the "sizes" */
void
hypre_create_ija( HYPRE_Int       m,
                  HYPRE_Int      *d_c,
                  HYPRE_Int      *d_i,
                  HYPRE_Int     **d_j,
                  HYPRE_Complex **d_a,
                  HYPRE_Int      *nnz )
{
   hypre_Memset(d_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL(inclusive_scan, d_c, d_c + m, d_i + 1);

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

__global__ void
hypre_SpGemmGhashSize1( HYPRE_Int  num_rows,
                        HYPRE_Int *row_id,
                        HYPRE_Int  num_ghash,
                        HYPRE_Int *row_sizes,
                        HYPRE_Int *ghash_sizes,
                        HYPRE_Int  SHMEM_HASH_SIZE )
{
   const HYPRE_Int global_thread_id = hypre_cuda_get_grid_thread_id<1, 1>();

   if (global_thread_id >= num_ghash)
   {
      return;
   }

   HYPRE_Int j = 0;

   for (HYPRE_Int i = global_thread_id; i < num_rows; i += num_ghash)
   {
      const HYPRE_Int rid = row_id ? read_only_load(&row_id[i]) : i;
      const HYPRE_Int rnz = read_only_load(&row_sizes[rid]);
      const HYPRE_Int j1 = next_power_of_2(rnz - SHMEM_HASH_SIZE);
      j = hypre_max(j, j1);
   }

   ghash_sizes[global_thread_id] = j;
}

__global__ void
hypre_SpGemmGhashSize2( HYPRE_Int  num_rows,
                        HYPRE_Int *row_id,
                        HYPRE_Int  num_ghash,
                        HYPRE_Int *row_sizes,
                        HYPRE_Int *ghash_sizes,
                        HYPRE_Int  SHMEM_HASH_SIZE )
{
   const HYPRE_Int i = hypre_cuda_get_grid_thread_id<1, 1>();

   if (i < num_rows)
   {
      const HYPRE_Int rid = row_id ? read_only_load(&row_id[i]) : i;
      const HYPRE_Int rnz = read_only_load(&row_sizes[rid]);
      ghash_sizes[rid] = next_power_of_2(rnz - SHMEM_HASH_SIZE);
   }
}

HYPRE_Int
hypre_SpGemmCreateGlobalHashTable( HYPRE_Int       num_rows,        /* number of rows */
                                   HYPRE_Int      *row_id,          /* row_id[i] is index of ith row; i if row_id == NULL */
                                   HYPRE_Int       num_ghash,       /* number of hash tables <= num_rows */
                                   HYPRE_Int      *row_sizes,       /* row_sizes[rowid[i]] is the size of ith row */
                                   HYPRE_Int       SHMEM_HASH_SIZE,
                                   HYPRE_Int     **ghash_i_ptr,     /* of length num_ghash + 1 */
                                   HYPRE_Int     **ghash_j_ptr,
                                   HYPRE_Complex **ghash_a_ptr,
                                   HYPRE_Int      *ghash_size_ptr,
                                   HYPRE_Int       type )
{
   hypre_assert(type == 2 || num_ghash <= num_rows);

   HYPRE_Int *ghash_i, ghash_size;
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();

   if (type == 1)
   {
      ghash_i = hypre_TAlloc(HYPRE_Int, num_ghash + 1, HYPRE_MEMORY_DEVICE);
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_ghash, "thread", bDim);
      HYPRE_CUDA_LAUNCH( hypre_SpGemmGhashSize1, gDim, bDim,
                         num_rows, row_id, num_ghash, row_sizes, ghash_i, SHMEM_HASH_SIZE );
   }
   else if (type == 2)
   {
      ghash_i = hypre_CTAlloc(HYPRE_Int, num_ghash + 1, HYPRE_MEMORY_DEVICE);
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);
      HYPRE_CUDA_LAUNCH( hypre_SpGemmGhashSize2, gDim, bDim,
                         num_rows, row_id, num_ghash, row_sizes, ghash_i, SHMEM_HASH_SIZE );
   }

   hypreDevice_IntegerExclusiveScan(num_ghash + 1, ghash_i);

   hypre_TMemcpy(&ghash_size, ghash_i + num_ghash, HYPRE_Int, 1, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);

   if (!ghash_size)
   {
      hypre_TFree(ghash_i, HYPRE_MEMORY_DEVICE);  hypre_assert(ghash_i == NULL);
   }

   if (ghash_i_ptr)
   {
      *ghash_i_ptr = ghash_i;
   }

   if (ghash_j_ptr)
   {
      *ghash_j_ptr = hypre_TAlloc(HYPRE_Int, ghash_size, HYPRE_MEMORY_DEVICE);
   }

   if (ghash_a_ptr)
   {
      *ghash_a_ptr = hypre_TAlloc(HYPRE_Complex, ghash_size, HYPRE_MEMORY_DEVICE);
   }

   if (ghash_size_ptr)
   {
      *ghash_size_ptr = ghash_size;
   }

   return hypre_error_flag;
}

#endif

