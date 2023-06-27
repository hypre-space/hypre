/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_GPU)

#if defined(HYPRE_USING_SYCL)
struct row_size
#else
struct row_size : public thrust::unary_function<HYPRE_Int, HYPRE_Int>
#endif
{
   HYPRE_Int SHMEM_HASH_SIZE;

   row_size(HYPRE_Int SHMEM_HASH_SIZE_ = HYPRE_Int()) { SHMEM_HASH_SIZE = SHMEM_HASH_SIZE_; }

   __device__ HYPRE_Int operator()(const HYPRE_Int &x) const
   {
      // RL: ???
      return next_power_of_2(x - SHMEM_HASH_SIZE) + x;
   }
};

/* Assume d_c is of length m and contains the size of each row
 *        d_i has size (m+1) on entry
 * generate (i,j,a) with d_c */
void
hypre_create_ija( HYPRE_Int       m,
                  HYPRE_Int      *row_id, /* length of m, row indices; if null, it is [0,1,2,3,...] */
                  HYPRE_Int      *d_c,    /* d_c[row_id[i]] is the size of ith row */
                  HYPRE_Int      *d_i,
                  HYPRE_Int     **d_j,
                  HYPRE_Complex **d_a,
                  HYPRE_Int
                  *nnz_ptr /* in/out: if input >= 0, it must be the sum of d_c, remain unchanged in output
                                                     if input <  0, it is computed as the sum of d_c and output */)
{
   HYPRE_Int nnz = 0;

   hypre_Memset(d_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   if (row_id)
   {
      HYPRE_ONEDPL_CALL(std::inclusive_scan,
                        oneapi::dpl::make_permutation_iterator(d_c, row_id),
                        oneapi::dpl::make_permutation_iterator(d_c, row_id) + m,
                        d_i + 1);
   }
   else
   {
      HYPRE_ONEDPL_CALL(std::inclusive_scan,
                        d_c,
                        d_c + m,
                        d_i + 1);
   }
#else
   if (row_id)
   {
      HYPRE_THRUST_CALL(inclusive_scan,
                        thrust::make_permutation_iterator(d_c, row_id),
                        thrust::make_permutation_iterator(d_c, row_id) + m,
                        d_i + 1);
   }
   else
   {
      HYPRE_THRUST_CALL(inclusive_scan,
                        d_c,
                        d_c + m,
                        d_i + 1);
   }
#endif

   if (*nnz_ptr >= 0)
   {
#if defined(HYPRE_DEBUG)
      hypre_TMemcpy(&nnz, d_i + m, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_assert(nnz == *nnz_ptr);
#endif
      nnz = *nnz_ptr;
   }
   else
   {
      hypre_TMemcpy(&nnz, d_i + m, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      *nnz_ptr = nnz;
   }

   if (d_j)
   {
      *d_j = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
   }

   if (d_a)
   {
      *d_a = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);
   }
}

/* Assume d_c is of length m and contains the size of each row
 *        d_i has size (m+1) on entry
 * generate (i,j,a) with row_size(d_c) see above (over allocation) */
void
hypre_create_ija( HYPRE_Int       SHMEM_HASH_SIZE,
                  HYPRE_Int       m,
                  HYPRE_Int      *row_id,        /* length of m, row indices; if null, it is [0,1,2,3,...] */
                  HYPRE_Int      *d_c,           /* d_c[row_id[i]] is the size of ith row */
                  HYPRE_Int      *d_i,
                  HYPRE_Int     **d_j,
                  HYPRE_Complex **d_a,
                  HYPRE_Int      *nnz_ptr )
{
   HYPRE_Int nnz = 0;

   hypre_Memset(d_i, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   if (row_id)
   {
      HYPRE_ONEDPL_CALL( std::inclusive_scan,
                         oneapi::dpl::make_transform_iterator(oneapi::dpl::make_permutation_iterator(d_c, row_id),
                                                              row_size(SHMEM_HASH_SIZE)),
                         oneapi::dpl::make_transform_iterator(oneapi::dpl::make_permutation_iterator(d_c, row_id),
                                                              row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
   else
   {
      HYPRE_ONEDPL_CALL( std::inclusive_scan,
                         oneapi::dpl::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)),
                         oneapi::dpl::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
#else
   if (row_id)
   {
      HYPRE_THRUST_CALL( inclusive_scan,
                         thrust::make_transform_iterator(thrust::make_permutation_iterator(d_c, row_id),
                                                         row_size(SHMEM_HASH_SIZE)),
                         thrust::make_transform_iterator(thrust::make_permutation_iterator(d_c, row_id),
                                                         row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
   else
   {
      HYPRE_THRUST_CALL( inclusive_scan,
                         thrust::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)),
                         thrust::make_transform_iterator(d_c, row_size(SHMEM_HASH_SIZE)) + m,
                         d_i + 1 );
   }
#endif

   hypre_TMemcpy(&nnz, d_i + m, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   if (nnz_ptr)
   {
      *nnz_ptr = nnz;
   }

   if (d_j)
   {
      *d_j = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
   }

   if (d_a)
   {
      *d_a = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);
   }
}

__global__ void
hypre_SpGemmGhashSize( hypre_DeviceItem &item,
                       HYPRE_Int  num_rows,
                       HYPRE_Int *row_id,
                       HYPRE_Int  num_ghash,
                       HYPRE_Int *row_sizes,
                       HYPRE_Int *ghash_sizes,
                       HYPRE_Int  SHMEM_HASH_SIZE )
{
   const HYPRE_Int global_thread_id = hypre_gpu_get_grid_thread_id<1, 1>(item);

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

HYPRE_Int
hypre_SpGemmCreateGlobalHashTable( HYPRE_Int       num_rows,        /* number of rows */
                                   HYPRE_Int      *row_id,          /* row_id[i] is index of ith row; i if row_id == NULL */
                                   HYPRE_Int       num_ghash,       /* number of hash tables <= num_rows */
                                   HYPRE_Int      *row_sizes,       /* row_sizes[rowid[i]] is the size of ith row */
                                   HYPRE_Int       SHMEM_HASH_SIZE,
                                   HYPRE_Int     **ghash_i_ptr,     /* of length num_ghash + 1 */
                                   HYPRE_Int     **ghash_j_ptr,
                                   HYPRE_Complex **ghash_a_ptr,
                                   HYPRE_Int      *ghash_size_ptr )
{
   hypre_assert(num_ghash <= num_rows);

   HYPRE_Int *ghash_i, ghash_size;
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();

   ghash_i = hypre_TAlloc(HYPRE_Int, num_ghash + 1, HYPRE_MEMORY_DEVICE);
   hypre_Memset(ghash_i + num_ghash, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_ghash, "thread", bDim);
   HYPRE_GPU_LAUNCH( hypre_SpGemmGhashSize, gDim, bDim,
                     num_rows, row_id, num_ghash, row_sizes, ghash_i, SHMEM_HASH_SIZE );

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

HYPRE_Int hypre_SpGemmCreateBins( HYPRE_Int  m,
                                  char       s,
                                  char       t,
                                  char       u,
                                  HYPRE_Int *d_rc,
                                  bool       d_rc_indice_in,
                                  HYPRE_Int *d_rc_indice,
                                  HYPRE_Int *h_bin_ptr )
{
#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   HYPRE_Real t1 = hypre_MPI_Wtime();
#endif

   HYPRE_Int  num_bins = hypre_HandleSpgemmNumBin(hypre_handle());
   HYPRE_Int *d_bin_ptr = hypre_TAlloc(HYPRE_Int, num_bins + 1, HYPRE_MEMORY_DEVICE);

   /* assume there are no more than 127 = 2^7-1 bins, which should be enough */
   char *d_bin_key = hypre_TAlloc(char, m, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      d_rc,
                      d_rc + m,
                      d_bin_key,
                      spgemm_bin_op<HYPRE_Int>(s, t, u) );

   if (!d_rc_indice_in)
   {
      hypreSycl_sequence(d_rc_indice, d_rc_indice + m, 0);
   }

   hypreSycl_stable_sort_by_key(d_bin_key, d_bin_key + m, d_rc_indice);

   HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                      d_bin_key,
                      d_bin_key + m,
                      oneapi::dpl::counting_iterator<HYPRE_Int>(1),
                      oneapi::dpl::counting_iterator<HYPRE_Int>(num_bins + 2),
                      d_bin_ptr );
#else
   HYPRE_THRUST_CALL( transform,
                      d_rc,
                      d_rc + m,
                      d_bin_key,
                      spgemm_bin_op<HYPRE_Int>(s, t, u) );

   if (!d_rc_indice_in)
   {
      HYPRE_THRUST_CALL( sequence, d_rc_indice, d_rc_indice + m);
   }

   HYPRE_THRUST_CALL( stable_sort_by_key, d_bin_key, d_bin_key + m, d_rc_indice );

   HYPRE_THRUST_CALL( lower_bound,
                      d_bin_key,
                      d_bin_key + m,
                      thrust::make_counting_iterator(1),
                      thrust::make_counting_iterator(num_bins + 2),
                      d_bin_ptr );
#endif

   hypre_TMemcpy(h_bin_ptr, d_bin_ptr, HYPRE_Int, num_bins + 1, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);

   hypre_assert(h_bin_ptr[num_bins] == m);

   hypre_TFree(d_bin_key, HYPRE_MEMORY_DEVICE);
   hypre_TFree(d_bin_ptr, HYPRE_MEMORY_DEVICE);

#ifdef HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   HYPRE_Real t2 = hypre_MPI_Wtime() - t1;
   HYPRE_SPGEMM_PRINT("%s[%d]: Binning time %f\n", __FILE__, __LINE__, t2);
#endif

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_GPU)

