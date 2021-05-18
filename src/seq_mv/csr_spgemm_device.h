/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CSR_SPGEMM_DEVICE_H
#define CSR_SPGEMM_DEVICE_H

#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#define COHEN_USE_SHMEM 0

/* these are under the assumptions made in spgemm on block sizes: only use in spmm routines */
static __device__ __forceinline__
hypre_int get_block_size()
{
   //return (blockDim.x * blockDim.y * blockDim.z);           // in general cases
   return (HYPRE_WARP_SIZE * blockDim.z);                           // if blockDim.x * blockDim.y = WARP_SIZE
}

static __device__ __forceinline__
hypre_int get_thread_id()
{
   //return (threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x); // in general cases
   return (threadIdx.z * HYPRE_WARP_SIZE + threadIdx.y * blockDim.x + threadIdx.x);                 // if blockDim.x * blockDim.y = WARP_SIZE
}

static __device__ __forceinline__
hypre_int get_warp_id()
{
   // return get_thread_id() >> HYPRE_WARP_BITSHIFT;        // in general cases
   return threadIdx.z;                                      // if blockDim.x * blockDim.y = WARP_SIZE
}

static __device__ __forceinline__
hypre_int get_lane_id()
{
   // return get_thread_id() & (WARP_SIZE-1);               // in general cases
   return threadIdx.y * blockDim.x + threadIdx.x;           // if blockDim.x * blockDim.y = WARP_SIZE
}


/* Hash functions */
static __device__ __forceinline__
HYPRE_Int Hash2Func(HYPRE_Int key)
{
   //return ( (key << 1) | 1 );
   //TODO: 6 --> should depend on hash1 size
   return ( (key >> 6) | 1 );
}

template <char type>
static __device__ __forceinline__
HYPRE_Int HashFunc(HYPRE_Int m, HYPRE_Int key, HYPRE_Int i, HYPRE_Int prev)
{
   HYPRE_Int hashval = 0;

   /* assume m is power of 2 */
   if (type == 'L')
   {
      //hashval = (key + i) % m;
      hashval = ( prev + 1 ) & (m - 1);
   }
   else if (type == 'Q')
   {
      //hashval = (key + (i + i*i)/2) & (m-1);
      hashval = ( prev + i ) & (m - 1);
   }
   else if (type == 'D')
   {
      //hashval = (key + i*Hash2Func(key) ) & (m - 1);
      hashval = ( prev + Hash2Func(key) ) & (m - 1);
   }

   return hashval;
}

void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz);

void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_c, HYPRE_Int **d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz);

HYPRE_Int csr_spmm_create_hash_table(HYPRE_Int m, HYPRE_Int *d_rc, HYPRE_Int *d_rf, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int num_ghash, HYPRE_Int **d_ghash_i, HYPRE_Int **d_ghash_j, HYPRE_Complex **d_ghash_a, HYPRE_Int *ghash_size);


#endif /* HYPRE_USING_CUDA || defined(HYPRE_USING_HIP) */
#endif
