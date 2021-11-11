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

#if defined(HYPRE_USING_CUDA)
#define HYPRE_SPGEMM_NUMER_HASH_SIZE 256
#define HYPRE_SPGEMM_SYMBL_HASH_SIZE 512
#elif defined(HYPRE_USING_HIP)
#define HYPRE_SPGEMM_NUMER_HASH_SIZE 256
#define HYPRE_SPGEMM_SYMBL_HASH_SIZE 512
#endif

#define HYPRE_SPGEMM_TIMING

//#define HYPRE_SPGEMM_NVTX

/* ----------------------------------------------------------------------------------------------- *
 * these are under the assumptions made in spgemm on block sizes: only use in csr_spgemm routines
 * where we assume CUDA block is 3D and blockDim.x * blockDim.y = GROUP_SIZE
 *------------------------------------------------------------------------------------------------ */

/* the number of threads in the block */
static __device__ __forceinline__
hypre_int get_block_size()
{
   return (blockDim.x * blockDim.y * blockDim.z);
}

/* the thread id in the block */
static __device__ __forceinline__
hypre_int get_thread_id()
{
   return (threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x);
}

/* the number of groups in block */
static __device__ __forceinline__
hypre_int get_num_groups()
{
   return blockDim.z;
}

/* the group id in the block */
static __device__ __forceinline__
hypre_int get_group_id()
{
   return threadIdx.z;
}

/* the thread id (lane) in the group */
static __device__ __forceinline__
hypre_int get_lane_id()
{
   return threadIdx.y * blockDim.x + threadIdx.x;
}

/* the warp id in the block */
static __device__ __forceinline__
hypre_int get_warp_id()
{
   return get_thread_id() >> HYPRE_WARP_BITSHIFT;
}

/* the warp id in the group */
template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
hypre_int get_warp_in_group_id()
{
   if (GROUP_SIZE == HYPRE_WARP_SIZE)
   {
      return 0;
   }
   else
   {
      return get_lane_id() >> HYPRE_WARP_BITSHIFT;
   }
}

/* return the thread id (lane) in warp */
static __device__ __forceinline__
hypre_int get_warp_lane_id()
{
   return get_lane_id() & (HYPRE_WARP_SIZE - 1);
}

template <typename T, HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(T in)
{
   T out = warp_reduce_sum(in);

   if (GROUP_SIZE != HYPRE_WARP_SIZE)
   {
      __shared__ volatile HYPRE_Int s_WarpData[NUM_GROUPS_PER_BLOCK * GROUP_SIZE / HYPRE_WARP_SIZE];
      const HYPRE_Int warp_lane_id = get_warp_lane_id();
      const HYPRE_Int warp_id = get_warp_id();

      if (warp_lane_id == 0)
      {
         s_WarpData[warp_id] = out;
      }

      __syncthreads();

      if (get_warp_in_group_id<GROUP_SIZE>() == 0)
      {
         const T a = warp_lane_id < GROUP_SIZE / HYPRE_WARP_SIZE ? s_WarpData[warp_id + warp_lane_id] : 0.0;
         out = warp_reduce_sum(a);
      }

      __syncthreads();
   }

   return out;
}

template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_sync()
{
   if (GROUP_SIZE == HYPRE_WARP_SIZE)
   {
      __syncwarp();
   }
   else
   {
      __syncthreads();
   }
}

/* Hash functions */
static __device__ __forceinline__
HYPRE_Int Hash2Func(HYPRE_Int key)
{
   //return ( (key << 1) | 1 );
   //TODO: 6 --> should depend on hash1 size
   return ( (key >> 6) | 1 );
}

static __device__ __forceinline__
HYPRE_Int HashFunc(char type, HYPRE_Int m, HYPRE_Int key, HYPRE_Int i, HYPRE_Int prev)
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

void hypre_create_ija(HYPRE_Int type, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int m, HYPRE_Int *row_id, HYPRE_Int *d_c, HYPRE_Int *d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz_ptr );

HYPRE_Int hypre_SpGemmCreateGlobalHashTable( HYPRE_Int num_rows, HYPRE_Int *row_id, HYPRE_Int num_ghash, HYPRE_Int *row_sizes, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int **ghash_i_ptr, HYPRE_Int **ghash_j_ptr, HYPRE_Complex **ghash_a_ptr, HYPRE_Int *ghash_size_ptr);

HYPRE_Int hypreDevice_CSRSpGemmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc);

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc, HYPRE_Int *d_rf);

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int hypreDevice_CSRSpGemmRownnz(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc);

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int hypreDevice_CSRSpGemmNumerWithRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int exact_rownnz, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

HYPRE_Int hypreDevice_CSRSpGemmNumerWithRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

#endif /* HYPRE_USING_CUDA || defined(HYPRE_USING_HIP) */
#endif /* #ifndef CSR_SPGEMM_DEVICE_H */

