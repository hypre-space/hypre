/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CSR_SPGEMM_DEVICE_H
#define CSR_SPGEMM_DEVICE_H

#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#define COHEN_USE_SHMEM 0

static const char HYPRE_SPGEMM_HASH_TYPE = 'D';

/* default settings associated with bin 5 */
#if defined(HYPRE_USING_CUDA)
#define HYPRE_SPGEMM_NUMER_HASH_SIZE 256
#endif
#if defined(HYPRE_USING_HIP)
#define HYPRE_SPGEMM_NUMER_HASH_SIZE 128
#endif
#define HYPRE_SPGEMM_SYMBL_HASH_SIZE 512
#define HYPRE_SPGEMM_BASE_GROUP_SIZE 32
/* unroll factor in the kernels */
#if defined(HYPRE_USING_CUDA)
#define HYPRE_SPGEMM_NUMER_UNROLL 256
#define HYPRE_SPGEMM_SYMBL_UNROLL 512
#elif defined(HYPRE_USING_HIP)
#define HYPRE_SPGEMM_NUMER_UNROLL 256
#define HYPRE_SPGEMM_SYMBL_UNROLL 512
#endif

//#define HYPRE_SPGEMM_TIMING
//#define HYPRE_SPGEMM_PRINTF
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
   if (GROUP_SIZE <= HYPRE_WARP_SIZE)
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

/* group reads 2 values from ptr to v1 and v2
 * GROUP_SIZE must be >= 2
 * lane = GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane : group_lane
 */
template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_read(const HYPRE_Int *ptr, bool valid_ptr, HYPRE_Int &v1, HYPRE_Int &v2, HYPRE_Int lane)
{
   if (GROUP_SIZE >= HYPRE_WARP_SIZE)
   {
      if (lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = __shfl_sync(HYPRE_WARP_FULL_MASK, v1, 1);
      v1 = __shfl_sync(HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      if (valid_ptr && lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = __shfl_sync(HYPRE_WARP_FULL_MASK, v1, 1, GROUP_SIZE);
      v1 = __shfl_sync(HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
   }
}

/* group reads a value from ptr to v1
 * lane = GROUP_SIZE >= HYPRE_WARP_SIZE ? warp_lane : group_lane
 */
template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_read(const HYPRE_Int *ptr, bool valid_ptr, HYPRE_Int &v1, HYPRE_Int lane)
{
   if (GROUP_SIZE >= HYPRE_WARP_SIZE)
   {
      if (!lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = __shfl_sync(HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      if (valid_ptr && !lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = __shfl_sync(HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
   }
}

template <typename T, HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(T in)
{
#if defined(HYPRE_DEBUG)
   hypre_device_assert(GROUP_SIZE <= HYPRE_WARP_SIZE);
#endif

#pragma unroll
   for (hypre_int d = GROUP_SIZE / 2; d > 0; d >>= 1)
   {
      in += __shfl_down_sync(HYPRE_WARP_FULL_MASK, in, d);
   }

   return in;
}

/* s_WarpData[NUM_GROUPS_PER_BLOCK * GROUP_SIZE / HYPRE_WARP_SIZE] */
template <typename T, HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(T in, volatile T *s_WarpData)
{
#if defined(HYPRE_DEBUG)
   hypre_device_assert(GROUP_SIZE > HYPRE_WARP_SIZE);
#endif

   T out = warp_reduce_sum(NULL, in);

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
      out = warp_reduce_sum(NULL, a);
   }

   __syncthreads();

   return out;
}

/* GROUP_SIZE must <= HYPRE_WARP_SIZE */
template <typename T, HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_prefix_sum(hypre_int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (hypre_int d = 2; d <= GROUP_SIZE; d <<= 1)
   {
      T t = __shfl_up_sync(HYPRE_WARP_FULL_MASK, in, d >> 1, GROUP_SIZE);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = __shfl_sync(HYPRE_WARP_FULL_MASK, in, GROUP_SIZE - 1, GROUP_SIZE);

   if (lane_id == GROUP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (hypre_int d = GROUP_SIZE >> 1; d > 0; d >>= 1)
   {
      T t = __shfl_xor_sync(HYPRE_WARP_FULL_MASK, in, d, GROUP_SIZE);

      if ( (lane_id & (d - 1)) == (d - 1))
      {
         if ( (lane_id & ((d << 1) - 1)) == ((d << 1) - 1) )
         {
            in += t;
         }
         else
         {
            in = t;
         }
      }
   }
   return in;
}

template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_sync()
{
   if (GROUP_SIZE <= HYPRE_WARP_SIZE)
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

template <char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int HashFunc(HYPRE_Int m, HYPRE_Int key, HYPRE_Int i, HYPRE_Int prev)
{
   HYPRE_Int hashval = 0;

   /* assume m is power of 2 */
   if (HASHTYPE == 'L')
   {
      //hashval = (key + i) % m;
      hashval = ( prev + 1 ) & (m - 1);
   }
   else if (HASHTYPE == 'Q')
   {
      //hashval = (key + (i + i*i)/2) & (m-1);
      hashval = ( prev + i ) & (m - 1);
   }
   else if (HASHTYPE == 'D')
   {
      //hashval = (key + i*Hash2Func(key) ) & (m - 1);
      hashval = ( prev + Hash2Func(key) ) & (m - 1);
   }

   return hashval;
}

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int HashFunc(HYPRE_Int key, HYPRE_Int i, HYPRE_Int prev)
{
   HYPRE_Int hashval = 0;

   /* assume m is power of 2 */
   if (HASHTYPE == 'L')
   {
      //hashval = (key + i) % SHMEM_HASH_SIZE;
      hashval = ( prev + 1 ) & (SHMEM_HASH_SIZE - 1);
   }
   else if (HASHTYPE == 'Q')
   {
      //hashval = (key + (i + i*i)/2) & (SHMEM_HASH_SIZE-1);
      hashval = ( prev + i ) & (SHMEM_HASH_SIZE - 1);
   }
   else if (HASHTYPE == 'D')
   {
      //hashval = (key + i*Hash2Func(key) ) & (SHMEM_HASH_SIZE - 1);
      hashval = ( prev + Hash2Func(key) ) & (SHMEM_HASH_SIZE - 1);
   }

   return hashval;
}

template<typename T>
struct spgemm_bin_op : public thrust::unary_function<T, char>
{
   char s, t, u; /* s: base size of bins; t: lowest bin; u: highest bin */

   spgemm_bin_op(char s_, char t_, char u_) { s = s_; t = t_; u = u_; }

   __device__ char operator()(const T &x)
   {
      if (x <= 0)
      {
         return 0;
      }

      const T y = (x + s - 1) / s;

      if ( y <= (1 << (t - 1)) )
      {
         return t;
      }

      for (char i = t; i < u - 1; i++)
      {
         if (y <= (1 << i))
         {
            return i + 1;
         }
      }

      return u;
   }
};

void hypre_create_ija(HYPRE_Int m, HYPRE_Int *row_id, HYPRE_Int *d_c, HYPRE_Int *d_i,
                      HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz_ptr );

void hypre_create_ija(HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int m, HYPRE_Int *row_id, HYPRE_Int *d_c,
                      HYPRE_Int *d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz_ptr );

HYPRE_Int hypre_SpGemmCreateGlobalHashTable( HYPRE_Int num_rows, HYPRE_Int *row_id,
                                             HYPRE_Int num_ghash, HYPRE_Int *row_sizes, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int **ghash_i_ptr,
                                             HYPRE_Int **ghash_j_ptr, HYPRE_Complex **ghash_a_ptr, HYPRE_Int *ghash_size_ptr);

HYPRE_Int hypreDevice_CSRSpGemmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                              HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc,
                                              HYPRE_Int row_est_mtd);

HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                                HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc,
                                                HYPRE_Int *d_rc, HYPRE_Int *rownnz_exact_ptr);

HYPRE_Int hypreDevice_CSRSpGemmRownnz(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int nnzA,
                                      HYPRE_Int *d_ia,
                                      HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int in_rc, HYPRE_Int *d_rc);

HYPRE_Int hypreDevice_CSRSpGemmNumerWithRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                                         HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb,
                                                         HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int exact_rownnz, HYPRE_Int **d_ic_out,
                                                         HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

HYPRE_Int hypre_SpGemmCreateBins( HYPRE_Int m, char s, char t, char u, HYPRE_Int *d_rc,
                                  bool d_rc_indice_in, HYPRE_Int *d_rc_indice, HYPRE_Int *h_bin_ptr );

template <HYPRE_Int BIN, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE, bool HAS_RIND>
HYPRE_Int hypre_spgemm_symbolic_rownnz( HYPRE_Int m, HYPRE_Int *row_ind, HYPRE_Int k, HYPRE_Int n,
                                        bool need_ghash, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb,
                                        HYPRE_Int *d_rc, bool can_fail, char *d_rf );

template <HYPRE_Int BIN, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE, bool HAS_RIND>
HYPRE_Int hypre_spgemm_numerical_with_rownnz( HYPRE_Int m, HYPRE_Int *row_ind, HYPRE_Int k,
                                              HYPRE_Int n, bool need_ghash,
                                              HYPRE_Int exact_rownnz, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                                              HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int *d_ic,
                                              HYPRE_Int *d_jc, HYPRE_Complex *d_c );

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int hypre_spgemm_symbolic_max_num_blocks( HYPRE_Int multiProcessorCount,
                                                HYPRE_Int *num_blocks_ptr, HYPRE_Int *block_size_ptr );

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int hypre_spgemm_numerical_max_num_blocks( HYPRE_Int multiProcessorCount,
                                                 HYPRE_Int *num_blocks_ptr, HYPRE_Int *block_size_ptr );

HYPRE_Int hypreDevice_CSRSpGemmBinnedGetBlockNumDim();

template <HYPRE_Int GROUP_SIZE> HYPRE_Int hypreDevice_CSRSpGemmNumerPostCopy( HYPRE_Int m,
                                                                              HYPRE_Int *d_rc, HYPRE_Int *nnzC, HYPRE_Int **d_ic, HYPRE_Int **d_jc, HYPRE_Complex **d_c);

template <HYPRE_Int GROUP_SIZE>
static constexpr HYPRE_Int
hypre_spgemm_get_num_groups_per_block()
{
#if defined(HYPRE_USING_CUDA)
   return hypre_min(hypre_max(512 / GROUP_SIZE, 1), 64);
#elif defined(HYPRE_USING_HIP)
   return hypre_max(512 / GROUP_SIZE, 1);
#endif
}

#if defined(HYPRE_SPGEMM_PRINTF) || defined(HYPRE_SPGEMM_TIMING)
#define HYPRE_SPGEMM_PRINT(...) hypre_ParPrintf(hypre_MPI_COMM_WORLD, __VA_ARGS__)
#else
#define HYPRE_SPGEMM_PRINT(...)
#endif

#endif /* HYPRE_USING_CUDA || defined(HYPRE_USING_HIP) */
#endif /* #ifndef CSR_SPGEMM_DEVICE_H */

