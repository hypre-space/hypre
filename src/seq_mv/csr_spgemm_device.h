/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef CSR_SPGEMM_DEVICE_H
#define CSR_SPGEMM_DEVICE_H

#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)

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
   // return get_thread_id() >> 5;                          // in general cases
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


HYPRE_Int hypreDevice_CSRSpGemmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc);

HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc, HYPRE_Int *d_rf);

HYPRE_Int hypreDevice_CSRSpGemmRownnz(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc);

HYPRE_Int hypreDevice_CSRSpGemmWithRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int exact_rownnz, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

HYPRE_Int hypreDevice_CSRSpGemmWithRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

HYPRE_Int hypreDevice_CSRSpGemmCusparse(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int nnzB, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out);

void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz);

void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_c, HYPRE_Int **d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz);

HYPRE_Int csr_spmm_create_hash_table(HYPRE_Int m, HYPRE_Int *d_rc, HYPRE_Int *d_rf, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int num_ghash, HYPRE_Int **d_ghash_i, HYPRE_Int **d_ghash_j, HYPRE_Complex **d_ghash_a, HYPRE_Int *ghash_size);

//Used in tripmat product
template <char type>
static __device__ __forceinline__
void rownnz_naive_rowi(HYPRE_Int rowi, HYPRE_Int lane_id, HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Int *ib,
                       HYPRE_Int &row_nnz_sum, HYPRE_Int &row_nnz_max);

template <char type>
static __device__ __forceinline__
void rownnz_naive_rowi(HYPRE_Int rowi, HYPRE_Int lane_id, HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Int *ib,
                       HYPRE_Int &row_nnz_sum, HYPRE_Int &row_nnz_max)
{
   /* load the start and end position of row i of A */
   HYPRE_Int j = -1;
   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }
   const HYPRE_Int istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   const HYPRE_Int iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);

   row_nnz_sum = 0;
   row_nnz_max = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart; i < iend; i += HYPRE_WARP_SIZE)
   {
      if (i + lane_id < iend)
      {
         HYPRE_Int colA = read_only_load(ja + i + lane_id);
         HYPRE_Int rowB_start = read_only_load(ib+colA);
         HYPRE_Int rowB_end   = read_only_load(ib+colA+1);
         if (type == 'U' || type == 'B')
         {
            row_nnz_sum += rowB_end - rowB_start;
         }
         if (type == 'L' || type == 'B')
         {
            row_nnz_max = max(row_nnz_max, rowB_end - rowB_start);
         }
      }
   }
}

__global__
void expdistfromuniform(HYPRE_Int n, float *x);


template <typename T, HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_SIZE_PER_WARP, HYPRE_Int layer>
__global__
void cohen_rowest_kernel(HYPRE_Int nrow, HYPRE_Int *rowptr, HYPRE_Int *colidx, T *V_in, T *V_out,
                         HYPRE_Int *rc, HYPRE_Int nsamples, HYPRE_Int *low, HYPRE_Int *upp, T mult);
//Tripmat products
HYPRE_Int hypreDevice_CSRSpGemmmRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int r, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_ic, HYPRE_Int *d_jc, HYPRE_Int *d_rd, HYPRE_Int *d_rf);

HYPRE_Int hypreDevice_CSRSpGemmmWithRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int r, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_ic, HYPRE_Int *d_jc, HYPRE_Complex *d_c, HYPRE_Int *d_rd, HYPRE_Int exact_rownnz, HYPRE_Int **d_id_out, HYPRE_Int **d_jd_out, HYPRE_Complex **d_d_out, HYPRE_Int *nnzD);

HYPRE_Int
hypreDevice_CSRSpGemmmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, 
    HYPRE_Int r, HYPRE_Int n,
    HYPRE_Int *d_ia, HYPRE_Int *d_ja, 
    HYPRE_Int *d_ib, HYPRE_Int *d_jb, 
    HYPRE_Int *d_ic, HYPRE_Int *d_jc, 
    HYPRE_Int *d_rc);

#endif /* HYPRE_USING_CUDA */
#endif

