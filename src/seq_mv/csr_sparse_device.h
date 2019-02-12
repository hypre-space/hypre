/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef CSR_SPARSE_DEVICE_H
#define CSR_SPARSE_DEVICE_H

#if defined(HYPRE_USING_CUDA)

#include <curand.h>

#define COHEN_USE_SHMEM 0
#define DEBUG_MODE      1

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(1);}} while(0)

typedef struct
{
   HYPRE_Int   rownnz_estimate_method;
   HYPRE_Int   rownnz_estimate_nsamples;
   float       rownnz_estimate_mult_factor;
   char        hash_type;
   HYPRE_Int   do_timing;
} hypre_DeviceCSRSparseOpts;

typedef struct
{
   size_t      ghash_size, ghash2_size;
   HYPRE_Int   nnzC_gpu;

   HYPRE_Real  rownnz_estimate_time;
   HYPRE_Real  rownnz_estimate_curand_time;
   size_t      rownnz_estimate_mem;

   HYPRE_Real  spmm_create_hashtable_time;

   HYPRE_Real  spmm_attempt1_time;
   HYPRE_Real  spmm_post_attempt1_time;
   HYPRE_Real  spmm_attempt2_time;
   HYPRE_Real  spmm_post_attempt2_time;
   size_t      spmm_attempt_mem;

   HYPRE_Real  spmm_symbolic_time;
   size_t      spmm_symbolic_mem;
   HYPRE_Real  spmm_post_symbolic_time;
   HYPRE_Real  spmm_numeric_time;
   size_t      spmm_numeric_mem;
   HYPRE_Real  spmm_post_numeric_time;

   HYPRE_Real  spadd_expansion_time;
   HYPRE_Real  spadd_sorting_time;
   HYPRE_Real  spadd_compression_time;
   HYPRE_Real  spadd_convert_ptr_time;
   HYPRE_Real  spadd_time;

   HYPRE_Real  sptrans_expansion_time;
   HYPRE_Real  sptrans_sorting_time;
   HYPRE_Real  sptrans_rowptr_time;
   HYPRE_Real  sptrans_time;

} hypre_DeviceCSRSparseHandle;

extern hypre_DeviceCSRSparseOpts   *hypre_device_sparse_opts;
extern hypre_DeviceCSRSparseHandle *hypre_device_sparse_handle;

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

HYPRE_Int hypreDevice_CSRSpGemm(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC);

HYPRE_Int hypreDevice_CSRSpGemmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc, hypre_DeviceCSRSparseOpts *opts, hypre_DeviceCSRSparseHandle *handle);

HYPRE_Int hypreDevice_CSRSpGemmRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc, HYPRE_Int *d_rf, hypre_DeviceCSRSparseOpts *opts, hypre_DeviceCSRSparseHandle *handle);

HYPRE_Int hypreDevice_CSRSpGemmRownnz(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc, hypre_DeviceCSRSparseOpts *opts, hypre_DeviceCSRSparseHandle *handle);

HYPRE_Int hypreDevice_CSRSpGemmWithRownnzUpperbound(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_b, HYPRE_Int *d_rc, HYPRE_Int exact_rownnz, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_c_out, HYPRE_Int *nnzC, hypre_DeviceCSRSparseOpts *opts, hypre_DeviceCSRSparseHandle *handle);

void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz);

void csr_spmm_create_ija(HYPRE_Int m, HYPRE_Int *d_c, HYPRE_Int **d_i, HYPRE_Int **d_j, HYPRE_Complex **d_a, HYPRE_Int *nnz);

HYPRE_Int csr_spmm_create_hash_table(HYPRE_Int m, HYPRE_Int *d_rc, HYPRE_Int *d_rf, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int num_ghash, HYPRE_Int **d_ghash_i, HYPRE_Int **d_ghash_j, HYPRE_Complex **d_ghash_a, HYPRE_Int *ghash_size);

HYPRE_Int hypreDevice_CSRSpAdd(HYPRE_Int ma, HYPRE_Int mb, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int nnzB, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Complex *d_ab, HYPRE_Int *d_num_b, HYPRE_Int *nnzC_out, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out);

HYPRE_Int hypreDevice_CSRSpTrans(HYPRE_Int m, HYPRE_Int n, HYPRE_Int nnzA, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_aa, HYPRE_Int **d_ic_out, HYPRE_Int **d_jc_out, HYPRE_Complex **d_ac_out, HYPRE_Int want_data);

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

#endif /* HYPRE_USING_CUDA */
#endif

