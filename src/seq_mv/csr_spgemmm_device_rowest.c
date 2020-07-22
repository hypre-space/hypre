/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                Row size estimations
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */

#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(HYPRE_USING_CUDA)

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                       NAIVE
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
/*
 * @brief Calculates naive bounds for the ith row of D=A*B*C
 */
template <char type>
static __device__ __forceinline__
void rownnz_naive_rowi_triple(HYPRE_Int rowi, HYPRE_Int lane_id, 
    HYPRE_Int *ia, HYPRE_Int *ja, 
    HYPRE_Int *ib, HYPRE_Int *jb,
    HYPRE_Int *ic,
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
         HYPRE_Int rowB_nnz_min, rowB_nnz_max;
         rownnz_naive_rowi<type>(colA, lane_id, ib, jb, ic, rowB_nnz_min, rowB_nnz_max);
         // printf("Row %i min: %i, max: %i \n", rowi, rowB_nnz_min, rowB_nnz_max);
         if (type == 'U' || type == 'B')
         {
            row_nnz_sum += rowB_nnz_max;
         }
         if (type == 'L' || type == 'B')
         {
            row_nnz_max = max(row_nnz_max, rowB_nnz_min);
         }
         // printf("Row %i sum: %i, max: %i \n", rowi, row_nnz_sum, row_nnz_max);
      }
   }
}

template <char type, HYPRE_Int NUM_WARPS_PER_BLOCK>
__global__
void csr_spmmm_rownnz_naive(HYPRE_Int M, /*HYPRE_Int K,*/ HYPRE_Int N, 
    HYPRE_Int *ia, HYPRE_Int *ja,
    HYPRE_Int *ib, HYPRE_Int *jb, 
    HYPRE_Int *ic, HYPRE_Int *jc, 
    HYPRE_Int *rcL, HYPRE_Int *rcU)
{
   const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim.x;
   /* warp id inside the block */
   const HYPRE_Int warp_id = get_warp_id();
   /* lane id inside the warp */
   volatile const HYPRE_Int lane_id = get_lane_id();

#ifdef HYPRE_DEBUG
   assert(blockDim.x * blockDim.y == HYPRE_WARP_SIZE);
#endif

   for (HYPRE_Int i = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
            i < M;
            i += num_warps)
   {
      HYPRE_Int jU, jL;

      rownnz_naive_rowi_triple<type>(i, lane_id, ia, ja, ib, jb, ic, jU, jL);

      if (type == 'U' || type == 'B')
      {
         jU = warp_reduce_sum(jU);
         jU = min(jU, N);
      }

      if (type == 'L' || type == 'B')
      {
         jL = warp_reduce_max(jL);
      }

      if (lane_id == 0)
      {
         if (type == 'L' || type == 'B')
         {
            rcL[i] = jL;
         }

         if (type == 'U' || type == 'B')
         {
            rcU[i] = jU;
         }
      }
   }
}



template <typename T, HYPRE_Int BDIMX, HYPRE_Int BDIMY, HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_SIZE_PER_WARP>
void csr_spmmm_rownnz_cohen(HYPRE_Int M, HYPRE_Int K, HYPRE_Int R, HYPRE_Int N, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_ic, HYPRE_Int *d_jc, HYPRE_Int *d_low, HYPRE_Int *d_upp, HYPRE_Int *d_rc, HYPRE_Int nsamples, T mult_factor, T *work)
{
   dim3 bDim(BDIMX, BDIMY, NUM_WARPS_PER_BLOCK);
   hypre_assert(bDim.x * bDim.y == HYPRE_WARP_SIZE);

   T *d_V1, *d_V2, *d_V3, *d_V4;

   d_V1 = work;
   d_V2 = d_V1 + nsamples*N;
   d_V3 = d_V2 + nsamples*R;
   //d_V1 = hypre_TAlloc(T, nsamples*N, HYPRE_MEMORY_DEVICE);
   //d_V2 = hypre_TAlloc(T, nsamples*K, HYPRE_MEMORY_DEVICE);

   curandGenerator_t gen = hypre_HandleCurandGenerator(hypre_handle());
   //CURAND_CALL(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_SEEDED));
   /* random V1: uniform --> exp */
   HYPRE_CURAND_CALL(curandGenerateUniform(gen, d_V1, nsamples * N));
   //  CURAND_CALL(curandGenerateUniformDouble(gen, d_V1, nsamples * N));
   dim3 gDim( (nsamples * N + bDim.z * HYPRE_WARP_SIZE - 1) / (bDim.z * HYPRE_WARP_SIZE) );

   HYPRE_CUDA_LAUNCH( expdistfromuniform, gDim, bDim, nsamples * N, d_V1 );

   /* step-1: layer 4-3 */
   gDim.x = (R + bDim.z - 1) / bDim.z;
   HYPRE_CUDA_LAUNCH( (cohen_rowest_kernel<T, NUM_WARPS_PER_BLOCK, SHMEM_SIZE_PER_WARP, 2>), gDim, bDim,
                      R, d_ic, d_jc, d_V1, d_V2, NULL, nsamples, NULL, NULL, -1.0);

   /* step-2: layer 3-2 */
   gDim.x = (K + bDim.z - 1) / bDim.z;
   HYPRE_CUDA_LAUNCH( (cohen_rowest_kernel<T, NUM_WARPS_PER_BLOCK, SHMEM_SIZE_PER_WARP, 2>), gDim, bDim,
                      K, d_ib, d_jb, d_V2, d_V3, NULL, nsamples, NULL, NULL, -1.0);

   //hypre_TFree(d_V1, HYPRE_MEMORY_DEVICE);

   /* step-3: layer 2-1 */
   d_V4 = (T*) d_rc;


   gDim.x = (M + bDim.z - 1) / bDim.z;
   HYPRE_CUDA_LAUNCH( (cohen_rowest_kernel<T, NUM_WARPS_PER_BLOCK, SHMEM_SIZE_PER_WARP, 1>), gDim, bDim,
                      M, d_ia, d_ja, d_V3, d_V4, d_rc, nsamples, d_low, d_upp, mult_factor);

   /* done */
   //hypre_TFree(d_V2, HYPRE_MEMORY_DEVICE);
}


HYPRE_Int
hypreDevice_CSRSpGemmmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, 
    HYPRE_Int r, HYPRE_Int n,
    HYPRE_Int *d_ia, HYPRE_Int *d_ja, 
    HYPRE_Int *d_ib, HYPRE_Int *d_jb, 
    HYPRE_Int *d_ic, HYPRE_Int *d_jc, 
    HYPRE_Int *d_rc)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_ROWNNZ] -= hypre_MPI_Wtime();
#endif

   const HYPRE_Int num_warps_per_block =  16;
   const HYPRE_Int shmem_size_per_warp = 128;
   const HYPRE_Int BDIMX               =   2;
   const HYPRE_Int BDIMY               =  16;

   /* CUDA kernel configurations */
   dim3 bDim(BDIMX, BDIMY, num_warps_per_block);
   hypre_assert(bDim.x * bDim.y == HYPRE_WARP_SIZE);
   // for cases where one WARP works on a row
   dim3 gDim( (m + bDim.z - 1) / bDim.z );

   HYPRE_Int   row_est_mtd    = hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle());
   HYPRE_Int   cohen_nsamples = hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle());
   float cohen_mult           = hypre_HandleSpgemmRownnzEstimateMultFactor(hypre_handle());

   if (row_est_mtd == 1)
   {
      /* naive overestimate */
      HYPRE_CUDA_LAUNCH( (csr_spmmm_rownnz_naive<'U', num_warps_per_block>), gDim, bDim,
                         m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_ic, d_jc, NULL, d_rc );
   }
   else if (row_est_mtd == 2)
   {
      /* naive underestimate */
      HYPRE_CUDA_LAUNCH( (csr_spmmm_rownnz_naive<'L', num_warps_per_block>), gDim, bDim,
                         m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_ic, d_jc, d_rc, NULL );
   }
   else if (row_est_mtd == 3)
   {
      /* [optional] first run naive estimate for naive lower and upper bounds,
                    which will be given to Cohen's alg as corrections */
      char *work_mem = hypre_TAlloc(char, cohen_nsamples*(n+k+r)*sizeof(float)+2*m*sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
      char *work_mem_saved = work_mem;

      //HYPRE_Int *d_low_upp = hypre_TAlloc(HYPRE_Int, 2 * m, HYPRE_MEMORY_DEVICE);
      //HYPRE_Int *d_low_upp = (HYPRE_Int *) work_mem;
      work_mem += 2*m*sizeof(HYPRE_Int);

      /* Avoid doing the naive bounds, as the current method does not work well */
      // HYPRE_Int *d_low = d_low_upp;
      // HYPRE_Int *d_upp = d_low_upp + m;

      // HYPRE_CUDA_LAUNCH( (csr_spmmm_rownnz_naive<'B', num_warps_per_block>), gDim, bDim,
      //                    m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_ic, d_jc, d_low, d_upp );

      /* Cohen's algorithm, stochastic approach */
      csr_spmmm_rownnz_cohen<float, BDIMX, BDIMY, num_warps_per_block, shmem_size_per_warp>
         (m, k, r, n, d_ia, d_ja, d_ib, d_jb, d_ic, d_jc, NULL, NULL, d_rc, cohen_nsamples, cohen_mult, (float *)work_mem);

      //hypre_TFree(d_low_upp, HYPRE_MEMORY_DEVICE);
      hypre_TFree(work_mem_saved, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      printf("Unknown row nnz estimation method %d! \n", row_est_mtd);
      exit(-1);
   }

#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_ROWNNZ] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA */

