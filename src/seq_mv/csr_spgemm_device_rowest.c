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

#if defined(HYPRE_USING_GPU)

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                       NAIVE
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
template <char type>
static __device__ __forceinline__
void rownnz_naive_rowi(HYPRE_Int rowi, HYPRE_Int lane_id, HYPRE_Int *ia, HYPRE_Int *ja,
                       HYPRE_Int *ib,
                       HYPRE_Int &row_nnz_sum, HYPRE_Int &row_nnz_max
#ifdef HYPRE_USING_SYCL
                       , sycl::nd_item<3>& item
#endif
                      )
{
   /* load the start and end position of row i of A */
   HYPRE_Int j = -1;
   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int warp_size = SG.get_local_range().get(0);
   const HYPRE_Int istart = SG.shuffle(j, 0);
   const HYPRE_Int iend   = SG.shuffle(j, 1);
#else
   HYPRE_Int warp_size = HYPRE_WARP_SIZE;
   const HYPRE_Int istart = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0);
   const HYPRE_Int iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1);
#endif

   row_nnz_sum = 0;
   row_nnz_max = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart; i < iend; i += warp_size)
   {
      if (i + lane_id < iend)
      {
         HYPRE_Int colA = read_only_load(ja + i + lane_id);
         HYPRE_Int rowB_start = read_only_load(ib + colA);
         HYPRE_Int rowB_end   = read_only_load(ib + colA + 1);
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

template <char type, HYPRE_Int NUM_WARPS_PER_BLOCK>
__global__
void csr_spmm_rownnz_naive(
#ifdef HYPRE_USING_SYCL
   sycl::nd_item<3>& item,
#endif
   HYPRE_Int M, /*HYPRE_Int K,*/ HYPRE_Int N, HYPRE_Int *ia, HYPRE_Int *ja,
   HYPRE_Int *ib, HYPRE_Int *jb, HYPRE_Int *rcL, HYPRE_Int *rcU)
{
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int warp_size = SG.get_local_range().get(0);
   HYPRE_Int blockDim_x = item.get_local_range(2);
   HYPRE_Int blockDim_y = item.get_local_range(1);
   HYPRE_Int gridDim_x  = item.get_group_range(2);
   HYPRE_Int blockIdx_x = item.get_group(2);
   /* warp id inside the block */
   const HYPRE_Int warp_id = get_warp_id(item);
   /* lane id inside the warp */
   volatile const HYPRE_Int lane_id = get_lane_id(item);
#else
   HYPRE_Int warp_size   = HYPRE_WARP_SIZE;
   HYPRE_Int blockDim_x  = blockDim.x;
   HYPRE_Int blockDim_y  = blockDim.y;
   HYPRE_Int gridDim_x   = gridDim.x;
   HYPRE_Int blockIdx_x  = blockIdx.x;
   /* warp id inside the block */
   const HYPRE_Int warp_id = get_warp_id();
   /* lane id inside the warp */
   volatile const HYPRE_Int lane_id = get_lane_id();
#endif

   const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim_x;
   hypre_device_assert(blockDim_x * blockDim_y == warp_size);

   for (HYPRE_Int i = blockIdx_x * NUM_WARPS_PER_BLOCK + warp_id;
        i < M;
        i += num_warps)
   {
      HYPRE_Int jU, jL;

      rownnz_naive_rowi<type>(i, lane_id, ia, ja, ib, jU, jL
#ifdef HYPRE_USING_SYCL
                              , item
#endif
                             );

      if (type == 'U' || type == 'B')
      {
#ifdef HYPRE_USING_SYCL
         jU = warp_reduce_sum(jU, item);
#else
         jU = warp_reduce_sum(jU);
#endif
         jU = min(jU, N);
      }

      if (type == 'L' || type == 'B')
      {
#ifdef HYPRE_USING_SYCL
         jL = warp_reduce_max(jL, item);
#else
         jL = warp_reduce_max(jL);
#endif
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

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                       COHEN
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
__global__
void expdistfromuniform(
#ifdef HYPRE_USING_SYCL
   sycl::nd_item<3>& item,
#endif
   HYPRE_Int n, float *x)
{
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG   = item.get_sub_group();
   HYPRE_Int warp_size  = SG.get_local_range().get(0);
   HYPRE_Int blockDim_x = item.get_local_range(2);
   HYPRE_Int blockDim_y = item.get_local_range(1);
   const HYPRE_Int global_thread_id  = item.get_group(2) * get_block_size(item) + get_thread_id(item);
   const HYPRE_Int total_num_threads = item.get_group_range(2) * get_block_size(item);
#else
   HYPRE_Int warp_size  = HYPRE_WARP_SIZE;
   HYPRE_Int blockDim_x = blockDim.x;
   HYPRE_Int blockDim_y = blockDim.y;
   const HYPRE_Int global_thread_id  = blockIdx.x * get_block_size() + get_thread_id();
   const HYPRE_Int total_num_threads = gridDim.x  * get_block_size();
#endif
   hypre_device_assert(blockDim_x * blockDim_y == warp_size);

   for (HYPRE_Int i = global_thread_id; i < n; i += total_num_threads)
   {
      x[i] = -logf(x[i]);
   }
}

/* T = float: single precision should be enough */
template <typename T, HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_SIZE_PER_WARP, HYPRE_Int layer>
__global__
void cohen_rowest_kernel(
#ifdef HYPRE_USING_SYCL
   sycl::nd_item<3>& item,
#endif
   HYPRE_Int nrow, HYPRE_Int *rowptr, HYPRE_Int *colidx, T *V_in, T *V_out,
   HYPRE_Int *rc, HYPRE_Int nsamples, HYPRE_Int *low, HYPRE_Int *upp, T mult)
{
#ifdef HYPRE_USING_SYCL
   sycl::sub_group SG = item.get_sub_group();
   HYPRE_Int warp_size = SG.get_local_range().get(0);
   HYPRE_Int blockDim_z = item.get_local_range(0);
   HYPRE_Int blockDim_y = item.get_local_range(1);
   HYPRE_Int blockDim_x = item.get_local_range(2);
   HYPRE_Int blockIdx_x = item.get_group(2);
   HYPRE_Int gridDim_x  = item.get_group_range(2);
   /* warp id inside the block */
   const HYPRE_Int warp_id = get_warp_id(item);
   /* lane id inside the warp */
   volatile HYPRE_Int lane_id = get_lane_id(item);
#else
   HYPRE_Int warp_size  = HYPRE_WARP_SIZE;
   HYPRE_Int blockDim_z = blockDim.z;
   HYPRE_Int blockDim_y = blockDim.y;
   HYPRE_Int blockDim_x = blockDim.x;
   HYPRE_Int blockIdx_x = blockIdx.x;
   HYPRE_Int gridDim_x  = gridDim.x;
   /* warp id inside the block */
   const HYPRE_Int warp_id = get_warp_id();
   /* lane id inside the warp */
   volatile HYPRE_Int lane_id = get_lane_id();
#if COHEN_USE_SHMEM
   __shared__ volatile HYPRE_Int s_col[NUM_WARPS_PER_BLOCK * SHMEM_SIZE_PER_WARP];
   volatile HYPRE_Int  *warp_s_col = s_col + warp_id * SHMEM_SIZE_PER_WARP;
#endif // COHEN_USE_SHMEM

#endif // HYPRE_USING_SYCL

   const HYPRE_Int num_warps = NUM_WARPS_PER_BLOCK * gridDim_x;
   hypre_device_assert(blockDim_z              == NUM_WARPS_PER_BLOCK);
   hypre_device_assert(blockDim_x * blockDim_y == warp_size);
   hypre_device_assert(sizeof(T) == sizeof(float));

   for (HYPRE_Int i = blockIdx_x * NUM_WARPS_PER_BLOCK + warp_id;
        i < nrow;
        i += num_warps)
   {
      /* load the start and end position of row i */
      HYPRE_Int tmp = -1;
      if (lane_id < 2)
      {
         tmp = read_only_load(rowptr + i + lane_id);
      }
#ifdef HYPRE_USING_SYCL
      const HYPRE_Int istart = SG.shuffle(tmp, 0);
      const HYPRE_Int iend   = SG.shuffle(tmp, 1);
#else
      const HYPRE_Int istart = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 0);
      const HYPRE_Int iend   = __shfl_sync(HYPRE_WARP_FULL_MASK, tmp, 1);
#endif

      /* works on WARP_SIZE samples at a time */
      for (HYPRE_Int r = 0; r < nsamples; r += warp_size)
      {
         T vmin = HYPRE_FLT_LARGE;
         for (HYPRE_Int j = istart; j < iend; j += warp_size)
         {
            HYPRE_Int col = -1;
            const HYPRE_Int j1 = j + lane_id;
#if COHEN_USE_SHMEM
            const HYPRE_Int j2 = j1 - istart;
            if (r == 0)
            {
               if (j1 < iend)
               {
                  col = read_only_load(colidx + j1);
                  if (j2 < SHMEM_SIZE_PER_WARP)
                  {
                     warp_s_col[j2] = col;
                  }
               }

            }
            else
            {
               if (j1 < iend)
               {
                  if (j2 < SHMEM_SIZE_PER_WARP)
                  {
                     col = warp_s_col[j2];
                  }
                  else
                  {
                     col = read_only_load(colidx + j1);
                  }
               }
            }
#else
            if (j1 < iend)
            {
               col = read_only_load(colidx + j1);
            }
#endif

            for (HYPRE_Int k = 0; k < warp_size; k++)
            {
#ifdef HYPRE_USING_SYCL
               HYPRE_Int colk = SG.shuffle(col, k);
#else
               HYPRE_Int colk = __shfl_sync(HYPRE_WARP_FULL_MASK, col, k);
#endif
               if (colk == -1)
               {
                  hypre_device_assert(j + warp_size >= iend);

                  break;
               }
               if (r + lane_id < nsamples)
               {
                  T val = read_only_load(V_in + r + lane_id + colk * nsamples);
                  vmin = min(vmin, val);
               }
            }
         }

         if (layer == 2)
         {
            if (r + lane_id < nsamples)
            {
               V_out[r + lane_id + i * nsamples] = vmin;
            }
         }
         else if (layer == 1)
         {
            if (r + lane_id >= nsamples)
            {
               vmin = 0.0;
            }

            /* partial sum along r */
#ifdef HYPRE_USING_SYCL
            vmin = warp_reduce_sum(vmin, item);
#else
            vmin = warp_reduce_sum(vmin);
#endif

            if (lane_id == 0)
            {
               if (r == 0)
               {
                  V_out[i] = vmin;
               }
               else
               {
                  V_out[i] += vmin;
               }
            }
         }
      } /* for (r = 0; ...) */

      if (layer == 1)
      {
         if (lane_id == 0)
         {
            /* estimated length of row i*/
            HYPRE_Int len = rintf( (nsamples - 1) / V_out[i] * mult );

            if (low)
            {
               len = max(low[i], len);
            }
            if (upp)
            {
               len = min(upp[i], len);
            }
            if (rc)
            {
               rc[i] = len;
            }
         }
      }
   } /* for (i = ...) */
}

template <typename T, HYPRE_Int BDIMX, HYPRE_Int BDIMY, HYPRE_Int NUM_WARPS_PER_BLOCK, HYPRE_Int SHMEM_SIZE_PER_WARP>
void csr_spmm_rownnz_cohen(HYPRE_Int M, HYPRE_Int K, HYPRE_Int N, HYPRE_Int *d_ia, HYPRE_Int *d_ja,
                           HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_low, HYPRE_Int *d_upp, HYPRE_Int *d_rc,
                           HYPRE_Int nsamples, T mult_factor, T *work)
{
#ifdef HYPRE_USING_SYCL
   sycl::range<3> bDim(NUM_WARPS_PER_BLOCK, BDIMY, BDIMX);
   hypre_assert(bDim[2] * bDim[1] == HYPRE_WARP_SIZE);
   sycl::range<3> gDim( 1, 1, (nsamples * N + bDim[0] * HYPRE_WARP_SIZE - 1) /
                        (bDim[0] * HYPRE_WARP_SIZE) );
   sycl::range<3> gDim_step1( 1, 1, (K + bDim[0] - 1) / bDim[0] );
   sycl::range<3> gDim_step2( 1, 1, (M + bDim[0] - 1) / bDim[0] );
#else
   dim3 bDim(BDIMX, BDIMY, NUM_WARPS_PER_BLOCK);
   hypre_assert(bDim.x * bDim.y == HYPRE_WARP_SIZE);
   dim3 gDim( (nsamples * N + bDim.z * HYPRE_WARP_SIZE - 1) / (bDim.z * HYPRE_WARP_SIZE) );
   dim3 gDim_step1( (K + bDim.z - 1) / bDim.z );
   dim3 gDim_step2( (M + bDim.z - 1) / bDim.z );
#endif
   T *d_V1, *d_V2, *d_V3;

   d_V1 = work;
   d_V2 = d_V1 + nsamples * N;
   //d_V1 = hypre_TAlloc(T, nsamples*N, HYPRE_MEMORY_DEVICE);
   //d_V2 = hypre_TAlloc(T, nsamples*K, HYPRE_MEMORY_DEVICE);

   /* random V1: uniform --> exp */
   hypre_CurandUniformSingle(nsamples * N, d_V1, 0, 0, 0, 0);

   HYPRE_GPU_LAUNCH( expdistfromuniform, gDim, bDim, nsamples * N, d_V1 );

   /* step-1: layer 3-2 */
   HYPRE_GPU_LAUNCH( (cohen_rowest_kernel<T, NUM_WARPS_PER_BLOCK, SHMEM_SIZE_PER_WARP, 2>), gDim_step1,
                     bDim,
                     K, d_ib, d_jb, d_V1, d_V2, NULL, nsamples, NULL, NULL, -1.0);

   //hypre_TFree(d_V1, HYPRE_MEMORY_DEVICE);

   /* step-2: layer 2-1 */
   d_V3 = (T*) d_rc;

   HYPRE_GPU_LAUNCH( (cohen_rowest_kernel<T, NUM_WARPS_PER_BLOCK, SHMEM_SIZE_PER_WARP, 1>), gDim_step2,
                     bDim,
                     M, d_ia, d_ja, d_V2, d_V3, d_rc, nsamples, d_low, d_upp, mult_factor);
   /* done */
   //hypre_TFree(d_V2, HYPRE_MEMORY_DEVICE);
}


HYPRE_Int
hypreDevice_CSRSpGemmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                    HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib, HYPRE_Int *d_jb, HYPRE_Int *d_rc)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_ROWNNZ] -= hypre_MPI_Wtime();
#endif

   const HYPRE_Int num_warps_per_block =  16;
   const HYPRE_Int shmem_size_per_warp = 128;
   const HYPRE_Int BDIMX               =   2;
   const HYPRE_Int BDIMY               = HYPRE_WARP_SIZE / BDIMX;

#ifdef HYPRE_USING_SYCL
   /* SYCL kernel configurations */
   sycl::range<3> bDim(num_warps_per_block, BDIMY, BDIMX);
   hypre_assert(bDim[2] * bDim[1] == HYPRE_WARP_SIZE);
   // for cases where one WARP works on a row
   sycl::range<3> gDim( 1, 1, (m + bDim[0] - 1) / bDim[0] );
#else
   /* CUDA kernel configurations */
   dim3 bDim(BDIMX, BDIMY, num_warps_per_block);
   hypre_assert(bDim.x * bDim.y == HYPRE_WARP_SIZE);
   // for cases where one WARP works on a row
   dim3 gDim( (m + bDim.z - 1) / bDim.z );
#endif
   HYPRE_Int   row_est_mtd    = hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle());
   HYPRE_Int   cohen_nsamples = hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle());
   float cohen_mult           = hypre_HandleSpgemmRownnzEstimateMultFactor(hypre_handle());

   if (row_est_mtd == 1)
   {
      /* naive overestimate */
      HYPRE_GPU_LAUNCH( (csr_spmm_rownnz_naive<'U', num_warps_per_block>), gDim, bDim,
                        m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, NULL, d_rc );
   }
   else if (row_est_mtd == 2)
   {
      /* naive underestimate */
      HYPRE_GPU_LAUNCH( (csr_spmm_rownnz_naive<'L', num_warps_per_block>), gDim, bDim,
                        m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_rc, NULL );
   }
   else if (row_est_mtd == 3)
   {
      /* [optional] first run naive estimate for naive lower and upper bounds,
                    which will be given to Cohen's alg as corrections */
      char *work_mem = hypre_TAlloc(char,
                                    cohen_nsamples * (n + k) * sizeof(float) +2 * m * sizeof(HYPRE_Int),
                                    HYPRE_MEMORY_DEVICE);
      char *work_mem_saved = work_mem;

      //HYPRE_Int *d_low_upp = hypre_TAlloc(HYPRE_Int, 2 * m, HYPRE_MEMORY_DEVICE);
      HYPRE_Int *d_low_upp = (HYPRE_Int *) work_mem;
      work_mem += 2 * m * sizeof(HYPRE_Int);

      HYPRE_Int *d_low = d_low_upp;
      HYPRE_Int *d_upp = d_low_upp + m;

      HYPRE_GPU_LAUNCH( (csr_spmm_rownnz_naive<'B', num_warps_per_block>), gDim, bDim,
                        m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_low, d_upp );

      /* Cohen's algorithm, stochastic approach */
      csr_spmm_rownnz_cohen<float, BDIMX, BDIMY, num_warps_per_block, shmem_size_per_warp>
      (m, k, n, d_ia, d_ja, d_ib, d_jb, d_low, d_upp, d_rc, cohen_nsamples, cohen_mult,
       (float *)work_mem);

      //hypre_TFree(d_low_upp, HYPRE_MEMORY_DEVICE);
      hypre_TFree(work_mem_saved, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      char msg[256];
      hypre_sprintf(msg, "Unknown row nnz estimation method %d! \n", row_est_mtd);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
   }

#ifdef HYPRE_PROFILE
   cudaThreadSynchronize();
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_ROWNNZ] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_GPU */
