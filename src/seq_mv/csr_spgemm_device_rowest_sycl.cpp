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

#if defined(HYPRE_USING_SYCL)

/*- - - - - - - - - - - - - - - - - - - - - - - - - - *
                       NAIVE
 *- - - - - - - - - - - - - - - - - - - - - - - - - - */
template <char type>
static __inline__ __attribute__((always_inline))
void
rownnz_naive_rowi(sycl::nd_item<3>& item, HYPRE_Int rowi, HYPRE_Int lane_id, HYPRE_Int *ia,
                  HYPRE_Int *ja, HYPRE_Int *ib, HYPRE_Int &row_nnz_sum,
                  HYPRE_Int &row_nnz_max)
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();
   HYPRE_Int sub_group_size = SG.get_local_range().get(0);

   /* load the start and end position of row i of A */
   HYPRE_Int j = -1;
   if (lane_id < 2)
   {
      j = read_only_load(ia + rowi + lane_id);
   }

   const HYPRE_Int istart = SG.shuffle(j, 0);
   const HYPRE_Int iend = SG.shuffle(j, 1);

   row_nnz_sum = 0;
   row_nnz_max = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart; i < iend; i += sub_group_size)
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
            row_nnz_max = sycl::max((int)row_nnz_max, (int)(rowB_end - rowB_start));
         }
      }
   }
}

template <char type, HYPRE_Int NUM_SUBGROUPS_PER_WG>
void csr_spmm_rownnz_naive(sycl::nd_item<3>& item,
			   HYPRE_Int M, /*HYPRE_Int K,*/ HYPRE_Int N,
			   HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Int *ib,
			   HYPRE_Int *jb, HYPRE_Int *rcL, HYPRE_Int *rcU)
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();

   /* total number of sub-groups in global iteration space (no_of_sub_groups_per_WG * total_no_of_WGs) */
   const HYPRE_Int num_subgroups = NUM_SUBGROUPS_PER_WG *
     (item.get_group_range(0) * item.get_group_range(1) * item.get_group_range(2));
   /* sub-group id inside the work-group */
   const HYPRE_Int subgroup_id = SG.get_group_linear_id();
   /* lane id inside the sub-group */
   volatile const HYPRE_Int lane_id = SG.get_local_linear_id();

#ifdef HYPRE_DEBUG
   assert(item.get_local_range(2) * item.get_local_range(1) == SG.get_local_range().get(0));
#endif
   
   for (HYPRE_Int i = item.get_group(2) * NUM_SUBGROUPS_PER_WG + subgroup_id;
	i < M; i += num_subgroups)
   {
      HYPRE_Int jU, jL;

      rownnz_naive_rowi<type>(item, i, lane_id, ia, ja, ib, jU, jL);

      if (type == 'U' || type == 'B')
      {
	jU = warp_reduce_sum(jU, item);
	jU = sycl::min((int)jU, (int)N);
      }

      if (type == 'L' || type == 'B')
      {
	jL = warp_reduce_max(jL, item);
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

void expdistfromuniform(sycl::nd_item<3>& item, HYPRE_Int n, float *x)
{
   const HYPRE_Int global_thread_id = item.get_global_linear_id();
   const HYPRE_Int total_num_threads = item.get_global_range(0) * item.get_global_range(1) *
     item.get_global_range(2);

   for (HYPRE_Int i = global_thread_id; i < n; i += total_num_threads)
   {
     x[i] = -sycl::log(x[i]);
   }
}

/* T = float: single precision should be enough */
template <typename T, HYPRE_Int NUM_SUBGROUPS_PER_WG, HYPRE_Int SHMEM_SIZE_PER_SUBGROUP, HYPRE_Int layer>
void cohen_rowest_kernel(sycl::nd_item<3>& item,
			 HYPRE_Int nrow, HYPRE_Int *rowptr, HYPRE_Int *colidx, T *V_in, T *V_out,
                         HYPRE_Int *rc, HYPRE_Int nsamples, HYPRE_Int *low, HYPRE_Int *upp, T mult)
{
   sycl::ONEAPI::sub_group SG = item.get_sub_group();

   /* total number of sub-groups in global iteration space (no_of_sub_groups_per_WG * total_no_of_WGs) */
   const HYPRE_Int num_subgroups = NUM_SUBGROUPS_PER_WG * (item.get_group_range(0) * item.get_group_range(1) * item.get_group_range(2));
   /* sub-group id inside the work-group */
   const HYPRE_Int subgroup_id = SG.get_group_linear_id();
   /* lane id inside the sub-group */
   volatile const HYPRE_Int lane_id = SG.get_local_linear_id();

   HYPRE_Int sub_group_size = SG.get_local_range().get(0);

// #ifdef HYPRE_DEBUG
//    assert(blockDim.z              == NUM_SUBGROUPS_PER_WG);
//    assert(blockDim.x * blockDim.y == warpSize);
//    assert(sizeof(T) == sizeof(float));
// #endif

   for (HYPRE_Int i = item.get_group(2) * NUM_SUBGROUPS_PER_WG + subgroup_id; i < nrow; i += num_subgroups)
   {
      /* load the start and end position of row i */
      HYPRE_Int tmp = -1;
      if (lane_id < 2)
      {
         tmp = read_only_load(rowptr + i + lane_id);
      }

      const HYPRE_Int istart = SG.shuffle(tmp, 0);
      const HYPRE_Int iend = SG.shuffle(tmp, 1);

      /* works on WARP_SIZE samples at a time */
      for (HYPRE_Int r = 0; r < nsamples; r += sub_group_size)
      {
         T vmin = HYPRE_FLT_LARGE;
         for (HYPRE_Int j = istart; j < iend; j += sub_group_size)
         {
            HYPRE_Int col = -1;
            const HYPRE_Int j1 = j + lane_id;
            if (j1 < iend)
            {
               col = read_only_load(colidx + j1);
            }

            for (HYPRE_Int k = 0; k < sub_group_size; k++)
            {
               HYPRE_Int colk = SG.shuffle(col, k);
               if (colk == -1)
               {
#ifdef HYPRE_DEBUG
                  assert(j + sub_group_size >= iend);
#endif
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
            vmin = warp_reduce_sum(vmin, item);

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
               len = sycl::max((int)(low[i]), (int)len);
            }
            if (upp)
            {
               len = sycl::min((int)(upp[i]), (int)len);
            }
            if (rc)
            {
               rc[i] = len;
            }
         }
      }
   } /* for (i = ...) */
}

template <typename T, HYPRE_Int BDIMX, HYPRE_Int BDIMY,
	  HYPRE_Int NUM_SUBGROUPS_PER_WG, HYPRE_Int SHMEM_SIZE_PER_SUBGROUP>
void csr_spmm_rownnz_cohen(HYPRE_Int M, HYPRE_Int K, HYPRE_Int N,
			   HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Int *d_ib,
			   HYPRE_Int *d_jb, HYPRE_Int *d_low, HYPRE_Int *d_upp,
			   HYPRE_Int *d_rc, HYPRE_Int nsamples,
			   T mult_factor, T *work)
{
   sycl::range<3> bDim(NUM_SUBGROUPS_PER_WG, BDIMY, BDIMX);
   hypre_assert(bDim[2] * bDim[1] == HYPRE_SUBGROUP_SIZE);

   T *d_V1, *d_V2, *d_V3;

   d_V1 = work;
   d_V2 = d_V1 + nsamples*N;
   //d_V1 = hypre_TAlloc(T, nsamples*N, HYPRE_MEMORY_DEVICE);
   //d_V2 = hypre_TAlloc(T, nsamples*K, HYPRE_MEMORY_DEVICE);

   oneapi::mkl::rng::philox4x32x10* engine = hypre_HandleonemklrngGenerator(hypre_handle());
   oneapi::mkl::rng::uniform<T, oneapi::mkl::rng::uniform_method::by_default> distr(0.0, 1.0); // distribution object
   /* random V1: uniform --> exp */
   HYPRE_SYCL_CALL( oneapi::mkl::rng::generate(distr, *engine, nsamples * N, d_V1) );

   sycl::range<3> gDim(1, 1, (nsamples * N + bDim[0] * HYPRE_SUBGROUP_SIZE - 1) / (bDim[0] * HYPRE_SUBGROUP_SIZE));

   HYPRE_SYCL_3D_LAUNCH( expdistfromuniform, gDim, bDim, nsamples * N, d_V1 );

   /* step-1: layer 3-2 */
   gDim[2] = (K + bDim[0] - 1) / bDim[0];
   HYPRE_SYCL_3D_LAUNCH( (cohen_rowest_kernel<T, NUM_SUBGROUPS_PER_WG, SHMEM_SIZE_PER_SUBGROUP, 2>),
			 gDim, bDim, K, d_ib, d_jb, d_V1, d_V2, nullptr, nsamples, nullptr, nullptr, -1.0);

   //hypre_TFree(d_V1, HYPRE_MEMORY_DEVICE);

   /* step-2: layer 2-1 */
   d_V3 = (T*) d_rc;

   gDim[2] = (M + bDim[0] - 1) / bDim[0];
   HYPRE_SYCL_3D_LAUNCH( (cohen_rowest_kernel<T, NUM_SUBGROUPS_PER_WG, SHMEM_SIZE_PER_SUBGROUP, 1>),
			 gDim, bDim, M, d_ia, d_ja, d_V2, d_V3, d_rc, nsamples, d_low, d_upp, mult_factor);

   /* done */
   //hypre_TFree(d_V2, HYPRE_MEMORY_DEVICE);
}


HYPRE_Int
hypreDevice_CSRSpGemmRownnzEstimate(HYPRE_Int m, HYPRE_Int k, HYPRE_Int n,
                                    HYPRE_Int *d_ia, HYPRE_Int *d_ja,
				    HYPRE_Int *d_ib, HYPRE_Int *d_jb,
				    HYPRE_Int *d_rc)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_ROWNNZ] -= hypre_MPI_Wtime();
#endif

   const HYPRE_Int num_subgroups_per_WG = 16;
   const HYPRE_Int shmem_size_per_subgroup = 128;
   const HYPRE_Int BDIMX               =   2;
   const HYPRE_Int BDIMY               =  16;

   /* SYCL kernel configurations */
   sycl::range<3> bDim(num_subgroups_per_WG, BDIMY, BDIMX);
   hypre_assert(bDim[2] * bDim[1] == HYPRE_SUBGROUP_SIZE);
   // for cases where one WARP works on a row
   sycl::range<3> gDim(1, 1, (m + bDim[0] - 1) / bDim[0]);

   HYPRE_Int   row_est_mtd    = hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle());
   HYPRE_Int   cohen_nsamples = hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle());
   float cohen_mult           = hypre_HandleSpgemmRownnzEstimateMultFactor(hypre_handle());

   if (row_est_mtd == 1)
   {
     /* naive overestimate */
     HYPRE_SYCL_3D_LAUNCH( (csr_spmm_rownnz_naive<'U', num_subgroups_per_WG>),
			   gDim, bDim,
			   m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, nullptr, d_rc );
   }
   else if (row_est_mtd == 2)
   {
     /* naive underestimate */
     HYPRE_SYCL_3D_LAUNCH( (csr_spmm_rownnz_naive<'L', num_subgroups_per_WG>),
			   gDim, bDim,
			   m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_rc, nullptr );
   }
   else if (row_est_mtd == 3)
   {
      /* [optional] first run naive estimate for naive lower and upper bounds,
                    which will be given to Cohen's alg as corrections */
      char *work_mem = hypre_TAlloc(char, cohen_nsamples*(n+k)*sizeof(float)+2*m*sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
      char *work_mem_saved = work_mem;

      //HYPRE_Int *d_low_upp = hypre_TAlloc(HYPRE_Int, 2 * m, HYPRE_MEMORY_DEVICE);
      HYPRE_Int *d_low_upp = (HYPRE_Int *) work_mem;
      work_mem += 2*m*sizeof(HYPRE_Int);

      HYPRE_Int *d_low = d_low_upp;
      HYPRE_Int *d_upp = d_low_upp + m;

      HYPRE_SYCL_3D_LAUNCH( (csr_spmm_rownnz_naive<'B', num_subgroups_per_WG>),
			    gDim, bDim,
			    m, /*k,*/ n, d_ia, d_ja, d_ib, d_jb, d_low, d_upp );

      /* Cohen's algorithm, stochastic approach */
      csr_spmm_rownnz_cohen<float, BDIMX, BDIMY, num_subgroups_per_WG, shmem_size_per_subgroup>
	(m, k, n, d_ia, d_ja, d_ib, d_jb, d_low, d_upp, d_rc, cohen_nsamples, cohen_mult, (float *)work_mem);

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
   hypre_HandleSyclComputeQueue(hypre_handle())->wait_and_throw();
   hypre_profile_times[HYPRE_TIMER_ID_SPMM_ROWNNZ] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* HYPRE_USING_SYCL */
