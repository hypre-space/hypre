#include "spkernels.h"

template <HYPRE_Int K>
__global__
void hypreCUDAKernel_GSRowNumDep(HYPRE_Int n, HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Int *depL, HYPRE_Int *depU)
{
   const HYPRE_Int grid_group_id = (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);

   if ( __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < n) )
   {
      HYPRE_Int p = 0, q = 0;
      if (grid_group_id < n && group_lane < 2)
      {
         p = read_only_load(&ia[grid_group_id+group_lane]);
      }

      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);

      HYPRE_Int l_sum = 0, u_sum = 0;

      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
      {
         if (p < q)
         {
            const HYPRE_Int col = read_only_load(&ja[p]);
            if (col < grid_group_id)
            {
               l_sum ++;
            }
            else if (col > grid_group_id)
            {
               u_sum ++;
            }
         }
      }

      // parallel reduction
#pragma unroll
      for (HYPRE_Int d = K/2; d > 0; d >>= 1)
      {
         l_sum += __shfl_down_sync(HYPRE_WARP_FULL_MASK, l_sum, d);
         u_sum += __shfl_down_sync(HYPRE_WARP_FULL_MASK, u_sum, d);
      }

      if (grid_group_id < n && group_lane == 0)
      {
         depL[grid_group_id] = l_sum;
         depU[grid_group_id] = u_sum;
      }
   }
}

template <char UPLO, typename T>
__global__
void hypreCUDAKernel_GaussSeidelRowDynSchd(HYPRE_Int n, T *b, T *x, T *a, HYPRE_Int *ja, HYPRE_Int *ia, HYPRE_Int *jb, HYPRE_Int *ib, HYPRE_Int *jlev, HYPRE_Int *dep)
{
   const HYPRE_Int grid_warp_id = (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) >> 5;
   const HYPRE_Int warp_lane = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   // make dep volatile to tell compiler do not use cached value
   volatile HYPRE_Int *vdep = dep;
   //volatile HYPRE_Complex *vb = b;

   if ( grid_warp_id >= n )
   {
      return;
   }

   HYPRE_Int p = 0, q = 0, r = -1, s = 0, t = 0;
   T sum = 0.0, diag = 0.0, b_r;
   bool find_diag = false;

   if (warp_lane < 2)
   {
      r = read_only_load(&jlev[grid_warp_id]);
      p = read_only_load(&ia[r+warp_lane]);
      s = read_only_load(&ib[r+warp_lane]);
   }

   r = __shfl_sync(HYPRE_WARP_FULL_MASK, r, 0);
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);
   t = __shfl_sync(HYPRE_WARP_FULL_MASK, s, 1);
   s = __shfl_sync(HYPRE_WARP_FULL_MASK, s, 0);

   if (warp_lane == 0)
   {
      b_r = read_only_load(&b[r]);
      while (vdep[r] != 0);
   }

   for (p += warp_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += HYPRE_WARP_SIZE)
   {
      if (p < q)
      {
         const HYPRE_Int col = read_only_load(&ja[p]);
         const T v = read_only_load(&a[p]);
         if (col != r)
         {
            sum += v * x[col];
         }
         else
         {
            diag = v;
            find_diag = true;
         }
      }
   }

   // parallel all-reduce
#pragma unroll
   for (HYPRE_Int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
   {
      sum += __shfl_xor_sync(HYPRE_WARP_FULL_MASK, sum, d);
   }

   b_r = __shfl_sync(HYPRE_WARP_FULL_MASK, b_r, 0);

   if (find_diag)
   {
      x[r] = (b_r - sum) / diag;
      __threadfence();
   }

   for (s += warp_lane; __any_sync(HYPRE_WARP_FULL_MASK, s < t); s += HYPRE_WARP_SIZE)
   {
      if (s < t)
      {
         const HYPRE_Int z = read_only_load(&jb[s]);
         if (UPLO == 'L')
         {
            if (z > r)
            {
               atomicSub(&dep[z], 1);
            }
         }
         else if (UPLO == 'U')
         {
            if (z < r)
            {
               atomicSub(&dep[z], 1);
            }
         }
      }
   }
}

template <bool TEST>
HYPRE_Int
GaussSeidelRowDynSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int *d_ia, *d_ja, *d_jt, *d_jlevL, *d_jlevU, *d_ib, *d_jb, *d_depL, *d_depU, *d_dep;
   HYPRE_Real *d_aa, *d_b, *d_x;
   double t1, t2, ta;
   struct level_t lev;

   allocLevel(n, &lev);

   /*------------------- allocate Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(int));
   cudaMalloc((void **)&d_aa, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_b, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_x, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_jlevL, n*sizeof(int));
   cudaMalloc((void **)&d_jlevU, n*sizeof(int));
   cudaMalloc((void **)&d_depL, n*sizeof(int));
   cudaMalloc((void **)&d_depU, n*sizeof(int));
   cudaMalloc((void **)&d_dep, n*sizeof(int));
   cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_jb, nnz*sizeof(int));
   cudaMalloc((void **)&d_jt, nnz*sizeof(int));

   /*------------------- Memcpy */
   cudaMemcpy(d_ia, csr->i, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->j, nnz*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_aa, csr->data, nnz*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, b, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);

   const int bDim = SPTRSV_BLOCKDIM;
   const HYPRE_Int num_warps_per_block = bDim / HYPRE_WARP_SIZE;
   const HYPRE_Int gDim = (n + num_warps_per_block - 1) / num_warps_per_block;

   /*------------------- analysis */
   ta = wall_timer();
   for (int j = 0; j < REPEAT; j++)
   {
      cudaMemcpy(csr->i, d_ia, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->j, d_ja, nnz*sizeof(int), cudaMemcpyDeviceToHost);

      makeLevelCSR(n, csr->i, csr->j, &lev);

      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);

      const int bDim = SPTRSV_BLOCKDIM;
      hypreCUDAKernel_GSRowNumDep<HYPRE_WARP_SIZE> <<<gDim, bDim>>> (n, d_ia, d_ja, d_depL, d_depU);

      /* convert a CSC */
      cudaMemcpy(d_jt, d_ja, nnz*sizeof(int), cudaMemcpyDeviceToDevice);
      hypreDevice_CsrRowPtrsToIndices_v2(n, d_ia, d_jb);
      thrust::stable_sort_by_key(thrust::device, d_jt, d_jt + nnz, d_jb);
      hypreDevice_CsrRowIndicesToPtrs_v2(n, nnz, d_jt, d_ib);
   }
   cudaThreadSynchronize();
   ta = wall_timer() - ta;

   /*------------------- solve */
   t1 = wall_timer();
   for (HYPRE_Int j = 0; j < REPEAT; j++)
   {
      // Forward-solve
      cudaMemcpy(d_dep, d_depL, n*sizeof(int), cudaMemcpyDeviceToDevice);
      hypreCUDAKernel_GaussSeidelRowDynSchd<'L'> <<<gDim, bDim>>>
         (n, d_b, d_x, d_aa, d_ja, d_ia, d_jb, d_ib, d_jlevL, d_dep);

      // check dep counters == 0
      if (TEST)
      {
         HYPRE_Int *tmp = (HYPRE_Int *) malloc(n*sizeof(HYPRE_Int));
         cudaMemcpy(tmp, d_dep, n*sizeof(int), cudaMemcpyDeviceToHost);
         for (HYPRE_Int i=0; i<n; i++)
         {
            assert(tmp[i] == 0);
         }
         free(tmp);
      }

      // Backward-solve
      cudaMemcpy(d_dep, d_depU, n*sizeof(int), cudaMemcpyDeviceToDevice);
      hypreCUDAKernel_GaussSeidelRowDynSchd<'U'> <<<gDim, bDim>>>
         (n, d_b, d_x, d_aa, d_ja, d_ia, d_jb, d_ib, d_jlevU, d_dep);

      // check dep counters == 0
      if (TEST)
      {
         HYPRE_Int *tmp = (HYPRE_Int *) malloc(n*sizeof(HYPRE_Int));
         cudaMemcpy(tmp, d_dep, n*sizeof(int), cudaMemcpyDeviceToHost);
         for (HYPRE_Int i=0; i<n; i++)
         {
            assert(tmp[i] == 0);
         }
         free(tmp);
      }
   }

   //Barrier for GPU calls
   cudaThreadSynchronize();
   t2 = wall_timer() - t1;

   if (print)
   {
      printf(" [GPU] G-S dynamic-scheduling, #lev in L %d, #lev in U %d\n", lev.nlevL, lev.nlevU);
      printf("  time(s) = %.2e, Gflops = %-5.3f", t2/REPEAT, REPEAT*4*((nnz)/1e9)/t2);
      printf("  [analysis time %.2e (%.1e x T_sol)] ", ta/REPEAT, ta/t2);
   }
   /*-------- copy x to host mem */
   cudaMemcpy(x, d_x, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_aa);
   cudaFree(d_b);
   cudaFree(d_x);
   FreeLev(&lev);
   cudaFree(d_jlevL);
   cudaFree(d_jlevU);
   cudaFree(d_depL);
   cudaFree(d_depU);
   cudaFree(d_dep);
   cudaFree(d_ib);
   cudaFree(d_jb);
   cudaFree(d_jt);

   return 0;
}

template HYPRE_Int GaussSeidelRowDynSchd<false>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);

template HYPRE_Int GaussSeidelRowDynSchd<true>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
