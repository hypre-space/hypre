#include "spkernels.h"

template <char UPLO, typename T>
__global__
void hypreCUDAKernel_GaussSeidelRowDynSchd_v2(HYPRE_Int n, T *b, T *x, T *a, HYPRE_Int *ja, HYPRE_Int *ia, HYPRE_Int *jlev, char *done)
{
   const HYPRE_Int grid_warp_id = (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) >> 5;
   const HYPRE_Int warp_lane = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   // make done volatile to tell compiler do not use cached value
   volatile char *vdone = done;
   volatile T *vx = x;

   if ( grid_warp_id >= n )
   {
      return;
   }

   HYPRE_Int p = 0, q = 0, r = -1;
   T sum = 0.0, diag = 0.0;
   bool find_diag = false;

   if (warp_lane < 2)
   {
      r = read_only_load(&jlev[grid_warp_id]);
      p = read_only_load(&ia[r+warp_lane]);
   }

   r = __shfl_sync(HYPRE_WARP_FULL_MASK, r, 0);
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   if (warp_lane == 0)
   {
      sum = read_only_load(&b[r]);
   }

   for (p += warp_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += HYPRE_WARP_SIZE)
   {
      if (p < q)
      {
         const HYPRE_Int col = read_only_load(&ja[p]);
         const T val = read_only_load(&a[p]);

         if (col == r)
         {
            diag = val;
            find_diag = true;
         }
         else if ( (UPLO == 'L' && col > r) || (UPLO == 'U' && col < r) )
         {
            sum -= val * vx[col];
         }
         else
         {
            while (vdone[col] == 0);
            sum -= val * vx[col];
         }
      }
   }

   // parallel all-reduce
#pragma unroll
   for (HYPRE_Int d = HYPRE_WARP_SIZE/2; d > 0; d >>= 1)
   {
      sum += __shfl_xor_sync(HYPRE_WARP_FULL_MASK, sum, d);
   }

   if (find_diag)
   {
      x[r] = sum / diag;
      __threadfence();
      vdone[r] = 1;
   }
}

template <bool TEST>
HYPRE_Int
GaussSeidelRowDynSchd_v2(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int *d_ia, *d_ja, *d_jlevL, *d_jlevU;
   char *d_done;
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
   cudaMalloc((void **)&d_done, n*sizeof(char));

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
   }
   cudaThreadSynchronize();
   ta = wall_timer() - ta;

   /*------------------- solve */
   t1 = wall_timer();
   for (HYPRE_Int j = 0; j < REPEAT; j++)
   {
      // Forward-solve
      cudaMemset(d_done, 0, n*sizeof(char));
      hypreCUDAKernel_GaussSeidelRowDynSchd_v2<'L'> <<<gDim, bDim>>>
         (n, d_b, d_x, d_aa, d_ja, d_ia, d_jlevL, d_done);

      // check done == 1
      if (TEST)
      {
         char *tmp = (char *) malloc(n*sizeof(char));
         cudaMemcpy(tmp, d_done, n*sizeof(char), cudaMemcpyDeviceToHost);
         for (HYPRE_Int i = 0; i < n; i++)
         {
            assert(tmp[i] == 1);
         }
         free(tmp);
      }

      // Backward-solve
      cudaMemset(d_done, 0, n*sizeof(char));
      hypreCUDAKernel_GaussSeidelRowDynSchd_v2<'U'> <<<gDim, bDim>>>
         (n, d_b, d_x, d_aa, d_ja, d_ia, d_jlevU, d_done);

      // check done == 1
      if (TEST)
      {
         char *tmp = (char *) malloc(n*sizeof(char));
         cudaMemcpy(tmp, d_done, n*sizeof(char), cudaMemcpyDeviceToHost);
         for (HYPRE_Int i = 0; i < n; i++)
         {
            assert(tmp[i] == 1);
         }
         free(tmp);
      }
   }

   //Barrier for GPU calls
   cudaThreadSynchronize();
   t2 = wall_timer() - t1;

   if (print)
   {
      printf(" [GPU] G-S R-dynamic-scheduling-v2, #lev in L %d, #lev in U %d\n", lev.nlevL, lev.nlevU);
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

   return 0;
}

template HYPRE_Int GaussSeidelRowDynSchd_v2<false>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);

template HYPRE_Int GaussSeidelRowDynSchd_v2<true>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
