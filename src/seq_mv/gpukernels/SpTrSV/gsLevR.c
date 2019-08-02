#include "spkernels.h"

template <HYPRE_Int K, typename T>
__global__
void hypreCUDAKernel_GaussSeidelRowLevSchd(T *b, T *x, T *a, int *ja, int *ia, int *jlev, int *ilev, int k1, int k2 )
{
   //const HYPRE_Int grid_ngroups = gridDim.x * (SPTRSV_BLOCKDIM / K);
   for(int i = k1; i < k2; i++)
   { 
      int l1 = ilev[i];
      int l2 = ilev[i+1];
      const HYPRE_Int grid_group_id = (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) / K + l1;
      const HYPRE_Int group_lane = threadIdx.x & (K - 1);
      if ( __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < l2) )
      {
         HYPRE_Int p = 0, q = 0, r = -1;
         T sum = 0.0, diag = 0.0;
         bool find_diag = false;

         if (grid_group_id < l2 && group_lane < 2)
         {
            r = read_only_load(&jlev[grid_group_id]);
            p = read_only_load(&ia[r+group_lane]);
         }

         q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
         p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);
         r = __shfl_sync(HYPRE_WARP_FULL_MASK, r, 0, K);

         for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
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
         for (HYPRE_Int d = K/2; d > 0; d >>= 1)
         {
            sum += __shfl_xor_sync(HYPRE_WARP_FULL_MASK, sum, d);
         }

         if (find_diag)
         {
            x[r] = (read_only_load(&b[r]) - sum) / diag;
         }
      }
      __syncthreads();
   }
}

template <HYPRE_Int K, typename T>
__global__
void hypreCUDAKernel_GaussSeidelRowLevSchd(T *b, T *x, T *a, int *ja, int *ia, int *jlev, int l1, int l2)
{
   //const HYPRE_Int grid_ngroups = gridDim.x * (SPTRSV_BLOCKDIM / K);
   const HYPRE_Int grid_group_id = (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) / K + l1;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);

   if ( __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < l2) )
   {
      HYPRE_Int p = 0, q = 0, r = -1;
      T sum = 0.0, diag = 0.0;
      bool find_diag = false;

      if (grid_group_id < l2 && group_lane < 2)
      {
         r = read_only_load(&jlev[grid_group_id]);
         p = read_only_load(&ia[r+group_lane]);
      }

      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);
      r = __shfl_sync(HYPRE_WARP_FULL_MASK, r, 0, K);

      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
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
      for (HYPRE_Int d = K/2; d > 0; d >>= 1)
      {
         sum += __shfl_xor_sync(HYPRE_WARP_FULL_MASK, sum, d);
      }

      if (find_diag)
      {
         x[r] = (read_only_load(&b[r]) - sum) / diag;
      }
   }
}

/*
void
hypreDevice_GaussSeidelRowLevSchd(HYPRE_Int n, HYPRE_Int nnz, HYPRE_Int *d_ia, HYPRE_Int *d_ja,
                                  HYPRE_Complex *d_a, HYPRE_Complex *d_x, HYPRE_Complex *d_b)
{
}
*/

template <bool TEST>
HYPRE_Int
GaussSeidelRowLevSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int i, j, *d_ia, *d_ja, *d_jlevL, *d_jlevU, *d_ilevL, *d_ilevU;
   HYPRE_Real *d_a, *d_b, *d_x;
   double t1, t2, ta;
   struct level_t lev;

   allocLevel(n, &lev);
   cudaMalloc((void **)&d_jlevL, n*sizeof(int));
   cudaMalloc((void **)&d_jlevU, n*sizeof(int));
   cudaMalloc((void **)&d_ilevL, n*sizeof(int));
   cudaMalloc((void **)&d_ilevU, n*sizeof(int));

   /*------------------- allocate Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(int));
   cudaMalloc((void **)&d_a, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_b, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_x, n*sizeof(HYPRE_Real));

   /*------------------- Memcpy */
   cudaMemcpy(d_ia, csr->i, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->j, nnz*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_a, csr->data, nnz*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, b, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);

   /*------------------- analysis */
   ta = wall_timer();
   for (int j=0; j < REPEAT; j++)
   {
      cudaMemcpy(csr->i, d_ia, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->j, d_ja, nnz*sizeof(int), cudaMemcpyDeviceToHost);
      makeLevelCSR(n, csr->i, csr->j, &lev);
      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ilevL, lev.ilevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ilevU, lev.ilevU, n*sizeof(int), cudaMemcpyHostToDevice);
   }
   cudaThreadSynchronize();
   ta = wall_timer() - ta;

   /*------------------- solve */
   t1 = wall_timer();
   for (j = 0; j < REPEAT; j++)
   {
      const int bDim = SPTRSV_BLOCKDIM;
      int gDim;
      // Forward-solve
      for (i = 0; i < lev.num_klevL; i++)
      {
         int k1 = lev.klevL[i];
         int k2 = lev.klevL[i+1];
         const HYPRE_Int group_size = 32;
         
         if (k2 == k1 + 1)
         {
            int l1 = lev.ilevL[k1];
            int l2 = lev.ilevL[k2];
         const HYPRE_Int group_size = 32;
         const HYPRE_Int num_groups_per_block = bDim / group_size;
            gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;          
         }
         else
         {
            gDim = 1;
         }
         
         //HYPRE_Int gDim = lev.block_klevL[i];
         hypreCUDAKernel_GaussSeidelRowLevSchd<group_size> <<<gDim, bDim>>> (d_b, d_x, d_a, d_ja, d_ia, d_jlevL, d_ilevL, k1, k2);
      }

      // Backward-solve
      for (i = 0; i < lev.num_klevU; i++)
      {
         int k1 = lev.klevU[i];
         int k2 = lev.klevU[i+1];
         const HYPRE_Int group_size = 32;
         
         if (k2 == k1 + 1)
         {
            int l1 = lev.ilevU[k1];
            int l2 = lev.ilevU[k2];
         const HYPRE_Int group_size = 32;
         const HYPRE_Int num_groups_per_block = bDim / group_size;
            gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;          
         }
         else
         {
            gDim = 1;
         }
                  
         hypreCUDAKernel_GaussSeidelRowLevSchd<group_size> <<<gDim, bDim>>> (d_b, d_x, d_a, d_ja, d_ia, d_jlevU, d_ilevU, k1, k2);
      }
   }
   cudaThreadSynchronize();
   
   t2 = wall_timer() - t1;

   if (print)
   {
      printf(" [GPU] G-S level-scheduling, #lev in L %d, #lev in U %d\n", lev.nlevL, lev.nlevU);
      printf("  time(s) = %.2e, Gflops = %-5.3f", t2/REPEAT, REPEAT*4*((nnz)/1e9)/t2);
      printf("  [analysis time %.2e (%.1e x T_sol)] ", ta/REPEAT, ta/t2);
   }
   /*-------- copy x to host mem */
   cudaMemcpy(x, d_x, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_x);
   FreeLev(&lev);
   cudaFree(d_jlevL);
   cudaFree(d_jlevU);
   cudaFree(d_ilevL);
   cudaFree(d_ilevU);

   return 0;
}

template HYPRE_Int GaussSeidelRowLevSchd<false>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
template HYPRE_Int GaussSeidelRowLevSchd<true>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
