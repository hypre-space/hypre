#include "spkernels.h"

__global__
void hypreCUDAKernel_FindDiagPos(HYPRE_Int nnz, HYPRE_Int *row_idx, HYPRE_Int *col_idx, HYPRE_Int *diag_pos);

template <char UPLO, HYPRE_Int K, typename T>
__global__
void hypreCUDAKernel_OneBlockGaussSeidelColLevSchd(T *x, HYPRE_Int *jb, HYPRE_Int *ib, T *ab, HYPRE_Int *db, HYPRE_Int *jlev, HYPRE_Int *ilev, HYPRE_Int l1, HYPRE_Int l2_k2, HYPRE_Int k1, HYPRE_Int k2)
{
   //const HYPRE_Int grid_ngroups = gridDim.x * (SPTRSV_BLOCKDIM / K);
   const HYPRE_Int grid_group_id = (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) >> 5;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);
   HYPRE_Int l2;

   for (HYPRE_Int i = k1; i < k2; i++)
   {
      if (i < k2 - 1)
      {
         if (group_lane == 0)
         {
            l2 = read_only_load(&ilev[i+1]);
         }
         l2 = __shfl_sync(HYPRE_WARP_FULL_MASK, l2, 0, K);
      }
      else
      {
         l2 = l2_k2;
      }

      if ( __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id + l1 < l2) )
      {
         HYPRE_Int p = 0, q = 0;
         T t;

         if (grid_group_id + l1 < l2 && group_lane == 0)
         {
            const HYPRE_Int r = read_only_load(&jlev[grid_group_id + l1]);
            if (UPLO == 'L')
            {
               p = read_only_load(&db[r]) + 1;
               q = read_only_load(&ib[r+1]);
               t = x[r] / read_only_load(&ab[p-1]);
            }
            else if (UPLO == 'U')
            {
               p = read_only_load(&ib[r]);
               q = read_only_load(&db[r]);
               t = x[r] / read_only_load(&ab[q]);
            }
            x[r] = t;
         }

         p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);
         q = __shfl_sync(HYPRE_WARP_FULL_MASK, q, 0, K);
         t = __shfl_sync(HYPRE_WARP_FULL_MASK, t, 0, K);

         for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
         {
            if (p < q)
            {
               const HYPRE_Int col = read_only_load(&jb[p]);
               atomicAdd( &x[col], -t * read_only_load(&ab[p]) );
            }
         }
      }

      if (i < k2 - 1)
      {
         l1 = l2;

         __syncthreads();
      }
   }
}

template <char UPLO, HYPRE_Int K, typename T>
__global__
void hypreCUDAKernel_GaussSeidelColLevSchd(T *x, HYPRE_Int *jb, HYPRE_Int *ib, T *ab, HYPRE_Int *db, HYPRE_Int *jlev, HYPRE_Int l1, HYPRE_Int l2)
{
   //const HYPRE_Int grid_ngroups = gridDim.x * (SPTRSV_BLOCKDIM / K);
   const HYPRE_Int grid_group_id = ( (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) >> 5 ) + l1;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);

   if ( __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < l2) )
   {
      HYPRE_Int p = 0, q = 0;
      T t;

      if (grid_group_id < l2 && group_lane == 0)
      {
         const HYPRE_Int r = read_only_load(&jlev[grid_group_id]);
         if (UPLO == 'L')
         {
            p = read_only_load(&db[r]) + 1;
            q = read_only_load(&ib[r+1]);
            t = x[r] / read_only_load(&ab[p-1]);
         }
         else if (UPLO == 'U')
         {
            p = read_only_load(&ib[r]);
            q = read_only_load(&db[r]);
            t = x[r] / read_only_load(&ab[q]);
         }
         x[r] = t;
      }

      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, q, 0, K);
      t = __shfl_sync(HYPRE_WARP_FULL_MASK, t, 0, K);

      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
      {
         if (p < q)
         {
            const HYPRE_Int col = read_only_load(&jb[p]);
            atomicAdd( &x[col], -t * read_only_load(&ab[p]) );
         }
      }
   }
}

template <bool TEST>
HYPRE_Int
GaussSeidelColLevSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int *d_ia, *d_ja, *d_jlevL, *d_jlevU, *d_ilevL, *d_ilevU, *d_ib, *d_jb, *d_wk, *d_db;
   HYPRE_Real *d_aa, *d_b, *d_x, *d_ab, *d_r, *d_y;
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
   cudaMalloc((void **)&d_ilevL, (n+1)*sizeof(HYPRE_Int));
   cudaMalloc((void **)&d_ilevU, (n+1)*sizeof(HYPRE_Int));
   cudaMalloc((void **)&d_ib, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ab, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_jb, nnz*sizeof(int));
   cudaMalloc((void **)&d_db, n*sizeof(int));
   cudaMalloc((void **)&d_wk, 3*nnz*sizeof(int));
   cudaMalloc((void **)&d_r, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_y, n*sizeof(HYPRE_Real));

   /*------------------- Memcpy */
   cudaMemcpy(d_ia, csr->i, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->j, nnz*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_aa, csr->data, nnz*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, b, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);

   /*------------------- analysis */
   ta = wall_timer();
   for (int j = 0; j < REPEAT; j++)
   {
      cudaMemcpy(csr->i, d_ia, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(csr->j, d_ja, nnz*sizeof(int), cudaMemcpyDeviceToHost);

      makeLevelCSR(n, csr->i, csr->j, &lev);

      cudaMemcpy(d_jlevL, lev.jlevL, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_jlevU, lev.jlevU, n*sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ilevL, lev.ilevL, (lev.nlevL+1)*sizeof(HYPRE_Int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ilevU, lev.ilevU, (lev.nlevU+1)*sizeof(HYPRE_Int), cudaMemcpyHostToDevice);

      /* convert a CSC */
      hypreDevice_CSRSpTrans_v2(n, n, nnz, d_ia, d_ja, d_aa, d_ib, d_jb, d_ab, 1, d_wk);
      /* find the positions of diagonal entries */
      const int bDim = SPTRSV_BLOCKDIM;
      const HYPRE_Int gDim2 = (nnz + bDim - 1) / bDim;
      hypreCUDAKernel_FindDiagPos <<<gDim2, bDim>>> (nnz, d_jb, d_wk, d_db);
   }
   cudaThreadSynchronize();
   ta = wall_timer() - ta;

   /*------------------- solve */
   t1 = wall_timer();
   for (HYPRE_Int j = 0; j < REPEAT; j++)
   {
      const int bDim = SPTRSV_BLOCKDIM;
      // Forward-solve
      hypre_SeqCSRMatvecDevice(n, nnz, d_ia, d_ja, d_aa, d_x, d_r);
      thrust::transform(thrust::device, d_b, d_b + n, d_r, d_r, _1 - _2 );
      cudaMemcpy(d_y, d_r, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToDevice);
      for (HYPRE_Int i = 0; i < lev.num_klevL; i++)
      {
         HYPRE_Int k1 = lev.klevL[i];
         HYPRE_Int k2 = lev.klevL[i+1];
         const HYPRE_Int group_size = 32;

         HYPRE_Int l1 = lev.ilevL[k1];
         HYPRE_Int l2 = lev.ilevL[k2];

         if (k2 == k1 + 1)
         {
            const HYPRE_Int group_size = 32;
            const HYPRE_Int num_groups_per_block = bDim / group_size;
            const HYPRE_Int gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;
            hypreCUDAKernel_GaussSeidelColLevSchd<'L', group_size> <<<gDim, bDim>>>
               (d_y, d_jb, d_ib, d_ab, d_db, d_jlevL, l1, l2);
         }
         else
         {
            hypreCUDAKernel_OneBlockGaussSeidelColLevSchd<'L', group_size> <<<1, bDim>>>
               (d_y, d_jb, d_ib, d_ab, d_db, d_jlevL, d_ilevL, l1, l2, k1, k2);
         }
      }
      thrust::transform(thrust::device, d_y, d_y + n, d_x, d_x, _1 + _2 );

      // Backward-solve
      hypre_SeqCSRMatvecDevice(n, nnz, d_ia, d_ja, d_aa, d_x, d_r);
      thrust::transform(thrust::device, d_b, d_b + n, d_r, d_r, _1 - _2 );
      cudaMemcpy(d_y, d_r, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToDevice);
      for (HYPRE_Int i = 0; i < lev.num_klevU; i++)
      {
         HYPRE_Int k1 = lev.klevU[i];
         HYPRE_Int k2 = lev.klevU[i+1];
         const HYPRE_Int group_size = 32;

         HYPRE_Int l1 = lev.ilevU[k1];
         HYPRE_Int l2 = lev.ilevU[k2];

         if (k2 == k1 + 1)
         {
            const HYPRE_Int group_size = 32;
            const HYPRE_Int num_groups_per_block = bDim / group_size;
            const HYPRE_Int gDim = (l2 - l1 + num_groups_per_block - 1) / num_groups_per_block;
            hypreCUDAKernel_GaussSeidelColLevSchd<'U', group_size> <<<gDim, bDim>>>
               (d_y, d_jb, d_ib, d_ab, d_db, d_jlevU, l1, l2);
         }
         else
         {
            hypreCUDAKernel_OneBlockGaussSeidelColLevSchd<'U', group_size> <<<1, bDim>>>
               (d_y, d_jb, d_ib, d_ab, d_db, d_jlevU, d_ilevU, l1, l2, k1, k2);
         }
      }
      thrust::transform(thrust::device, d_y, d_y + n, d_x, d_x, _1 + _2 );
   }

   //Barrier for GPU calls
   cudaThreadSynchronize();

   t2 = wall_timer() - t1;

   if (print)
   {
      printf(" [GPU] G-S C-level-scheduling, #lev in L %d, #lev in U %d\n", lev.nlevL, lev.nlevU);
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
   cudaFree(d_ilevL);
   cudaFree(d_ilevU);
   cudaFree(d_ib);
   cudaFree(d_jb);
   cudaFree(d_ab);
   cudaFree(d_wk);
   cudaFree(d_r);
   cudaFree(d_y);

   return 0;
}

template HYPRE_Int GaussSeidelColLevSchd<false>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);

template HYPRE_Int GaussSeidelColLevSchd<true>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);

