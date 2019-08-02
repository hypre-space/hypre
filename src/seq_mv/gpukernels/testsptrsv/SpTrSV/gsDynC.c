#include "../../common/spkernels.h"


template <HYPRE_Int K>
__global__
void hypreCUDAKernel_GSRowNumDep(HYPRE_Int n, HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Int *depL, HYPRE_Int *depU);

__global__
void hypreCUDAKernel_FindDiagPos(HYPRE_Int nnz, HYPRE_Int *row_idx, HYPRE_Int *col_idx, HYPRE_Int *diag_pos)
{
   HYPRE_Int tid = hypre_cuda_get_grid_thread_id<1,1>();

   if (tid >= nnz)
   {
      return;
   }

   HYPRE_Int i = read_only_load(row_idx+tid);
   HYPRE_Int j = read_only_load(col_idx+tid);

   if (i == j)
   {
      diag_pos[i] = tid;
   }
}

template <char UPLO, typename T>
__global__
void hypreCUDAKernel_GaussSeidelColDynSchd(HYPRE_Int n, T *x, HYPRE_Int *jb, HYPRE_Int *ib, T *ab, HYPRE_Int *db, HYPRE_Int *jlev, HYPRE_Int *dep)
{
   const HYPRE_Int grid_warp_id = (blockIdx.x * SPTRSV_BLOCKDIM + threadIdx.x) >> 5;
   const HYPRE_Int warp_lane = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   // make dep volatile to tell compiler do not use cached value
   volatile HYPRE_Int *vdep = dep;
   volatile T *vx = x;
#if __CUDA_ARCH__ < 700
   volatile __shared__ HYPRE_Int sh_ind[SPTRSV_BLOCKDIM];
   volatile __shared__ T sh_val[SPTRSV_BLOCKDIM];
#else
   HYPRE_Int sh_ind;
   T sh_val;
#endif

   if ( grid_warp_id >= n )
   {
      return;
   }

   HYPRE_Int r = -1, p = 0, q = 0;

   if (warp_lane == 0)
   {
      r = read_only_load(&jlev[grid_warp_id]);
      if (UPLO == 'L')
      {
         p = read_only_load(&db[r]) + 1;
         q = read_only_load(&ib[r+1]);
      }
      else if (UPLO == 'U')
      {
         p = read_only_load(&ib[r]);
         q = read_only_load(&db[r]);
      }
   }

   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, q, 0);

   p += warp_lane;
   if (p < q)
   {
#if __CUDA_ARCH__ < 700
      sh_ind[threadIdx.x] = read_only_load(&jb[p]);
      sh_val[threadIdx.x] = read_only_load(&ab[p]);
#else
      sh_ind = read_only_load(&jb[p]);
      sh_val = read_only_load(&ab[p]);
#endif
   }

   T xr;

   if (warp_lane == 0)
   {
      if (UPLO == 'L')
      {
         xr = 1.0 / read_only_load(&ab[p-1]);
      }
      else if (UPLO == 'U')
      {
         xr = 1.0 / read_only_load(&ab[q]);
      }

      while (vdep[r] != 0);

      xr *= vx[r];
      x[r] = xr;
   }

   xr = __shfl_sync(HYPRE_WARP_FULL_MASK, xr, 0);

   if (p < q)
   {
#if __CUDA_ARCH__ < 700
      const HYPRE_Int col = sh_ind[threadIdx.x];
      atomicAdd(&x[col], -xr * sh_val[threadIdx.x]);
#else
      const HYPRE_Int col = sh_ind;
      atomicAdd(&x[col], -xr * sh_val);
#endif
      __threadfence();
      atomicSub(&dep[col], 1);
   }
   p += HYPRE_WARP_SIZE;

   for (/*p += warp_lane*/; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += HYPRE_WARP_SIZE)
   {
      if (p < q)
      {
         const HYPRE_Int col = read_only_load(&jb[p]);
         atomicAdd( &x[col], -xr * read_only_load(&ab[p]) );
         __threadfence();
         atomicSub(&dep[col], 1);
      }
   }
}

template <bool TEST>
HYPRE_Int
GaussSeidelColDynSchd(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int *d_ia, *d_ja, *d_jlevL, *d_jlevU, *d_ib, *d_jb, *d_depL, *d_depU, *d_dep, *d_wk, *d_db;
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
   cudaMalloc((void **)&d_depL, n*sizeof(int));
   cudaMalloc((void **)&d_depU, n*sizeof(int));
   cudaMalloc((void **)&d_dep, n*sizeof(int));
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
      hypreDevice_CSRSpTrans_v2(n, n, nnz, d_ia, d_ja, d_aa, d_ib, d_jb, d_ab, 1, d_wk);
      /* find the positions of diagonal entries */
      const HYPRE_Int gDim2 = (nnz + bDim - 1) / bDim;
      hypreCUDAKernel_FindDiagPos <<<gDim2, bDim>>> (nnz, d_jb, d_wk, d_db);
   }
   cudaThreadSynchronize();
   ta = wall_timer() - ta;

   /*------------------- solve */
   t1 = wall_timer();
   for (HYPRE_Int j = 0; j < REPEAT; j++)
   {
      // Forward-solve
      hypre_SeqCSRMatvecDevice(n, nnz, d_ia, d_ja, d_aa, d_x, d_r);
      thrust::transform(thrust::device, d_b, d_b + n, d_r, d_r, _1 - _2 );
      cudaMemcpy(d_dep, d_depL, n*sizeof(int), cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_y, d_r, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToDevice);
      hypreCUDAKernel_GaussSeidelColDynSchd<'L'> <<<gDim, bDim>>>
         (n, d_y, d_jb, d_ib, d_ab, d_db, d_jlevL, d_dep);
      thrust::transform(thrust::device, d_y, d_y + n, d_x, d_x, _1 + _2 );

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
      hypre_SeqCSRMatvecDevice(n, nnz, d_ia, d_ja, d_aa, d_x, d_r);
      thrust::transform(thrust::device, d_b, d_b + n, d_r, d_r, _1 - _2 );
      cudaMemcpy(d_dep, d_depU, n*sizeof(int), cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_y, d_r, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToDevice);
      hypreCUDAKernel_GaussSeidelColDynSchd<'U'> <<<gDim, bDim>>>
         (n, d_y, d_jb, d_ib, d_ab, d_db, d_jlevU, d_dep);
      thrust::transform(thrust::device, d_y, d_y + n, d_x, d_x, _1 + _2 );

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
      printf(" [GPU] G-S C-dynamic-scheduling, #lev in L %d, #lev in U %d\n", lev.nlevL, lev.nlevU);
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
   cudaFree(d_ab);
   cudaFree(d_wk);
   cudaFree(d_r);
   cudaFree(d_y);

   return 0;
}

template HYPRE_Int GaussSeidelColDynSchd<false>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);

template HYPRE_Int GaussSeidelColDynSchd<true>(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print);
