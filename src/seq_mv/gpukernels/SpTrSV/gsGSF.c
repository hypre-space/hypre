#include "spkernels.h"

#define WARP_SIZE 32
#define WARP_PER_BLOCK 16

template <HYPRE_Int K>
__global__
void hypreCUDAKernel_GSRowNumDep(HYPRE_Int n, HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Int *depL, HYPRE_Int *depU);

__global__
void hypreCUDAKernel_FindDiagPos(HYPRE_Int nnz, HYPRE_Int *row_idx, HYPRE_Int *col_idx, HYPRE_Int *diag_pos);

template <char UPLO, typename VALUE_TYPE>
__global__
void spts_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                 const int* __restrict__        d_cscRowIdx,
                                 const VALUE_TYPE* __restrict__ d_cscVal,
                                 const int* __restrict__        d_diagPos,
                                 /*const int* __restrict__        d_csrRowPtr,*/
                                 int*                           d_csrRowHisto,
                                 VALUE_TYPE*                    d_left_sum,
                                 /*VALUE_TYPE*                    d_partial_sum,*/
                                 const int                      m,
                                 /*const int                      nnz,*/
                                 const VALUE_TYPE* __restrict__ d_b,
                                 VALUE_TYPE*                    d_x/*,
                                 int*                           d_while_profiler*/)

{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_x_id = global_id / WARP_SIZE;
    volatile __shared__ int s_csrRowHisto[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];
    volatile int *v_d_csrRowHisto = d_csrRowHisto;
    volatile VALUE_TYPE *v_d_left_sum = d_left_sum;

    if (global_x_id >= m) return;

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    int col_id;
    if (UPLO == 'L')
    {
       col_id = global_x_id;
    }
    else if (UPLO == 'U')
    {
       col_id = m - global_x_id - 1;
    }

    // Prefetch
    //const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[d_cscColPtr[global_x_id]];
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[d_diagPos[col_id]];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    if (threadIdx.x < WARP_PER_BLOCK) { s_csrRowHisto[threadIdx.x] = 1; s_left_sum[threadIdx.x] = 0; }
    __syncthreads();

    clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (s_csrRowHisto[local_warp_id] != v_d_csrRowHisto[col_id] + 1);


    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_csrRowHisto[global_x_id]) :: "memory");
    //}
    //while (s_csrRowHisto[local_warp_id] != graphInDegree );

    VALUE_TYPE xi = v_d_left_sum[col_id] + s_left_sum[local_warp_id];
    xi = (d_b[col_id] - xi) * coef;

    int p, q;
    if (UPLO == 'L')
    {
       p = d_diagPos[col_id] + 1;
       q = d_cscColPtr[col_id+1];
    }
    else if (UPLO == 'U')
    {
       p = d_cscColPtr[col_id];
       q = d_diagPos[col_id];
    }
    // Producer
    //for (int j = d_cscColPtr[global_x_id] + 1 + lane_id; j < d_cscColPtr[global_x_id+1]; j += WARP_SIZE) {
    for (int j = p + lane_id; j < q; j += WARP_SIZE) {
        if (UPLO == 'L')
        {
           int rowIdx = d_cscRowIdx[j];
           if (rowIdx < starting_x + WARP_PER_BLOCK) {
              atomicAdd((VALUE_TYPE *)&s_left_sum[rowIdx - starting_x], xi * d_cscVal[j]);
              __threadfence();
              atomicAdd((int *)&s_csrRowHisto[rowIdx - starting_x], 1);
           }
           else {
              atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
              __threadfence();
              atomicSub(&d_csrRowHisto[rowIdx], 1);
           }
        }
        else if (UPLO == 'U')
        {
           int rowIdx = d_cscRowIdx[j];
           int rowIdx2 = m - 1 - rowIdx;
           if (rowIdx2 < starting_x + WARP_PER_BLOCK) {
              atomicAdd((VALUE_TYPE *)&s_left_sum[rowIdx2 - starting_x], xi * d_cscVal[j]);
              __threadfence();
              atomicAdd((int *)&s_csrRowHisto[rowIdx2 - starting_x], 1);
           }
           else {
              atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
              __threadfence();
              atomicSub(&d_csrRowHisto[rowIdx], 1);
           }
        }
    }
    // Finish
    if (!lane_id) d_x[col_id] = xi;
}


HYPRE_Int
GaussSeidelColGSF(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int *d_ia, *d_ja, *d_ib, *d_jb, *d_dep, *d_depL, *d_depU, *d_wk, *d_db;
   HYPRE_Real *d_aa, *d_b, *d_x, *d_ab, *d_r, *d_y, *d_left_sum;
   double t1, t2, ta;

   /*------------------- allocate Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(int));
   cudaMalloc((void **)&d_aa, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_b, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_x, n*sizeof(HYPRE_Real));
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
   cudaMalloc((void **)&d_left_sum, sizeof(HYPRE_Real) * n);

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

      cudaMemset(d_left_sum, 0, sizeof(HYPRE_Real) * n);

      int num_threads = WARP_PER_BLOCK * WARP_SIZE;
      int num_blocks = ceil ((double)n / (double)(num_threads/WARP_SIZE));

      spts_syncfree_cuda_executor<'L'> <<< num_blocks, num_threads >>>
                                (d_ib, d_jb, d_ab, d_db,
                                 d_dep,
                                 d_left_sum,
                                 n, d_r, d_y);

      thrust::transform(thrust::device, d_y, d_y + n, d_x, d_x, _1 + _2 );

      // Backward-solve
      hypre_SeqCSRMatvecDevice(n, nnz, d_ia, d_ja, d_aa, d_x, d_r);
      thrust::transform(thrust::device, d_b, d_b + n, d_r, d_r, _1 - _2 );
      cudaMemcpy(d_dep, d_depU, n*sizeof(int), cudaMemcpyDeviceToDevice);

      cudaMemset(d_left_sum, 0, sizeof(HYPRE_Real) * n);

      spts_syncfree_cuda_executor<'U'> <<< num_blocks, num_threads >>>
                                (d_ib, d_jb, d_ab, d_db,
                                 d_dep,
                                 d_left_sum,
                                 n, d_r, d_y);

      thrust::transform(thrust::device, d_y, d_y + n, d_x, d_x, _1 + _2 );
   }

   //Barrier for GPU calls
   cudaThreadSynchronize();
   t2 = wall_timer() - t1;

   if (print)
   {
      printf(" [GPU] G-S C-GSF\n");
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
   cudaFree(d_depL);
   cudaFree(d_depU);
   cudaFree(d_dep);
   cudaFree(d_ib);
   cudaFree(d_jb);
   cudaFree(d_ab);
   cudaFree(d_wk);
   cudaFree(d_r);
   cudaFree(d_y);
   cudaFree(d_left_sum);

   return 0;
}

