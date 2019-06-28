#include "spmv.h"
#include <cuda_runtime.h>
#include "cusparse.h"

template <HYPRE_Int K, typename T>
__global__
void csr_v_k_shared(HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, T *d_a, T *d_x, T *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *              shared memory reduction
    *              K threads-Warp  per row
    *------------------------------------------------------------*/
   // num of full-warps
   HYPRE_Int nw = gridDim.x*BLOCKDIM/K;
   // full warp id
   HYPRE_Int wid = (blockIdx.x*BLOCKDIM+threadIdx.x)/K;
   // thread lane in each full warp
   HYPRE_Int lane = threadIdx.x & (K-1);
   // full warp lane in each block
   HYPRE_Int wlane = threadIdx.x/K;
   // shared memory for patial result
   volatile __shared__ T r[BLOCKDIM+K/2];
   volatile __shared__ HYPRE_Int startend[BLOCKDIM/K][2];
   for (HYPRE_Int row = wid; row < n; row += nw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[wlane][lane] = d_ia[row+lane];
      }
      HYPRE_Int p = startend[wlane][0];
      HYPRE_Int q = startend[wlane][1];
      T sum = 0.0;
      for (HYPRE_Int i=p+lane; i<q; i+=K)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
      // parallel reduction
      r[threadIdx.x] = sum;
#pragma unroll
      for (HYPRE_Int d = K/2; d > 0; d >>= 1)
      {
         r[threadIdx.x] = sum = sum + r[threadIdx.x+d];
      }
      if (lane == 0)
      {
         d_y[row] = r[threadIdx.x];
      }
   }
}

#define VERSION 1

/* K is the number of threads working on a single row. K = 2, 4, 8, 16, 32 */
template <HYPRE_Int K, typename T>
__global__
void csr_v_k_shuffle(HYPRE_Int n, HYPRE_Int *d_ia, HYPRE_Int *d_ja, T *d_a, T *d_x, T *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *              shared memory reduction
    *           (Group of K threads) per row
    *------------------------------------------------------------*/
   const HYPRE_Int grid_ngroups = gridDim.x * (BLOCKDIM / K);
   HYPRE_Int grid_group_id = (blockIdx.x * BLOCKDIM + threadIdx.x) / K;
   const HYPRE_Int group_lane = threadIdx.x & (K - 1);
   const HYPRE_Int warp_lane = threadIdx.x & (HYPRE_WARP_SIZE - 1);
   const HYPRE_Int warp_group_id = warp_lane / K;
   const HYPRE_Int warp_ngroups = HYPRE_WARP_SIZE / K;

   for (; __any_sync(HYPRE_WARP_FULL_MASK, grid_group_id < n); grid_group_id += grid_ngroups)
   {
#if 0
      HYPRE_Int p = 0, q = 0;
      if (grid_group_id < n && group_lane < 2)
      {
         p = read_only_load(&d_ia[grid_group_id+group_lane]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1, K);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0, K);
#else
      const HYPRE_Int s = grid_group_id - warp_group_id + warp_lane;
      HYPRE_Int p = 0, q = 0;
      if (s <= n && warp_lane <= warp_ngroups)
      {
         p = read_only_load(&d_ia[s]);
      }
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id+1);
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, warp_group_id);
#endif
      T sum = 0.0;
#if VERSION == 1
#pragma unroll(1)
      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K * 2)
      {
         if (p < q)
         {
            sum += read_only_load(&d_a[p]) * read_only_load(&d_x[read_only_load(&d_ja[p])]);
            if (p + K < q)
            {
               sum += read_only_load(&d_a[p+K]) * read_only_load(&d_x[read_only_load(&d_ja[p+K])]);
            }
         }
      }
#elif VERSION == 2
#pragma unroll(1)
      for (p += group_lane; __any_sync(HYPRE_WARP_FULL_MASK, p < q); p += K)
      {
         if (p < q)
         {
            sum += read_only_load(&d_a[p]) * read_only_load(&d_x[read_only_load(&d_ja[p])]);
         }
      }
#else
#pragma unroll(1)
      for (p += group_lane;  p < q; p += K)
      {
         sum += read_only_load(&d_a[p]) * read_only_load(&d_x[read_only_load(&d_ja[p])]);
      }
#endif
      // parallel reduction
#pragma unroll
      for (HYPRE_Int d = K/2; d > 0; d >>= 1)
      {
         sum += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum, d);
      }
      if (grid_group_id < n && group_lane == 0)
      {
         d_y[grid_group_id] = sum;
      }
   }
}

HYPRE_Int
hypre_SeqCSRMatvecDevice(HYPRE_Int nrows, HYPRE_Int nnz,
                         HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a,
                         HYPRE_Complex *d_x, HYPRE_Complex *d_y)
{
   const HYPRE_Int rownnz = (nnz + nrows - 1) / nrows;
   const HYPRE_Int bDim = BLOCKDIM;

   if (rownnz >= 64)
   {
      const HYPRE_Int group_size = 32;
      const HYPRE_Int num_groups_per_block = BLOCKDIM / group_size;
      const HYPRE_Int gDim = (nrows + num_groups_per_block - 1) / num_groups_per_block;
      //printf("  Number of Threads <%d*%d>\n",gDim,bDim);
      csr_v_k_shuffle<group_size, HYPRE_Real> <<<gDim, bDim>>>(nrows, d_ia, d_ja, d_a, d_x, d_y);
   }
   else if (rownnz >= 32)
   {
      const HYPRE_Int group_size = 16;
      const HYPRE_Int num_groups_per_block = BLOCKDIM / group_size;
      const HYPRE_Int gDim = (nrows + num_groups_per_block - 1) / num_groups_per_block;
      //printf("  Number of Threads <%d*%d>\n",gDim,bDim);
      csr_v_k_shuffle<group_size, HYPRE_Real> <<<gDim, bDim>>>(nrows, d_ia, d_ja, d_a, d_x, d_y);
   }
   else if (rownnz >= 16)
   {
      const HYPRE_Int group_size = 8;
      const HYPRE_Int num_groups_per_block = BLOCKDIM / group_size;
      const HYPRE_Int gDim = (nrows + num_groups_per_block - 1) / num_groups_per_block;
      //printf("  Number of Threads <%d*%d>\n",gDim,bDim);
      csr_v_k_shuffle<group_size, HYPRE_Real> <<<gDim, bDim>>>(nrows, d_ia, d_ja, d_a, d_x, d_y);
   }
   else if (rownnz >= 8)
   {
      const HYPRE_Int group_size = 4;
      const HYPRE_Int num_groups_per_block = BLOCKDIM / group_size;
      const HYPRE_Int gDim = (nrows + num_groups_per_block - 1) / num_groups_per_block;
      //printf("  Number of Threads <%d*%d>\n",gDim,bDim);
      csr_v_k_shuffle<group_size, HYPRE_Real> <<<gDim, bDim>>>(nrows, d_ia, d_ja, d_a, d_x, d_y);
   }
   else
   {
      const HYPRE_Int group_size = 4;
      const HYPRE_Int num_groups_per_block = BLOCKDIM / group_size;
      const HYPRE_Int gDim = (nrows + num_groups_per_block - 1) / num_groups_per_block;
      //printf("  Number of Threads <%d*%d>\n",gDim,bDim);
      csr_v_k_shuffle<group_size, HYPRE_Real> <<<gDim, bDim>>>(nrows, d_ia, d_ja, d_a, d_x, d_y);
   }

   return 0;
}

void spmv_csr_vector(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y)
{
   HYPRE_Int *d_ia, *d_ja, i;
   HYPRE_Real *d_a, *d_x, *d_y;
   HYPRE_Int n = csr->num_rows;
   HYPRE_Int nnz = csr->num_nonzeros;
   hypre_double t1,t2;
   /*---------- Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(HYPRE_Int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(HYPRE_Int));
   cudaMalloc((void **)&d_a, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_x, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_y, n*sizeof(HYPRE_Real));
   /*---------- Memcpy */
   cudaMemcpy(d_ia, csr->i, (n+1)*sizeof(HYPRE_Int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->j, nnz*sizeof(HYPRE_Int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_a, csr->data, nnz*sizeof(HYPRE_Real),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(HYPRE_Real),
   cudaMemcpyHostToDevice);
   /*-------- set spmv kernel */
   /*-------- num of half-warps per block */
   //printf("CSR<<<%4d, %3d>>>  ",gDim,bDim);
   /*-------- start timing */
   t1 = wall_timer();
   for (i=0; i<REPEAT; i++)
   {
      //cudaMemset((void *)d_y, 0, n*sizeof(HYPRE_Real));
      hypre_SeqCSRMatvecDevice(n, nnz, d_ia, d_ja, d_a, d_x, d_y);
   }
   /*-------- Barrier for GPU calls */
   cudaThreadSynchronize();
   /*-------- stop timing */
   t2 = wall_timer()-t1;
/*--------------------------------------------------*/
   printf("\n=== [GPU] CSR-vector Kernel ===\n");
   printf("  %.2f ms, %.2f GFLOPS, ",
   t2*1e3/REPEAT,2*nnz/t2/1e9*REPEAT);
/*-------- copy y to host mem */
   cudaMemcpy(y, d_y, n*sizeof(HYPRE_Real),
   cudaMemcpyDeviceToHost);
/*---------- CUDA free */
   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_a);
   cudaFree(d_x);
   cudaFree(d_y);
}

/*-----------------------------------------------*/
void cuda_init(HYPRE_Int argc, char **argv)
{
   HYPRE_Int deviceCount, dev;
   cudaGetDeviceCount(&deviceCount);
   printf("=========================================\n");
   if (deviceCount == 0)
   {
      printf("There is no device supporting CUDA\n");
   }
   for (dev = 0; dev < deviceCount; ++dev)
   {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      if (dev == 0)
      {
         if (deviceProp.major == 9999 && deviceProp.minor == 9999)
         {
            printf("There is no device supporting CUDA.\n");
         }
         else if (deviceCount == 1)
         {
            printf("There is 1 device supporting CUDA\n");
         }
         else
         {
            printf("There are %d devices supporting CUDA\n", deviceCount);
         }
      }
   printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
   printf("  Major revision number:          %d\n",deviceProp.major);
   printf("  Minor revision number:          %d\n",deviceProp.minor);
   printf("  Total amount of global memory:  %.2f GB\n",deviceProp.totalGlobalMem/1e9);
   }
   dev = 0;
   cudaSetDevice(dev);
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, dev);
   printf("\nRunning on Device %d: \"%s\"\n", dev, deviceProp.name);
   printf("=========================================\n");
}

/*---------------------------------------------------*/
void cuda_check_err()
{
   cudaError_t cudaerr = cudaGetLastError() ;
   if (cudaerr != cudaSuccess)
   {
       printf("error: %s\n",cudaGetErrorString(cudaerr));
   }
}

void spmv_cusparse_csr(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y)
{
   HYPRE_Int n = csr->num_rows;
   HYPRE_Int nnz = csr->num_nonzeros;
   HYPRE_Int *d_ia, *d_ja, i;
   HYPRE_Real *d_a, *d_x, *d_y;
   hypre_double t1, t2;
   HYPRE_Real done = 1.0, dzero = 0.0;
   /*------------------- allocate Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(HYPRE_Int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(HYPRE_Int));
   cudaMalloc((void **)&d_a, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_x, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_y, n*sizeof(HYPRE_Real));
   /*------------------- Memcpy */
   cudaMemcpy(d_ia, csr->i, (n+1)*sizeof(HYPRE_Int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->j, nnz*sizeof(HYPRE_Int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_a, csr->data, nnz*sizeof(HYPRE_Real),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(HYPRE_Real),
   cudaMemcpyHostToDevice);
   /*-------------------- cusparse library*/
   cusparseStatus_t status;
   cusparseHandle_t handle=0;
   cusparseMatDescr_t descr=0;

   /* initialize cusparse library */
   status= cusparseCreate(&handle);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
     printf("CUSPARSE Library initialization failed\n");
     exit(1);
   }
   /* create and setup matrix descriptor */
   status= cusparseCreateMatDescr(&descr);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
     printf("Matrix descriptor initialization failed\n");
     exit(1);
   }
   cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
   /*-------- start timing */
   t1 = wall_timer();
   if (sizeof(HYPRE_Real) == sizeof(hypre_double))
   {
      for (i=0; i<REPEAT; i++)
      {
         status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (hypre_double *) &done, descr, (hypre_double *) d_a,
                                 d_ia, d_ja, (hypre_double *) d_x,
                                 (hypre_double *) &dzero, (hypre_double *) d_y);
      }
   }
   else if (sizeof(HYPRE_Real) == sizeof(float))
   {
      for (i=0; i<REPEAT; i++)
      {
         status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (float *) &done, descr, (float *) d_a,
                                 d_ia, d_ja, (float *) d_x,
                                 (float *) &dzero, (float *) d_y);
      }
   }
   /*-------- barrier for GPU calls */
   cudaThreadSynchronize();
   /*-------- stop timing */
   t2 = wall_timer()-t1;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix-vector multiplication failed\n");
      exit(1);
   }
   /*--------------------------------------------------*/
   printf("\n=== [GPU] CUSPARSE CSR Kernel ===\n");
   printf("  %.2f ms, %.2f GFLOPS, ",
   t2*1e3/REPEAT, 2*nnz/t2/1e9*REPEAT);
   /*-------- copy y to host mem */
   cudaMemcpy(y, d_y, n*sizeof(HYPRE_Real),
   cudaMemcpyDeviceToHost);
   /*--------- CUDA free */
   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_a);
   cudaFree(d_x);
   cudaFree(d_y);
   /* destroy matrix descriptor */
   status = cusparseDestroyMatDescr(descr);
   descr = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor destruction failed\n");
      exit(1);
   }
   /* destroy handle */
   status = cusparseDestroy(handle);
   handle = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("CUSPARSE Library release of resources failed\n");
      exit(1);
   }
}
