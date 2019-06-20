#include "spmv.h"
#include <cuda_runtime.h>
#include "cusparse.h"


#if 0
__global__
void csr_v_k_8(int n, int *d_ia, int *d_ja, REAL *d_a, REAL *d_x, REAL *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *  shared memory reduction, texture memory fetching
    *           1/4-Warp (8 threads) per row
    *------------------------------------------------------------*/
   // num of 1/4-warps
   int nqw = gridDim.x*BLOCKDIM/8;
   // 1/4 warp id
   int qwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/8;
   // thread lane in each half warp
   int lane = threadIdx.x & (8-1);
   // 1/4 warp lane in each block
   int qwlane = threadIdx.x/8;
   // shared memory for patial result
   volatile __shared__ REAL r[BLOCKDIM+8];
   volatile __shared__ int startend[BLOCKDIM/8][2];
   for (int row = qwid; row < n; row += nqw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[qwlane][lane] = d_ia[row+lane];
      }
      int p = startend[qwlane][0];
      int q = startend[qwlane][1];
      REAL sum = 0.0;
      for (int i=p+lane; i<q; i+=8)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
      // parallel reduction
      r[threadIdx.x] = sum;
      r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+1];
      if (lane == 0)
      d_y[row] = r[threadIdx.x];
   }
}


__global__
void csr_v_k_8_shuffle(int n, int *d_ia, int *d_ja, REAL *d_a, REAL *d_x, REAL *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *  shared memory reduction, texture memory fetching
    *           1/4-Warp (8 threads) per row (shuffle reduction version)
    *------------------------------------------------------------*/
   // num of 1/4-warps
   int nqw = gridDim.x*BLOCKDIM/8;
   // 1/4 warp id
   int qwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/8;
   // thread lane in each half warp
   int lane = threadIdx.x & (8-1);
   // 1/4 warp lane in each block
   int qwlane = threadIdx.x/8;
   // shared memory for patial result
   volatile __shared__ REAL r[BLOCKDIM+8];
   volatile __shared__ int startend[BLOCKDIM/8][2];
   for (int row = qwid; row < n; row += nqw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[qwlane][lane] = d_ia[row+lane];
      }
      int p = startend[qwlane][0];
      int q = startend[qwlane][1];
      REAL sum = 0.0;
      for (int i=p+lane; i<q; i+=8)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
// parallel reduction
#pragma unroll
      for (int d = 8/2; d > 0; d >>= 1)
      {
         r[threadIdx.x] = sum += __shfl_down(sum, d);
      }
      if (lane == 0)
      {
         d_y[row] = r[threadIdx.x];
      }
   }
}




__global__
void csr_v_k_16(int n, int *d_ia, int *d_ja, REAL *d_a, REAL *d_x, REAL *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *              shared memory reduction
    *           Half-Warp (16 threads) per row
    *------------------------------------------------------------*/
   // num of half-warps
   int nhw = gridDim.x*BLOCKDIM/HALFWARP;
   // half warp id
   int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
   // thread lane in each half warp
   int lane = threadIdx.x & (HALFWARP-1);
   // half warp lane in each block
   int hwlane = threadIdx.x/HALFWARP;
   // shared memory for patial result
   volatile __shared__ REAL r[BLOCKDIM+8];
   volatile __shared__ int startend[BLOCKDIM/HALFWARP][2];
   for (int row = hwid; row < n; row += nhw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[hwlane][lane] = d_ia[row+lane];
      }
      int p = startend[hwlane][0];
      int q = startend[hwlane][1];
      REAL sum = 0.0;
      for (int i=p+lane; i<q; i+=HALFWARP)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
      // parallel reduction
      r[threadIdx.x] = sum;
      r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+1];
      if (lane == 0)
      {
         d_y[row] = r[threadIdx.x];
      }
   }
}



__global__
void csr_v_k_16_shuffle(int n, int *d_ia, int *d_ja, REAL *d_a, REAL *d_x, REAL *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *               shared memory reduction
    *           Half-Warp (16 threads) per row(shuffle version)
    *------------------------------------------------------------*/
   // num of half-warps
   int nhw = gridDim.x*BLOCKDIM/HALFWARP;
   // half warp id
   int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
   // thread lane in each half warp
   int lane = threadIdx.x & (HALFWARP-1);
   // half warp lane in each block
   int hwlane = threadIdx.x/HALFWARP;
   // shared memory for patial result
   volatile __shared__ int startend[BLOCKDIM/HALFWARP][2];
   for (int row = hwid; row < n; row += nhw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[hwlane][lane] = d_ia[row+lane];
      }
      int p = startend[hwlane][0];
      int q = startend[hwlane][1];
      REAL sum = 0.0;
      for (int i=p+lane; i<q; i+=HALFWARP)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
// parallel reduction
#pragma unroll
      for (int d = HALFWARP/2; d > 0; d >>= 1)
      {
         sum += __shfl_down(sum, d);
      }
      if (lane == 0)
      {
         d_y[row] = sum;
      }
   }
}

__global__
void csr_v_k_32(int n, int *d_ia, int *d_ja, REAL *d_a, REAL *d_x, REAL *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *              shared memory reduction
    *          FULL-Warp (32 threads) per row
    *------------------------------------------------------------*/
   // num of full-warps
   int nw = gridDim.x*BLOCKDIM/WARP;
   // full warp id
   int wid = (blockIdx.x*BLOCKDIM+threadIdx.x)/WARP;
   // thread lane in each full warp
   int lane = threadIdx.x & (WARP-1);
   // full warp lane in each block
   int wlane = threadIdx.x/WARP;
   // shared memory for patial result
   volatile __shared__ REAL r[BLOCKDIM+16];
   volatile __shared__ int startend[BLOCKDIM/WARP][2];
   for (int row = wid; row < n; row += nw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[wlane][lane] = d_ia[row+lane];
      }
      int p = startend[wlane][0];
      int q = startend[wlane][1];
      REAL sum = 0.0;
      for (int i=p+lane; i<q; i+=WARP)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
      // parallel reduction
      r[threadIdx.x] = sum;
      r[threadIdx.x] = sum = sum + r[threadIdx.x+16];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+1];
      if (lane == 0)
      {
         d_y[row] = r[threadIdx.x];
      }
   }
}


__global__
void csr_v_k_32_shuffle(int n, int *d_ia, int *d_ja, REAL *d_a, REAL *d_x, REAL *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *              shared memory reduction
    *          FULL-Warp (32 threads) per row(shuffle version)
    *------------------------------------------------------------*/
   // num of full-warps
   int nw = gridDim.x*BLOCKDIM/WARP;
   // full warp id
   int wid = (blockIdx.x*BLOCKDIM+threadIdx.x)/WARP;
   // thread lane in each full warp
   int lane = threadIdx.x & (WARP-1);
   // full warp lane in each block
   int wlane = threadIdx.x/WARP;
   // shared memory for patial result
   volatile __shared__ REAL r[BLOCKDIM+16];
   volatile __shared__ int startend[BLOCKDIM/WARP][2];
   for (int row = wid; row < n; row += nw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[wlane][lane] = d_ia[row+lane];
      }
      int p = startend[wlane][0];
      int q = startend[wlane][1];
      REAL sum = 0.0;
      for (int i=p+lane; i<q; i+=WARP)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
      // parallel reduction
      #pragma unroll
      for (int d = WARP/2; d > 0; d >>= 1)
      {
      r[threadIdx.x] = sum += __shfl_down(sum, d);
      }
      if (lane == 0)
      {
         d_y[row] = r[threadIdx.x];
      }
   }
}

#endif

template <int K, typename T>
__global__
void csr_v_k_shared(int n, int *d_ia, int *d_ja, T *d_a, T *d_x, T *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *              shared memory reduction
    *              K threads-Warp  per row
    *------------------------------------------------------------*/
   // num of full-warps
   int nw = gridDim.x*BLOCKDIM/K;
   // full warp id
   int wid = (blockIdx.x*BLOCKDIM+threadIdx.x)/K;
   // thread lane in each full warp
   int lane = threadIdx.x & (K-1);
   // full warp lane in each block
   int wlane = threadIdx.x/K;
   // shared memory for patial result
   volatile __shared__ T r[BLOCKDIM+K/2];
   volatile __shared__ int startend[BLOCKDIM/K][2];
   for (int row = wid; row < n; row += nw)
   {
      // row start and end point
      if (lane < 2)
      {
         startend[wlane][lane] = d_ia[row+lane];
      }
      int p = startend[wlane][0];
      int q = startend[wlane][1];
      T sum = 0.0;
      for (int i=p+lane; i<q; i+=K)
      {
         sum += d_a[i] * d_x[d_ja[i]];
      }
      // parallel reduction
      r[threadIdx.x] = sum;
#pragma unroll
      for (int d = K/2; d > 0; d >>= 1)
      {
         r[threadIdx.x] = sum = sum + r[threadIdx.x+d];
      }
      if (lane == 0)
      {
         d_y[row] = r[threadIdx.x];
      }
   }
}


/* K is the number of threads working on a single row. K = 2, 4, 8, 16, 32 */
template <int K, typename T>
__global__
void csr_v_k_shuffle(int n, int *d_ia, int *d_ja, T *d_a, T *d_x, T *d_y)
{
   /*------------------------------------------------------------*
    *               CSR spmv-vector kernel
    *              shared memory reduction
    *           (Group of K threads) per row
    *------------------------------------------------------------*/
   int nw = gridDim.x * (BLOCKDIM / K);
   int wid = (blockIdx.x * BLOCKDIM + threadIdx.x) / K;
   int lane = threadIdx.x & (K - 1);
   for (int row = wid; row < n; row += nw)
   {
      int j, p, q;
      if (lane < 2)
      {
         j = read_only_load(&d_ia[row+lane]);
      }
      p = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 0, K);
      q = __shfl_sync(HYPRE_WARP_FULL_MASK, j, 1, K);
      T sum = 0.0;
#if 0
      for (int i = p + lane; i < q; i += K*2)
      {
         sum += read_only_load(&d_a[i]) * read_only_load(&d_x[read_only_load(&d_ja[i])]);
         if (i + K < q)
         {
            sum += read_only_load(&d_a[i+K]) * read_only_load(&d_x[read_only_load(&d_ja[i+K])]);
         }
      }
#else
      for (int i = p + lane; i < q; i += K)
      {
         sum += read_only_load(&d_a[i]) * read_only_load(&d_x[read_only_load(&d_ja[i])]);
      }
#endif
      // parallel reduction
#pragma unroll
      for (int d = K/2; d > 0; d >>= 1)
      {
         sum += __shfl_down_sync(HYPRE_WARP_FULL_MASK, sum, d);
      }
      if (lane == 0)
      {
         d_y[row] = sum;
      }
   }
}

//example

void spmv_csr_vector(struct csr_t *csr, REAL *x, REAL *y)
{
   int *d_ia, *d_ja, i;
   REAL *d_a, *d_x, *d_y;
   int n = csr->nrows;
   int nnz = csr->ia[n];
   double t1,t2;
   /*---------- Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(int));
   cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
   cudaMalloc((void **)&d_x, n*sizeof(REAL));
   cudaMalloc((void **)&d_y, n*sizeof(REAL));
   /*---------- Memcpy */
   cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(REAL),
   cudaMemcpyHostToDevice);
   /*-------- set spmv kernel */
   /*-------- num of half-warps per block */
   int hwb = BLOCKDIM/HALFWARP;
   int gDim = min(MAXTHREADS/BLOCKDIM, (n+hwb-1)/hwb);
   int bDim = BLOCKDIM;
   //printf("CSR<<<%4d, %3d>>>  ",gDim,bDim);
   /*-------- start timing */
   t1 = wall_timer();
   for (i=0; i<REPEAT; i++)
   {
      //cudaMemset((void *)d_y, 0, n*sizeof(REAL));
      csr_v_k_shuffle<16, REAL> <<<gDim, bDim>>>(n, d_ia, d_ja, d_a, d_x, d_y);
   }
   /*-------- Barrier for GPU calls */
   cudaThreadSynchronize();
   /*-------- stop timing */
   t2 = wall_timer()-t1;
/*--------------------------------------------------*/
   printf("\n=== [GPU] CSR-vector Kernel ===\n");
   printf("  Number of Threads <%d*%d>\n",gDim,bDim);
   printf("  %.2f ms, %.2f GFLOPS, ",
   t2*1e3,2*nnz/t2/1e9*REPEAT);
/*-------- copy y to host mem */
   cudaMemcpy(y, d_y, n*sizeof(REAL),
   cudaMemcpyDeviceToHost);
/*---------- CUDA free */
   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_a);
   cudaFree(d_x);
   cudaFree(d_y);
}

/*-----------------------------------------------*/
void cuda_init(int argc, char **argv)
{
   int deviceCount, dev;
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

void spmv_cusparse_csr(struct csr_t *csr, REAL *x, REAL *y)
{
   int n = csr->nrows;
   int nnz = csr->ia[n];
   int *d_ia, *d_ja, i;
   REAL *d_a, *d_x, *d_y;
   double t1, t2;
   REAL done = 1.0, dzero = 0.0;
   /*------------------- allocate Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(int));
   cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
   cudaMalloc((void **)&d_x, n*sizeof(REAL));
   cudaMalloc((void **)&d_y, n*sizeof(REAL));
   /*------------------- Memcpy */
   cudaMemcpy(d_ia, csr->ia, (n+1)*sizeof(int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->ja, nnz*sizeof(int),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_a, csr->a, nnz*sizeof(REAL),
   cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(REAL),
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
   for (i=0; i<REPEAT; i++)
   {
#if DOUBLEPRECISION
      status= cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
      &done, descr, d_a, d_ia, d_ja,
      d_x, &dzero, d_y);
#else
      status= cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
      &done, descr, d_a, d_ia, d_ja,
      d_x, &dzero, d_y);
#endif
      if (status != CUSPARSE_STATUS_SUCCESS)
      {
         printf("Matrix-vector multiplication failed\n");
         exit(1);
      }
   }
   /*-------- barrier for GPU calls */
   cudaThreadSynchronize();
   /*-------- stop timing */
   t2 = wall_timer()-t1;
   /*--------------------------------------------------*/
   printf("\n=== [GPU] CUSPARSE CSR Kernel ===\n");
   printf("  %.2f ms, %.2f GFLOPS, ",
   t2*1e3,2*nnz/t2/1e9*REPEAT);
   /*-------- copy y to host mem */
   cudaMemcpy(y, d_y, n*sizeof(REAL),
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

