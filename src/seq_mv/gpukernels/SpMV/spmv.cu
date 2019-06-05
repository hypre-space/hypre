#include "spmv.h"
#include <cuda_runtime.h>
#include "cusparse.h"

#if DOUBLEPRECISION
texture<int2, 1> texRef;
#else
texture<REAL, 1> texRef;
#endif


__global__ 
void csr_v_k(int n, int *d_ia, int *d_ja, REAL *d_a, REAL *d_y) {
/*------------------------------------------------------------*
 *               CSR spmv-vector kernel
 *  shared memory reduction, texture memory fetching
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
  for (int row = hwid; row < n; row += nhw) {
    // row start and end point
    if (lane < 2)
      startend[hwlane][lane] = d_ia[row+lane];
    int p = startend[hwlane][0];
    int q = startend[hwlane][1];
    REAL sum = 0.0;
    for (int i=p+lane; i<q; i+=HALFWARP) {
#if DOUBLEPRECISION
      int2 t = tex1Dfetch(texRef, d_ja[i-1]-1);
      sum += d_a[i-1] * __hiloint2double(t.y, t.x);
#else
      REAL t = tex1Dfetch(texRef, d_ja[i-1]-1);
      sum += d_a[i-1] * t;
#endif
    }
    // parallel reduction
    r[threadIdx.x] = sum;
    r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+1];
    if (lane == 0)
      d_y[row] = r[threadIdx.x];
  }
}

/*-------------------------------------------------------*/
void spmv_csr_vector(struct csr_t *csr, REAL *x, REAL *y) {
  int *d_ia, *d_ja, i;
  REAL *d_a, *d_x, *d_y;
  int n = csr->n;
  int nnz = csr->nnz; 
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
/*--------- texture binding */
  size_t offset;
#if DOUBLEPRECISION
  cudaBindTexture(&offset, texRef, d_x, n*sizeof(int2));
#else
  cudaBindTexture(&offset, texRef, d_x, n*sizeof(float));
#endif
  assert(offset == 0);
/*-------- set spmv kernel */
/*-------- num of half-warps per block */
  int hwb = BLOCKDIM/HALFWARP;
  int gDim = min(MAXTHREADS/BLOCKDIM, (n+hwb-1)/hwb);
  int bDim = BLOCKDIM;
  //printf("CSR<<<%4d, %3d>>>  ",gDim,bDim);
/*-------- start timing */
  t1 = wall_timer();
  for (i=0; i<REPEAT; i++) {
    //cudaMemset((void *)d_y, 0, n*sizeof(REAL));
    csr_v_k<<<gDim, bDim>>>(n, d_ia, d_ja, d_a, d_y);
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
/*--------- unbind texture */
  cudaUnbindTexture(texRef);
/*---------- CUDA free */
  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
}


/*----------------------------------------------*/
/*             THE JAD FORMAT                   */
/*----------------------------------------------*/
template <bool flag>
__global__ 
void jad_k(int n, int njad, int *d_ia, int *d_ja, 
           REAL *d_a, REAL *d_y) {
/*-----------------------------------------------*
 *  JAD SpMatVec Kernel(texture memory fetching)
 *            one thread per row
 *-----------------------------------------------*/
  int i,j,p,q;
  REAL r;
/*------------ each thread per row */
  int row = blockIdx.x*blockDim.x+threadIdx.x;
/*------------ number of threads */
  int nthreads = gridDim.x * blockDim.x;
  __shared__ int shia[BLOCKDIM];
  if (threadIdx.x <= njad)
    shia[threadIdx.x] = d_ia[threadIdx.x];
  __syncthreads(); 
  while (row < n) {
    r = 0.0;
    p = shia[0];
    q = shia[1];
    i=0;
    while ( ((p+row) < q) && (i < njad) ) {
      j = p+row;
#if DOUBLEPRECISION
/*--------- double precision texture fetching */
      int2 t = tex1Dfetch(texRef, d_ja[j-1]-1);
      r += d_a[j-1] * __hiloint2double(t.y, t.x);
#else
/*--------- single precision texture fetching */
      r += d_a[j-1] * tex1Dfetch(texRef, d_ja[j-1]-1);
#endif
      i++;
      if (i<njad) {
        p = q;
        q = shia[i+1];
      }
    }
    if (flag)
      d_y[row] += r;
    else
      d_y[row] = r;

    row += nthreads;
  }
}

__global__ 
void vperm_k(int n, REAL *d_y, REAL *d_x, int *d_p) {
/*------------------------------------------------*/
/*   vector permutation, y[p(i)] := x[i]          */
/*------------------------------------------------*/
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  int i;
  for (i=idx; i<n; i+=nthreads)
    d_y[d_p[i]-1] = d_x[i];
}

void spmv_jad(struct jad_t *jad, REAL *x, REAL *y) {
  int n = jad->n;
  int nnz = jad->nnz; 
  int *d_ia, *d_ja, *d_p, i, j;
  REAL *d_a, *d_x, *d_y, *d_y2;
  double t1,t2;
/*------------------- allocate Device Memory */
  cudaMalloc((void **)&d_ia, (jad->njad+1)*sizeof(int));
  cudaMalloc((void **)&d_ja, nnz*sizeof(int));
  cudaMalloc((void **)&d_a, nnz*sizeof(REAL));
  cudaMalloc((void **)&d_x, n*sizeof(REAL));
  cudaMalloc((void **)&d_y, n*sizeof(REAL));
  cudaMalloc((void **)&d_y2, n*sizeof(REAL));
  cudaMalloc((void **)&d_p, n*sizeof(int));
/*------------------- Memcpy */
  cudaMemcpy(d_ia, jad->ia, (jad->njad+1)*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_ja, jad->ja, nnz*sizeof(int),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, jad->a, nnz*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, n*sizeof(REAL),
  cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, jad->perm, n*sizeof(int),
  cudaMemcpyHostToDevice);
/*--------- texture binding */
  size_t offset;
#if DOUBLEPRECISION
  cudaBindTexture(&offset, texRef, d_x, n*sizeof(int2));
#else
  cudaBindTexture(&offset, texRef, d_x, n*sizeof(float));
#endif
  assert(offset == 0);
/*--------- set spmv kernel */
  int nthreads = min(MAXTHREADS, n);
  int gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
  int bDim = BLOCKDIM;
/*-------- start timing */
  t1 = wall_timer();
  for (i=0; i<REPEAT; i++) {
/*-------- handle (BLOCKDIM-1) jads each time */
    int njad = min(jad->njad, (BLOCKDIM-1));
    jad_k<0> <<<gDim, bDim>>> (n, njad, d_ia, d_ja, d_a, d_y);
    for (j=BLOCKDIM-1; j<jad->njad; j+=(BLOCKDIM-1)) {
      njad = min((jad->njad-j), (BLOCKDIM-1));
      jad_k<1> <<<gDim, bDim>>> (n, njad, d_ia+j, d_ja, d_a, d_y);
    }
  }
/*-------- permutation */
  vperm_k<<<gDim, bDim>>>(n, d_y2, d_y, d_p);
/*-------- barrier for GPU calls */
  cudaThreadSynchronize();
/*-------- stop timing */
  t2 = wall_timer()-t1;
/*--------------------------------------------------*/
  printf("\n=== [GPU] JAD (njad %d) Kernel ===\n", jad->njad);
  printf("  Number of Threads <%d*%d>\n",gDim,bDim);
  printf("  %.2f ms, %.2f GFLOPS, ",
  t2*1e3,2*nnz/t2/1e9*REPEAT);
/*-------- copy y to host mem */
  cudaMemcpy(y, d_y2, n*sizeof(REAL), 
  cudaMemcpyDeviceToHost);
/*--------- unbind texture */
  cudaUnbindTexture(texRef);
/*--------- CUDA free */
  cudaFree(d_ia);
  cudaFree(d_ja);
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_y2);
  cudaFree(d_p);
}

/*-----------------------------------------------*/
void cuda_init(int argc, char **argv) {
  int deviceCount, dev;
  cudaGetDeviceCount(&deviceCount);
  printf("=========================================\n");
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        printf("There is no device supporting CUDA.\n");
      else if (deviceCount == 1)
        printf("There is 1 device supporting CUDA\n");
      else
        printf("There are %d devices supporting CUDA\n", deviceCount);
    }
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    printf("  Major revision number:          %d\n",
           deviceProp.major);
    printf("  Minor revision number:          %d\n",
           deviceProp.minor);
    printf("  Total amount of global memory:  %.2f GB\n",
           deviceProp.totalGlobalMem/1e9);
  }
  dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\nRunning on Device %d: \"%s\"\n", dev, deviceProp.name);
  printf("=========================================\n");
}

/*---------------------------------------------------*/
void cuda_check_err() {
  cudaError_t cudaerr = cudaGetLastError() ;
  if (cudaerr != cudaSuccess) 
    printf("error: %s\n",cudaGetErrorString(cudaerr));
}

void spmv_cusparse_csr(struct csr_t *csr, REAL *x, REAL *y) {
  int n = csr->n;
  int nnz = csr->nnz; 
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
    
  cusparseStatus_t status;
  cusparseHandle_t handle=0;
  cusparseMatDescr_t descr=0;

  /* initialize cusparse library */
  status= cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library initialization failed\n");
     exit(1);
  }

  /* create and setup matrix descriptor */ 
  status= cusparseCreateMatDescr(&descr); 
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor initialization failed\n");
     exit(1);
  }       
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE); 

/*-------- start timing */
  t1 = wall_timer();
  for (i=0; i<REPEAT; i++) {
#if DOUBLEPRECISION
    status= cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                           &done, descr, d_a, d_ia, d_ja, 
                           d_x, &dzero, d_y);
#else    
    status= cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                           &done, descr, d_a, d_ia, d_ja, 
                           d_x, &dzero, d_y);
#endif
    if (status != CUSPARSE_STATUS_SUCCESS) {
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
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor destruction failed\n");
     exit(1);
  }    

  /* destroy handle */
  status = cusparseDestroy(handle);
  handle = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library release of resources failed\n");
     exit(1);
  }
}


void spmv_cusparse_hyb(struct csr_t *csr, REAL *x, REAL *y) {
  int n = csr->n;
  int nnz = csr->nnz; 
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
    
  cusparseStatus_t status;
  cusparseHandle_t handle=0;
  cusparseMatDescr_t descr=0;

  /* initialize cusparse library */
  status= cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library initialization failed\n");
     exit(1);
  }

  /* create and setup matrix descriptor */ 
  status= cusparseCreateMatDescr(&descr); 
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor initialization failed\n");
     exit(1);
  }       
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE); 
  
  cusparseHybMat_t hyb;
  status = cusparseCreateHybMat(&hyb);

  t1 = wall_timer();
#if DOUBLEPRECISION
  cusparseDcsr2hyb(handle, n, n, descr, d_a, d_ia, d_ja,
                   hyb, 20, CUSPARSE_HYB_PARTITION_AUTO);
#else    
  cusparseScsr2hyb(handle, n, n, descr, d_a, d_ia, d_ja,
                   hyb, 20, CUSPARSE_HYB_PARTITION_AUTO);
#endif
  //Barrier for GPU calls
  cudaThreadSynchronize();
  t1 = wall_timer() - t1;
  fprintf(stdout, "HYB conversion time %f\n", t1);

/*-------- start timing */
  t1 = wall_timer();
  for (i=0; i<REPEAT; i++) {
#if DOUBLEPRECISION
    status = cusparseDhybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &done, descr, hyb, d_x, &dzero, d_y);

#else    
    status = cusparseShybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &done, descr, hyb, d_x, &dzero, d_y);
#endif
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("Matrix-vector multiplication failed\n");
      exit(1);
    }
  }
/*-------- barrier for GPU calls */
  cudaThreadSynchronize();
/*-------- stop timing */
  t2 = wall_timer()-t1;
/*--------------------------------------------------*/
  printf("\n=== [GPU] CUSPARSE HYB Kernel ===\n");
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
  cusparseDestroyHybMat(hyb);
 
  /* destroy matrix descriptor */ 
  status = cusparseDestroyMatDescr(descr); 
  descr = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("Matrix descriptor destruction failed\n");
     exit(1);
  }    

  /* destroy handle */
  status = cusparseDestroy(handle);
  handle = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
     printf("CUSPARSE Library release of resources failed\n");
     exit(1);
  }
}
