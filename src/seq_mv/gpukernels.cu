#include <stdio.h>
#include "hypre_nvtx.h"
extern "C"{
void MemPrefetch(const void *ptr,int device,cudaStream_t stream);
}
#define gpuErrchk2(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     printf("GPUassert2: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}


__global__
void VecScaleKernelGSL(double *__restrict__ u, double* __restrict__ v, double* __restrict__ l1_norm, int num_rows){
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_rows; 	 
       i += blockDim.x * gridDim.x) {
    u[i]+=v[i]/l1_norm[i];
  }

}
__global__
void dummy(double *a, double *b,double *c, int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<10) printf("Hello world %d %lf %lf %lf\n",num_rows,a[0],b[0],c[0]);
}

__global__
void PrintDeviceArrayKernel(double *a,int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<num_rows) printf("PrintARRAYDEVICE %d %lf\n",i,a[i]);
}
  


extern "C"{
__global__
void VecScaleKernelText(double *__restrict__ u, const double *__restrict__ v, const double *__restrict__ l1_norm, int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    //double x=__ldg(v+i);
    //double y =__ldg(l1_norm+i);
    u[i]+=__ldg(v+i)/__ldg(l1_norm+i);
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}
extern "C"{
__global__
void VecScaleKernel(double *__restrict__ u, const double *__restrict__ v, const double * __restrict__ l1_norm, int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    u[i]+=v[i]/l1_norm[i];
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}

extern "C"{
  void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s){
    PUSH_RANGE_PAYLOAD("VECSCALE",1,num_rows);
    const int tpb=64;
    int num_blocks=num_rows/tpb+1;
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    //MemPrefetch(u,0,s);
    //MemPrefetch(v,0,s);
    MemPrefetch(l1_norm,0,s);
    VecScaleKernel<<<num_blocks,tpb,0,s>>>(u,v,l1_norm,num_rows);
    //dummy<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    gpuErrchk2(cudaStreamSynchronize(s));
    POP_RANGE;
  }
}

extern "C"{
__global__
void VecScaleKernelA(double *__restrict__ u, const double *__restrict__ v, const double * __restrict__ l1_norm, int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
    u[i]+=v[i]/l1_norm[i];
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
}
__global__  
void VecScaleKernelB(double *__restrict__ u, const double *__restrict__ v, const double * __restrict__ l1_norm, int num_rows,int offset){
  int i = offset+blockIdx.x * blockDim.x + threadIdx.x;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    u[i]+=v[i]/l1_norm[i];
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}

extern "C"{
  void VecScaleSplit(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s){
    PUSH_RANGE("VECSCALE",1);
    const int tpb=1024;
    int num_blocks=num_rows/tpb;
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    MemPrefetch(u,0,s);
    MemPrefetch(v,0,s);
    MemPrefetch(l1_norm,0,s);
    VecScaleKernelA<<<num_blocks,tpb,0,s>>>(u,v,l1_norm,num_rows);
    VecScaleKernelB<<<1,tpb,0,s>>>(u,v,l1_norm,num_rows,num_blocks*tpb);
    //dummy<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    gpuErrchk2(cudaStreamSynchronize(s));
    POP_RANGE;
  }
}



extern "C"{
  void VecScaleGSL(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s){
    //int num_blocks=num_rows/32+1;
    //printf("Vecscale %d %d = %d \n",num_blocks,num_rows,num_blocks*32);
    VecScaleKernelGSL<<<1024,32,0,s>>>(u,v,l1_norm,num_rows);
  }
}

extern "C"{
  void PrintDeviceVec(double *u, int num_rows,cudaStream_t s){
    int num_blocks=num_rows/32+1;
    //printf("Vecscale in Kernale call %d %d = %d %d\n",num_blocks,num_rows,num_blocks*32,sizeof(int));
    //printf("ARG Pointers %p %p %p\n",u,v,l1_norm);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    PrintDeviceArrayKernel<<<num_blocks,32,0,s>>>(u,num_rows);
  }
}

// Mods that calculate the l1_norm locally

extern "C"{
__global__
void VecScaleKernelWithNorms1(double *__restrict__ u, double *__restrict__ v, double *__restrict__ l1_norm, 
			     int *A_diag_I,  double *A_diag_data, int *A_offd_I,double *A_offd_data,
			     int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double ll1_norm=0.0;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    int j;
    for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
      ll1_norm += fabs(A_diag_data[j]);
    for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
      ll1_norm += fabs(A_offd_data[j]);
    u[i]+=v[i]/ll1_norm;
    l1_norm[i]=ll1_norm;
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}
extern "C"{
__global__
void VecScaleKernelWithNorms2(double *__restrict__ u, double *__restrict__ v, double *__restrict__ l1_norm, 
			     int *A_diag_I,  double *A_diag_data, int *A_offd_I,double *A_offd_data,
			     int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double ll1_norm=0.0;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    int j;
    for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
      ll1_norm += fabs(A_diag_data[j]);
    //for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
    //  l1_norm += fabs(A_offd_data[j]);
    u[i]+=v[i]/ll1_norm;
    l1_norm[i]=ll1_norm;
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}
extern "C"{
  void VecScaleWithNorms(double *u, double *v, double *l1_norm, 
			 int *A_diag_I,  double *A_diag_data, int *A_offd_I,double *A_offd_data,
			 int num_rows,cudaStream_t s){
    int tpb=64;
    int num_blocks=num_rows/tpb+1;
    // Let cuda caluclate gridsize and block size
    if (0){
    int minGridSize,blockSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
					VecScaleKernelWithNorms1, 0, 0); 
    printf("Grid & Block size for max occupancy Norm1 %d(%d) %d(%d)\n",minGridSize,num_blocks,blockSize,tpb);
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      VecScaleKernelWithNorms2, 0, 0); 
    printf("Grid & Block size for max occupancy Norm2 %d(%d) %d(%d)\n",minGridSize,num_blocks,blockSize,tpb);
    }
    //printf("Vecscale in Kernale call %d %d = %d %d\n",num_blocks,num_rows,num_blocks*32,sizeof(int));
    //printf("ARG Pointers %p %p %p\n",u,v,l1_norm);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    if (A_offd_I)
      VecScaleKernelWithNorms1<<<num_blocks,tpb,0,s>>>(u,v,l1_norm,A_diag_I,A_diag_data,A_offd_I,A_offd_data,num_rows);
  else
    VecScaleKernelWithNorms2<<<num_blocks,tpb,0,s>>>(u,v,l1_norm,A_diag_I,A_diag_data,A_offd_I,A_offd_data,num_rows);
    //dummy<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
  }
}
extern "C"{

  __global__
  void VecCopyKernel(double* __restrict__ tgt, const double* __restrict__ src, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<size) tgt[i]=src[i];
}
  void VecCopy(double* tgt, const double* src, int size,cudaStream_t s){
    int tpb=64;
    int num_blocks=size/tpb+1;
    PUSH_RANGE_PAYLOAD("VecCopy",5,size);
    //MemPrefetch(tgt,0,s);
    //MemPrefetch(src,0,s);
    VecCopyKernel<<<num_blocks,tpb,0,s>>>(tgt,src,size);
    gpuErrchk2(cudaStreamSynchronize(s));
    POP_RANGE;
  }
}
extern "C"{

  __global__
  void VecSetKernel(double* __restrict__ tgt, const double value,int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<size) tgt[i]=value;
}
  void VecSet(double* tgt, int size, double value, cudaStream_t s){
    int tpb=64;
    //cudaDeviceSynchronize();
    MemPrefetch(tgt,0,s);
    int num_blocks=size/tpb+1;
    VecSetKernel<<<num_blocks,tpb,0,s>>>(tgt,value,size);
    cudaStreamSynchronize(s);
    //cudaDeviceSynchronize();
  }
}
extern "C"{
  __global__
  void  PackOnDeviceKernel(double* __restrict__ send_data,const double* __restrict__ x_local_data, const int* __restrict__ send_map, int begin,int end){
    int i = begin+blockIdx.x * blockDim.x + threadIdx.x;
    if (i<end){
      send_data[i-begin]=x_local_data[send_map[i]];
    }
  }
  void PackOnDevice(double *send_data,double *x_local_data, int *send_map, int begin,int end){
    int tpb=64;
    int num_blocks=(end-begin)/tpb+1;
    PackOnDeviceKernel<<<num_blocks,tpb,0,0>>>(send_data,x_local_data,send_map,begin,end);
    cudaStreamSynchronize(0);
  }
}
  
  // Scale vector by scalar

extern "C"{
__global__
void VecScaleScalarKernel(double *__restrict__ u, const double alpha ,int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    u[i]*=alpha;
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}
extern "C"{
  int VecScaleScalar(double *u, const double alpha,  int num_rows,cudaStream_t s){
    PUSH_RANGE("SEQVECSCALE",4);
    int num_blocks=num_rows/64+1;
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    VecScaleScalarKernel<<<num_blocks,64,0,s>>>(u,alpha,num_rows);
    //dummy<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    gpuErrchk2(cudaStreamSynchronize(s));
    POP_RANGE;
    return 0;
  }
}
extern "C"{
__global__
void SpMVCudaKernel(double* __restrict__ y,double alpha, const double* __restrict__ A_data, const int* __restrict__ A_i, const int* __restrict__ A_j, const double* __restrict__ x, double beta, int num_rows)
{
  int i= blockIdx.x * blockDim.x + threadIdx.x;
  if (i<num_rows){
    double temp = 0.0;
    int jj;
    for (jj = A_i[i]; jj < A_i[i+1]; jj++){
      int ajj=A_j[jj];
      temp += A_data[jj] * x[ajj];
    }
    y[i] =y[i]*beta+alpha*temp;
  }
}

__global__
void SpMVCudaKernelZB(double* __restrict__ y,double alpha, const double* __restrict__ A_data, const int* __restrict__ A_i, const int* __restrict__ A_j, const double* __restrict__ x, int num_rows)
{
  int i= blockIdx.x * blockDim.x + threadIdx.x;
  if (i<num_rows){
    double temp = 0.0;
    int jj;
    for (jj = A_i[i]; jj < A_i[i+1]; jj++){
      int ajj=A_j[jj];
      temp += A_data[jj] * x[ajj];
    }
    y[i] = alpha*temp;
  }
}
  void SpMVCuda(int num_rows,double alpha, double *A_data,int *A_i, int *A_j, double *x, double beta, double *y){
    int num_threads=64;
    int num_blocks=num_rows/num_threads+1;
    //printf("SpMVCuda threads = %d Blocks = %d for num_rows =%d \n",num_threads,num_blocks,num_rows);
    gpuErrchk2(cudaPeekAtLastError());
    gpuErrchk2(cudaDeviceSynchronize());
    if (beta==0.0)
      SpMVCudaKernelZB<<<num_blocks,num_threads>>>(y,alpha,A_data,A_i,A_j,x,num_rows);
    else
      SpMVCudaKernel<<<num_blocks,num_threads>>>(y,alpha,A_data,A_i,A_j,x,beta,num_rows);
    gpuErrchk2(cudaPeekAtLastError());
    gpuErrchk2(cudaDeviceSynchronize());

}
}
