#include <stdio.h>
#include "hypre_nvtx.h"
#include "gpgpu.h"

extern "C"{
  void MemPrefetch(const void *ptr,int device,cudaStream_t stream);
  void MemPrefetchSized(const void *ptr,size_t size,int device,cudaStream_t stream);
  cudaStream_t getstream(int i);
  cudaEvent_t getevent(int i);
}
#define gpuErrchk2(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     printf("GPUassert2: %s %s %d\n", cudaGetErrorString(code), file, line);
     exit(2);
   }
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
    MemPrefetchSized(l1_norm,num_rows*sizeof(double),0,s);
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
    //gpuErrchk2(cudaStreamSynchronize(s));
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
    MemPrefetchSized(tgt,size*sizeof(double),0,s);
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
  void PackOnDevice(double *send_data,double *x_local_data, int *send_map, int begin,int end,cudaStream_t s){
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    if ((end-begin)<=0) return;
    int tpb=64;
    int num_blocks=(end-begin)/tpb+1;
    PackOnDeviceKernel<<<num_blocks,tpb,0,s>>>(send_data,x_local_data,send_map,begin,end);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    //gpuErrchk2(cudaEventRecord(getevent(4),s));
    //gpuErrchk2(cudaStreamWaitEvent(getstream(7),getevent(4),0));
    PUSH_RANGE("PACK_PREFETCH",1);
#ifndef HYPRE_GPU_USE_PINNED
    MemPrefetchSized((void*)send_data,(end-begin)*sizeof(double),cudaCpuDeviceId,s);
#endif
    POP_RANGE;
    //gpuErrchk2(cudaStreamSynchronize(s));
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
extern "C"{
  __global__
  void CompileFlagSafetyCheck(int actual){
#ifdef __CUDA_ARCH__
    int cudarch=__CUDA_ARCH__;
    if (cudarch!=actual){
      printf("WARNING :: nvcc -arch flag does not match actual device architecture\nWARNING :: The code can fail silently and produce wrong results\n");
      printf("Arch specified at compile = sm_%d Actual device = sm_%d\n",cudarch/10,actual/10);
    } //else printf("CompileFlagSafetyCheck:: Compile flag %d matches arch\n",cudarch);
#else
    printf("ERROR:: CUDA_ ARCH is not defined \n This should not be happening\n");
    //return 0;
#endif
  }
}
extern "C"{
  void CudaCompileFlagCheck(){
    int devCount;
    cudaGetDeviceCount(&devCount);
    int i;
    int cudarch_actual;
    for(i = 0; i < devCount; ++i)
      {
	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, i);
	cudarch_actual=props.major*100+props.minor*10;
    }
    gpuErrchk2(cudaPeekAtLastError());
    gpuErrchk2(cudaDeviceSynchronize());
    CompileFlagSafetyCheck<<<1,1,0,0>>>(cudarch_actual);
    cudaError_t code=cudaPeekAtLastError();
    if (code != cudaSuccess)
      {
	fprintf(stderr,"ERROR in CudaCompileFlagCheck%s \n", cudaGetErrorString(code));
	fprintf(stderr,"ERROR :: Check if compile arch flags match actual device arch = sm_%d\n",cudarch_actual/10);
	exit(2);
      }
    gpuErrchk2(cudaDeviceSynchronize());
  }
}
