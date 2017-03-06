#ifdef HYPRE_USE_GPU
#include "gpuErrorCheck.h"
#include "hypre_nvtx.h"
#include <stdlib.h>
#include <stdint.h>
#include "gpuUtils.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include "_hypre_utilities.h"

int ggc(int id);

cublasHandle_t getCublasHandle();
cusparseHandle_t getCusparseHandle();
void hypreGPUInit(){
  char pciBusId[80];
  int myid;
  int nDevices;
  int device;
  gpuErrchk(cudaGetDeviceCount(&nDevices));
  printf("There are %d GPUs on this node \n",nDevices);
  if (nDevices>1) printf("WARNING:: Code running without mpibind or similar\n");
  hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

  
  device=myid%nDevices; // Warning will not work on multiple nodes without mpibind
  gpuErrchk(cudaSetDevice(device));
  cudaDeviceGetPCIBusId ( pciBusId, 80, device);
  printf("MPI_RANK = %d runningon PCIBUS id :: %s as device %d of %d\n",myid,pciBusId,device,nDevices);

  // Initialize the handles and streams
  cublasHandle_t handle_1 = getCublasHandle();
  cusparseHandle_t handle_2 = getCusparseHandle();
  cudaStream_t s=getstream(4);
  
}
void MemAdviseReadOnly(void* ptr, int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
    gpuErrchk(cudaMemAdvise(ptr,size,cudaMemAdviseSetReadMostly,device));
}
void MemAdviseUnSetReadOnly(void* ptr, int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
    gpuErrchk(cudaMemAdvise(ptr,size,cudaMemAdviseUnsetReadMostly,device));
}


void MemAdviseSetPrefLocDevice(const void *ptr, int device){
  if (ptr==NULL) return;
  gpuErrchk(cudaMemAdvise(ptr,mempush(ptr,0,0),cudaMemAdviseSetPreferredLocation,device));
}
void MemAdviseSetPrefLocHost(const void *ptr){
  if (ptr==NULL) return;
  gpuErrchk(cudaMemAdvise(ptr,mempush(ptr,0,0),cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId));
}


void MemPrefetch(const void *ptr,int device,cudaStream_t stream){
  if (ptr==NULL) return;
  size_t size;
  size=mempush(ptr,0,0);
  PUSH_RANGE_DOMAIN("MemPreFetchForce",4,0);
  /* Do a prefetch every time until a possible UM bug is fixed */
  if (size>0){
  gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
  gpuErrchk(cudaStreamSynchronize(stream));
  POP_RANGE_DOMAIN(0);
  return;
 } else {
  //printf("WARNING :: Prefetching not done due to nvalid size  = %zu\n",size);
  return;
  }
  /* End forced prefetch */
  PUSH_RANGE_PAYLOAD("MemPreFetch",4,size);
  if (memloc(ptr,device)){
    size=mempush(ptr,0,0);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
    gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));

  } 
  POP_RANGE;
  return;
}
void MemPrefetchForce(const void *ptr,int device,cudaStream_t stream){
  if (ptr==NULL) return;
  size_t size=mempush(ptr,0,0);
  PUSH_RANGE_PAYLOAD("MemPreFetchForce",4,size);
  gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
  POP_RANGE;
  return;
}

void MemPrefetchReadOnly(const void *ptr,int device,cudaStream_t stream){
  if (ptr==NULL) return;
  if (memloc(ptr,device)){
    size_t size=mempush(ptr,0,0);
    PUSH_RANGE_PAYLOAD("MemAdviseRO",3,size);
    gpuErrchk(cudaMemAdvise(ptr,size,cudaMemAdviseSetReadMostly,device));
    POP_RANGE;
    PUSH_RANGE_PAYLOAD("MemPreFetchRO",4,size);
    gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
    POP_RANGE;
  } 
  return;
}

void PrintPointerAttributesNew(void *ptr){
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess){
    printf("PrintPointerAttributes:: Raw pointer\n");
    return;
  }
  if (ptr_att.isManaged){
    printf("PrintPointerAttributes:: Managed pointer\n");
    printf("Host address = %p, Device Address = %p\n",ptr_att.hostPointer, ptr_att.devicePointer);
    if (ptr_att.memoryType==cudaMemoryTypeHost) printf("Memory is located on host\n");
    if (ptr_att.memoryType==cudaMemoryTypeDevice) printf("Memory is located on device\n");
    printf("Device associated with this pointer is %d\n",ptr_att.device);
  } else {
    printf("PrintPointerAttributes:: Non-Managed & non-raw pointer\n Probably a device pointer\n");
  }
  return;
}
int OnHost(void *ptr){
  return 1;
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,ptr)==cudaSuccess){
    return (ptr_att.memoryType==cudaMemoryTypeHost);
  } else {
    printf("PrintPointerAttributes:: Raw pointer\n");
    return 0;
  } 
}

// DeviceShare code doesnt work and triggers errors that are caught by cudaPeekLastError
int DeviceShare(void *ptr,size_t size){
uintptr_t p = (uintptr_t)ptr;
uintptr_t start,end,lastpage;
int pagesize = 64*1024; // 64KB HARDWIRED !!
start=p-p%pagesize;
end=p+size;
lastpage=end-end%pagesize;
uintptr_t i;
int dc=0,hc=0;
printf("Device Share s=%p, e =%p, actual=%p, pages = %lu\n",(void*)start,(void*)lastpage,ptr,size/pagesize);
for(i=start;i<=lastpage;i+=pagesize){
  void *pptr=(void*)i;
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,pptr)==cudaSuccess){
    if (ptr_att.memoryType==cudaMemoryTypeHost)hc++;
	else dc++;
  } else {
    //printf("PrintPointerAttributes:: Raw pointer %p of size %d\n",ptr,size);
    // if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess) printf("Original Pointer is kaput as well\n");
    return -1;
  } 
}
int retval=(float)dc/(dc+hc)*100.0;
return retval;
}

/* Returns the same cublas handle with every call */
cublasHandle_t getCublasHandle(){
  cublasStatus_t stat;
  static cublasHandle_t handle;
  static int firstcall=1;
  if (firstcall){
    firstcall=0;
    stat = cublasCreate(&handle);
    if (stat!=CUBLAS_STATUS_SUCCESS) {
      printf("ERROR:: CUBLAS Library initialization failed\n");
      handle=0;
      exit(2);
    }
    cublasErrchk(cublasSetStream(handle,getstream(4)));
  } else return handle;
  return handle;
}

/* Returns the same cusparse handle with every call */
cusparseHandle_t getCusparseHandle(){
  cusparseStatus_t status;
  static cusparseHandle_t handle;
  static int firstcall=1;
  if (firstcall){
    firstcall=0;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("ERROR:: CUSPARSE Library initialization failed\n");
      handle=0;
      exit(2);
    }
    cusparseErrchk(cusparseSetStream(handle,getstream(4)));
  } else return handle;
  return handle;
}
#endif
