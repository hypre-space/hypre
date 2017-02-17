#ifdef HYPRE_USE_GPU
#include "gpuErrorCheck.h"
#include "hypre_nvtx.h"
#include <stdlib.h>
#include <stdint.h>
#include "gpuUtils.h"
#include <cublas_v2.h>
#include <cusparse.h>
int ggc(int id);
void MemAdviseReadOnly(void* ptr, int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
  //size_t size=mempush(ptr,0,0);
  //printf("MEMA %p  size = %lu \n",ptr,size);
  //return;
  gpuErrchk(cudaMemAdvise(ptr,mempush(ptr,0,0),cudaMemAdviseSetReadMostly,device));
}
void MemAdviseUnSetReadOnly(void* ptr, int device){
  if (ptr==NULL) return;
    size_t size=mempush(ptr,0,0);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
  //size_t size=mempush(ptr,0,0);
  //printf("MEMA %p  size = %lu \n",ptr,size);
  //return;
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
  //printf("MEMLCO %d\n",memloc(ptr,device));
  //size_t size=mempush(ptr,0,0);
  //gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
  //return;
  if (memloc(ptr,device)){
    //float rval=rand()/(float)RAND_MAX;
    //if (rval<0.1){
    size_t size=mempush(ptr,0,0);
    //printf("prefetch of %p of size %d \n",ptr,size);
    if (size==0) printf("WARNING:: Operations with 0 size vector \n");
    PUSH_RANGE_PAYLOAD("MemPreFetch",4,size);
    gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
    POP_RANGE;
  } else {
    //size_t size=mempush(ptr,0,0);
    //printf("Skipped prefetch of %p of size %d \n",ptr,size);
  }
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
  //printf("MEMLCO %d\n",memloc(ptr,device));
  if (memloc(ptr,device)){
    //printf("%d MemPrefetch Triggered %p\n",ggc(-1),ptr);
    size_t size=mempush(ptr,0,0);
    PUSH_RANGE_PAYLOAD("MemAdviseRO",3,size);
    gpuErrchk(cudaMemAdvise(ptr,size,cudaMemAdviseSetReadMostly,device));
    POP_RANGE;
    PUSH_RANGE_PAYLOAD("MemPreFetchRO",4,size);
    gpuErrchk(cudaMemPrefetchAsync(ptr,size,device,stream));
    POP_RANGE;
  } //else printf("%d MemPrefetch Skipped %p \n",ggc(-1),ptr);
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
int DeviceShare(void *ptr,size_t size){
uintptr_t p = (uintptr_t)ptr;
uintptr_t start,end,lastpage;
int pagesize = 64*1024; // 64KB HARDWIRED !!
start=p-p%pagesize;
end=p+size;
lastpage=end-end%pagesize;
uintptr_t i;
int dc=0,hc=0;
printf("Device Share s=%p, e =%p, actual=%p, pages = %d\n",(void*)start,(void*)lastpage,ptr,size/pagesize);
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

cublasHandle_t getCublasHandle(){
  cublasStatus_t stat;
  static cublasHandle_t handle;
  static firstcall=1;
  if (firstcall){
    firstcall=0;
    stat = cublasCreate(&handle);
    if (stat!=CUBLAS_STATUS_SUCCESS) {
      printf("ERROR:: CUBLAS Library initialization failed\n");
      handle=0;
      exit(2);
    }
  } else return handle;
  return handle;
}
cusparseHandle_t getCusparseHandle(){
  cusparseStatus_t status;
  static cusparseHandle_t handle;
  static firstcall=1;
  if (firstcall){
    firstcall=0;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      printf("ERROR:: CUSPARSE Library initialization failed\n");
      handle=0;
      exit(2);
    }
  } else return handle;
  return handle;
}
#endif
