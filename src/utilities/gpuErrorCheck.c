
#include "_hypre_utilities.h"

#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
#include <signal.h>
#ifdef HYPRE_USE_GPU
extern const char *cusparseErrorCheck(cusparseStatus_t error);
extern void gpuAssert(cudaError_t code, const char *file, int line);
extern void cusparseAssert(cusparseStatus_t code, const char *file, int line);
#endif

/*
  cudaSafeFree frees Managed memory allocated in hypre_MAlloc,hypre_CAlloc and hypre_ReAlloc
  It checks if the memory is managed before freeing and emits a warning if it is not memory
  allocated using the above routines. This behaviour can be changed by defining ABORT_ON_RAW_POINTER.
  The core file can then be used to find the location of the anomalous hypre_Free.
 */
void cudaSafeFree(void *ptr,int padding)
{
  PUSH_RANGE("SAFE_FREE",3);
  struct cudaPointerAttributes ptr_att;
  size_t *sptr=(size_t*)ptr-padding;
  cudaError_t err;

  err=cudaPointerGetAttributes(&ptr_att,ptr);
  if (err!=cudaSuccess){
    cudaGetLastError(); 
#define FULL_WARN
#ifndef ABORT_ON_RAW_POINTER
#ifdef FULL_WARN
    if (err==cudaErrorInvalidValue) fprintf(stderr,"WARNING :: Raw pointer passed to cudaSafeFree %p\n",ptr);
    if (err==cudaErrorInvalidDevice) fprintf(stderr,"WARNING :: cudaSafeFree :: INVALID DEVICE on ptr = %p\n",ptr);
    //PrintPointerAttributes(ptr);
#endif
#else
    fprintf(stderr,"ERROR:: cudaSafeFree Aborting on raw unmanaged pointer %p\n",ptr);
    raise(SIGABRT);
#endif
    free(ptr); /* Free the nonManaged pointer */
    return;
  }
  if (ptr_att.isManaged){
#if defined(HYPRE_USE_GPU) && defined(HYPRE_MEASURE_GPU_HWM)
    size_t mfree,mtotal;
    gpuErrchk(cudaMemGetInfo(&mfree,&mtotal));
    HYPRE_GPU_HWM=hypre_max((mtotal-mfree),HYPRE_GPU_HWM);
#endif
    /* Code below for handling managed memory pointers not allocated using hypre_CTAlloc oir hypre_TAlooc */
    if (PointerAttributes(ptr)!=PointerAttributes(sptr)){
      //fprintf(stderr,"ERROR IN Pointer for freeing %p %p\n",ptr,sptr);
      gpuErrchk(cudaFree(ptr)); 
      return;
    }
    gpuErrchk(cudaFree(sptr)); 
  } else {
    /* It is a pinned memory pointer */
    //printf("ERROR:: NON-managed pointer passed to cudaSafeFree\n");
    if (ptr_att.memoryType==cudaMemoryTypeHost){
      gpuErrchk(cudaFreeHost(sptr));
    } else if (ptr_att.memoryType==cudaMemoryTypeDevice){
      gpuErrchk(cudaFree(sptr)); 
    }
  }
  POP_RANGE;
  return;
}
hypre_int PrintPointerAttributes(const void *ptr){
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess){
    cudaGetLastError(); 
    fprintf(stderr,"PrintPointerAttributes:: Raw pointer %p\n",ptr);
    return HYPRE_HOST_POINTER;
  }
  if (ptr_att.isManaged){
    fprintf(stderr,"PrintPointerAttributes:: Managed pointer\n");
    fprintf(stderr,"Host address = %p, Device Address = %p\n",ptr_att.hostPointer, ptr_att.devicePointer);
    if (ptr_att.memoryType==cudaMemoryTypeHost) fprintf(stderr,"Memory is located on host\n");
    if (ptr_att.memoryType==cudaMemoryTypeDevice) fprintf(stderr,"Memory is located on device\n");
    fprintf(stderr,"Device associated with this pointer is %d\n",ptr_att.device);
    return HYPRE_MANAGED_POINTER;
  } else {
    fprintf(stderr,"PrintPointerAttributes:: Non-Managed & non-raw pointer\n Probably pinned host pointer\n");
    if (ptr_att.memoryType==cudaMemoryTypeHost) {
      fprintf(stderr,"Memory is located on host\n");
      return HYPRE_PINNED_POINTER;
    }
    if (ptr_att.memoryType==cudaMemoryTypeDevice) {
      fprintf(stderr,"Memory is located on device\n");
      return HYPRE_DEVICE_POINTER ;
    }
    return HYPRE_UNDEFINED_POINTER1;
  }
  return HYPRE_UNDEFINED_POINTER2;
}
hypre_int PointerAttributes(const void *ptr){
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess){
     cudaGetLastError(); 
     return HYPRE_HOST_POINTER;
  }
  if (ptr_att.isManaged){
    return HYPRE_MANAGED_POINTER; 
  } else {
    if (ptr_att.memoryType==cudaMemoryTypeHost) return HYPRE_PINNED_POINTER; /* Host pointer from cudaMallocHost */
    if (ptr_att.memoryType==cudaMemoryTypeDevice) return HYPRE_DEVICE_POINTER ; /* cudadevice pointer */
    return HYPRE_UNDEFINED_POINTER1; /* Shouldn't happen */
  }
  return HYPRE_UNDEFINED_POINTER2; /* Shouldnt happen */
}

#endif
