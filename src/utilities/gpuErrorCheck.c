#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
#include "gpuErrorCheck.h"
#include "hypre_nvtx.h"

#include <signal.h>
extern const char *cusparseErrorCheck(cusparseStatus_t error);
extern void gpuAssert(cudaError_t code, const char *file, int line);
extern void cusparseAssert(cusparseStatus_t code, const char *file, int line);

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

  err=cudaPointerGetAttributes(&ptr_att,sptr);
  if (err!=cudaSuccess){
#ifndef ABORT_ON_RAW_POINTER
    if (err==cudaErrorInvalidValue) fprintf(stderr,"WARNING :: Raw pointer passed to cudaSafeFree %p\n",ptr);
    if (err==cudaErrorInvalidDevice) fprintf(stderr,"WARNING :: cudaSafeFree :: INVALID DEVICE on ptr = %p\n",ptr);
    PrintPointerAttributes(ptr);
#else
    fprintf(stderr,"ERROR:: cudaSafeFree Aborting on raw unmanaged pointer %p\n",ptr);
    raise(SIGABRT);
#endif
    free(ptr); /* Free the nonManaged pointer */
    return;
  }
  if (ptr_att.isManaged){
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
void PrintPointerAttributes(const void *ptr){
  struct cudaPointerAttributes ptr_att;
  if (cudaPointerGetAttributes(&ptr_att,ptr)!=cudaSuccess){
    fprintf(stderr,"PrintPointerAttributes:: Raw pointer\n");
    return;
  }
  if (ptr_att.isManaged){
    fprintf(stderr,"PrintPointerAttributes:: Managed pointer\n");
    fprintf(stderr,"Host address = %p, Device Address = %p\n",ptr_att.hostPointer, ptr_att.devicePointer);
    if (ptr_att.memoryType==cudaMemoryTypeHost) fprintf(stderr,"Memory is located on host\n");
    if (ptr_att.memoryType==cudaMemoryTypeDevice) fprintf(stderr,"Memory is located on device\n");
    fprintf(stderr,"Device associated with this pointer is %d\n",ptr_att.device);
  } else {
    fprintf(stderr,"PrintPointerAttributes:: Non-Managed & non-raw pointer\n Probably pinned host pointer\n");
    if (ptr_att.memoryType==cudaMemoryTypeHost) fprintf(stderr,"Memory is located on host\n");
    if (ptr_att.memoryType==cudaMemoryTypeDevice) fprintf(stderr,"Memory is located on device\n");
  }
  return;
}
#endif
