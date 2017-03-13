#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
#include "gpuErrorCheck.h"
#include "hypre_nvtx.h"

#define FULL_WARNING

#ifdef ABORT_ON_RAW_POINTER
#endif
#include <signal.h>
extern const char *cusparseErrorCheck(cusparseStatus_t error);
extern void gpuAssert(cudaError_t code, const char *file, int line);
extern void cusparseAssert(cusparseStatus_t code, const char *file, int line);

void cudaSafeFree(void *ptr)
{
  PUSH_RANGE("SAFE_FREE",3);
  struct cudaPointerAttributes ptr_att;
  cudaError_t err;
#ifdef FULL_WARNING
  err=cudaPointerGetAttributes(&ptr_att,ptr);
  if (err!=cudaSuccess){
    if (err==cudaErrorInvalidValue) fprintf(stderr,"INVALID VALUE\n");
    if (err==cudaErrorInvalidDevice) fprintf(stderr,"INVALID DEVICE\n");
    fprintf(stderr,"WARNING :: Raw pointer passed to cudaSafeFree %p\n",ptr);
    PrintPointerAttributes(ptr);
#endif
#ifdef ABORT_ON_RAW_POINTER
    raise(SIGABRT);
#endif
    free(ptr);
    return;
  }
  if (ptr_att.isManaged){
    gpuErrchk(cudaFree(ptr));

  } else {
    printf("ERROR:: NON-managed pointer passed to Mfree\n");
    gpuErrchk(cudaFree(ptr));
  }
  POP_RANGE;
  return;
}
void PrintPointerAttributes(void *ptr){
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
#endif
