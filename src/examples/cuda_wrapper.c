#include <stdlib.h>
#include <stdio.h>
#include <HYPRE_config.h>

typedef size_t devptr_t;

#if defined(HYPRE_EXAMPLE_USING_CUDA)

#include <cuda_runtime.h>

#define CUDA_MALLOC_MANAGED  device_malloc_managed
#define CUDA_FREE            device_free

int CUDA_MALLOC_MANAGED (const int *nbytes, devptr_t *devicePtr)
{
   void *tPtr;
   int retVal = (int) cudaMallocManaged (&tPtr, *nbytes, cudaMemAttachGlobal);
   *devicePtr = (devptr_t)tPtr;
   return retVal;
}

int CUDA_FREE (const devptr_t *devicePtr)
{
   void *tPtr;
   tPtr = (void *)(*devicePtr);
   return (int)cudaFree (tPtr);
}

#endif /* #if defined(HYPRE_EXAMPLE_USING_CUDA) */

