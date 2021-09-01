#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <HYPRE_config.h>

typedef size_t devptr_t;

#if defined(HYPRE_USING_CUDA)

#define CUDA_MALLOC_MANAGED  device_malloc_managed_
#define CUDA_FREE            device_free_

int CUDA_MALLOC_MANAGED (const int *nbytes, devptr_t *devicePtr)
{
    void *tPtr;
    int retVal;
    retVal = (int) cudaMallocManaged (&tPtr, *nbytes, cudaMemAttachGlobal);
    *devicePtr = (devptr_t)tPtr;

    /* printf("allocate %p, size %d\n", tPtr, *nbytes); */

    return retVal;
}

int CUDA_FREE (const devptr_t *devicePtr)
{
    void *tPtr;
    tPtr = (void *)(*devicePtr);

    /* printf("free     %p\n", tPtr); */

    return (int)cudaFree (tPtr);
}

#endif

