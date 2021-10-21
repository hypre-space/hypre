/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <HYPRE_config.h>

#if defined(HYPRE_EXAMPLE_USING_CUDA)

#include <cuda_runtime.h>

typedef size_t devptr_t;

/* this depends on Fortran name mangling */
#if 0
#define CUDA_MALLOC_MANAGED  device_malloc_managed_
#define CUDA_FREE            device_free_
#else
#define CUDA_MALLOC_MANAGED  device_malloc_managed
#define CUDA_FREE            device_free
#endif

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

