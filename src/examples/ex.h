/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Header file for examples
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_EXAMPLES_INCLUDES
#define HYPRE_EXAMPLES_INCLUDES

#include <HYPRE_config.h>

#if defined(HYPRE_EXAMPLE_USING_CUDA) || defined(HYPRE_EXAMPLE_USING_HIP)
#ifndef HYPRE_USING_UNIFIED_MEMORY
#error *** Running the examples on GPUs requires Unified Memory. Please reconfigure and rebuild with --enable-unified-memory ***
#endif
#define custom_malloc(size) gpu_malloc(size)
#define custom_calloc(num, size) gpu_calloc(num, size)
#define custom_free(ptr) gpu_free((void**) &(ptr))
#endif

#if defined(HYPRE_EXAMPLE_USING_CUDA)

#include <cuda_runtime.h>

static inline void*
gpu_malloc(size_t size)
{
   void *ptr = NULL;
   cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
   return ptr;
}

static inline void*
gpu_calloc(size_t num, size_t size)
{
   void *ptr = NULL;
   cudaMallocManaged(&ptr, num * size, cudaMemAttachGlobal);
   cudaMemset(ptr, 0, num * size);
   return ptr;
}

static inline void
gpu_free(void** ptr)
{
   cudaFree(*ptr);
   *ptr = NULL;
}

#elif defined(HYPRE_EXAMPLE_USING_HIP)

#include <hip/hip_runtime.h>

static inline void*
gpu_malloc(size_t size)
{
   void *ptr = NULL;
   hipMallocManaged(&ptr, size, hipMemAttachGlobal);
   return ptr;
}

static inline void*
gpu_calloc(size_t num, size_t size)
{
   void *ptr = NULL;
   hipMallocManaged(&ptr, num * size, hipMemAttachGlobal);
   hipMemset(ptr, 0, num * size);
   return ptr;
}

static inline void
gpu_free(void** ptr)
{
   hipFree(*ptr);
   *ptr = NULL;
}

#else

/* Host: Default malloc, calloc, and free */
#define custom_malloc(size) malloc(size)
#define custom_calloc(num, size) calloc(num, size)
#define custom_free(ptr) (free(ptr), ptr = NULL)
#endif

#endif /* #ifdef HYPRE_EXAMPLES_INCLUDES */
