/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#include <cuda_runtime.h>
#define malloc(size) ( {void *ptr = NULL; cudaMallocManaged(&ptr, size, cudaMemAttachGlobal); ptr;} )
#define calloc(num, size) ( {void *ptr = NULL; cudaMallocManaged(&ptr, num*size, cudaMemAttachGlobal); cudaMemset(ptr, 0, num*size); ptr;} )
#define free(ptr) ( cudaFree(ptr), ptr = NULL )
#endif

#endif

