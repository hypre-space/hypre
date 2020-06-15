/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header for the BoxLoop
 *
 *****************************************************************************/
#ifndef hypre_BOXLOOP_HEADER
#define hypre_BOXLOOP_HEADER

#if defined(HYPRE_USING_RAJA)

#include "boxloop_raja.h"

#elif defined(HYPRE_USING_KOKKOS)

#include "boxloop_kokkos.h"

#elif defined(HYPRE_USING_CUDA)

#include "../utilities/hypre_cuda_utils.h"
#include "boxloop_cuda.h"

#elif defined(HYPRE_USING_DEVICE_OPENMP)

#include "../utilities/hypre_cuda_utils.h"
#include "../utilities/hypre_omp_device.h"
#include "boxloop_omp45.h"

#else

#include "boxloop_host.h"

#endif

#endif

