/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_MAGMA_HEADER
#define HYPRE_MAGMA_HEADER

#if defined(HYPRE_USING_MAGMA)

#include "error.h"

#ifdef __cplusplus
extern "C++"
{
#endif

#if !defined(MAGMA_GLOBAL)
#define ADD_
#endif
#include <magma_v2.h>

#ifdef __cplusplus
}
#endif

/*--------------------------------------------------------------------------
 * Wrappers to MAGMA functions according to hypre's precision
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_COMPLEX) || defined(HYPRE_LONG_DOUBLE)
#error MAGMA interface does not support complex and long double numbers

#elif defined(HYPRE_SINGLE)
#define hypre_magma_getrf_gpu              magma_sgetrf_gpu
#define hypre_magma_getrs_gpu              magma_sgetrs_gpu

#else /* Double precision */
#define hypre_magma_getrf_gpu              magma_dgetrf_gpu
#define hypre_magma_getrs_gpu              magma_dgetrs_gpu

#endif

/*--------------------------------------------------------------------------
 * General wrapper call to MAGMA functions
 *--------------------------------------------------------------------------*/

#define HYPRE_MAGMA_CALL(call) do {                   \
   magma_int_t err = call;                            \
   if (MAGMA_SUCCESS != err) {                        \
      printf("MAGMA ERROR (code = %d) at %s:%d\n",    \
            err, __FILE__, __LINE__);                 \
      hypre_assert(0);                                \
   } } while(0)

#endif /* HYPRE_USING_MAGMA */
#endif /* HYPRE_MAGMA_HEADER */
