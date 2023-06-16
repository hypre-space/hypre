/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_MAGMA_HEADER
#define HYPRE_MAGMA_HEADER

#include "HYPRE_config.h"

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

#define HYPRE_MAGMA_CALL(call) do {                   \
   magma_int_t err = call;                            \
   if (MAGMA_SUCCESS != err) {                        \
      printf("MAGMA ERROR (code = %d) at %s:%d\n",    \
            err, __FILE__, __LINE__);                 \
      hypre_assert(0);                                \
   } } while(0)

#endif /* HYPRE_USING_MAGMA */
#endif /* HYPRE_MAGMA_HEADER */
