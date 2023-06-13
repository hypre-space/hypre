/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#if defined(HYPRE_USING_MAGMA)

/*--------------------------------------------------------------------------
 * hypre_MagmaInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MagmaInitialize(void)
{
   /* Initialize MAGMA */
   magma_init();

   /* Create device queue */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_int device_id;

   hypre_GetDevice(&device_id);
   magma_queue_create((magma_int_t) device_id, &hypre_HandleMagmaQueue(hypre_handle()));
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MagmaFinalize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MagmaFinalize(void)
{
   /* Finalize MAGMA */
   magma_finalize();

   return hypre_error_flag;
}

#endif /* HYPRE_USING_MAGMA */
