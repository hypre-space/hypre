/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_SStructStencilRef
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructStencilRef( hypre_SStructStencil  *stencil,
                         hypre_SStructStencil **stencil_ref )
{
   hypre_SStructStencilRefCount(stencil) ++;
   *stencil_ref = stencil;

   return hypre_error_flag;
}
