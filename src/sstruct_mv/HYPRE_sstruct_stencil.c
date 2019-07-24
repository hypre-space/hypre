/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructStencil interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructStencilCreate( HYPRE_Int             ndim,
                            HYPRE_Int             size,
                            HYPRE_SStructStencil *stencil_ptr )
{
   hypre_SStructStencil  *stencil;
   hypre_StructStencil   *sstencil;
   HYPRE_Int             *vars;

   stencil = hypre_TAlloc(hypre_SStructStencil,  1, HYPRE_MEMORY_HOST);
   HYPRE_StructStencilCreate(ndim, size, &sstencil);
   vars = hypre_CTAlloc(HYPRE_Int,  hypre_StructStencilSize(sstencil), HYPRE_MEMORY_HOST);

   hypre_SStructStencilSStencil(stencil) = sstencil;
   hypre_SStructStencilVars(stencil)     = vars;
   hypre_SStructStencilRefCount(stencil) = 1;

   *stencil_ptr = stencil;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructStencilDestroy( HYPRE_SStructStencil stencil )
{
   if (stencil)
   {
      hypre_SStructStencilRefCount(stencil) --;
      if (hypre_SStructStencilRefCount(stencil) == 0)
      {
         HYPRE_StructStencilDestroy(hypre_SStructStencilSStencil(stencil));
         hypre_TFree(hypre_SStructStencilVars(stencil), HYPRE_MEMORY_HOST);
         hypre_TFree(stencil, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructStencilSetEntry( HYPRE_SStructStencil  stencil,
                              HYPRE_Int             entry,
                              HYPRE_Int            *offset,
                              HYPRE_Int             var )
{
   hypre_StructStencil  *sstencil = hypre_SStructStencilSStencil(stencil);

   HYPRE_StructStencilSetElement(sstencil, entry, offset);
   hypre_SStructStencilVar(stencil, entry) = var;

   return hypre_error_flag;
}


