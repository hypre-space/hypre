/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * TODO: the struct function hypre_StructStencilRef has a different
 *       prototype than this one.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructStencilRef( hypre_SStructStencil  *stencil,
                         hypre_SStructStencil **stencil_ref )
{
   if (stencil_ref)
   {
      hypre_SStructStencilRefCount(stencil) ++;
      *stencil_ref = stencil;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Appends to dst the boxes of the rows coupled by the stencil to the points
 * in src, that is, the rows i such that i + offset is in src for some stencil
 * offset. These are the boxes of src shifted by minus each stencil offset.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructStencilAppendCoupledBoxes( hypre_SStructStencil *stencil,
                                        hypre_BoxArray       *src,
                                        hypre_BoxArray       *dst )
{
   hypre_Box  *box;
   HYPRE_Int   b, e;

   box = hypre_BoxCreate(hypre_BoxArrayNDim(src));
   for (b = 0; b < hypre_BoxArraySize(src); b++)
   {
      for (e = 0; e < hypre_SStructStencilSize(stencil); e++)
      {
         hypre_CopyBox(hypre_BoxArrayBox(src, b), box);
         hypre_BoxShiftNeg(box, hypre_SStructStencilShape(stencil)[e]);
         hypre_AppendBox(box, dst);
      }
   }
   hypre_BoxDestroy(box);

   return hypre_error_flag;
}
