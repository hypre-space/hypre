/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilCreate( HYPRE_Int            dim,
                           HYPRE_Int            size,
                           HYPRE_StructStencil *stencil )
{
   hypre_Index  *shape;

   shape = hypre_CTAlloc(hypre_Index,  size, HYPRE_MEMORY_HOST);

   *stencil = hypre_StructStencilCreate(dim, size, shape);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilSetElement( HYPRE_StructStencil  stencil,
                               HYPRE_Int            element_index,
                               HYPRE_Int           *offset )
{
   hypre_Index  *shape;
   HYPRE_Int     d;

   shape = hypre_StructStencilShape(stencil);
   hypre_SetIndex(shape[element_index], 0);
   for (d = 0; d < hypre_StructStencilNDim(stencil); d++)
   {
      hypre_IndexD(shape[element_index], d) = offset[d];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilDestroy( HYPRE_StructStencil stencil )
{
   return ( hypre_StructStencilDestroy(stencil) );
}

