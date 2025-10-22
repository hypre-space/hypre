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
 * HYPRE_StructStencilSetEntry
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilSetEntry( HYPRE_StructStencil  stencil,
                             HYPRE_Int            entry,
                             HYPRE_Int           *offset )
{
   hypre_Index  *shape;
   HYPRE_Int     d;
   HYPRE_Int     is_diag = 1;

   shape = hypre_StructStencilShape(stencil);
   hypre_SetIndex(shape[entry], 0);
   for (d = 0; d < hypre_StructStencilNDim(stencil); d++)
   {
      hypre_IndexD(shape[entry], d) = offset[d];

      if (is_diag && offset[d] != 0)
      {
         is_diag = 0;
      }
   }

   if (is_diag)
   {
      hypre_StructStencilDiagEntry(stencil) = entry;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * OBSOLETE
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilSetElement( HYPRE_StructStencil  stencil,
                               HYPRE_Int            entry,
                               HYPRE_Int           *offset )
{
   HYPRE_StructStencilSetEntry(stencil, entry, offset);
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
