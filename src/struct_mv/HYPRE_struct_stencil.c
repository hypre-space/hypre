/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilCreate( HYPRE_Int            dim,
                           HYPRE_Int            size,
                           HYPRE_StructStencil *stencil )
{
   hypre_Index  *shape;
 
   shape = hypre_CTAlloc(hypre_Index, size);
 
   *stencil = hypre_StructStencilCreate(dim, size, shape);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilSetElement( HYPRE_StructStencil  stencil,
                               HYPRE_Int            element_index,
                               HYPRE_Int           *offset )
{
   HYPRE_Int     ierr = 0;
                
   hypre_Index  *shape;
   HYPRE_Int     d;
 
   shape = hypre_StructStencilShape(stencil);
   hypre_ClearIndex(shape[element_index]);
   for (d = 0; d < hypre_StructStencilDim(stencil); d++)
   {
      hypre_IndexD(shape[element_index], d) = offset[d];
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructStencilDestroy( HYPRE_StructStencil stencil )
{
   return ( hypre_StructStencilDestroy(stencil) );
}

