/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * HYPRE_SStructStencil interface
 *
 *****************************************************************************/

#include "headers.h"

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

   stencil = hypre_TAlloc(hypre_SStructStencil, 1);
   HYPRE_StructStencilCreate(ndim, size, &sstencil);
   vars = hypre_CTAlloc(HYPRE_Int, hypre_StructStencilSize(sstencil));

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
         hypre_TFree(hypre_SStructStencilVars(stencil));
         hypre_TFree(stencil);
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


