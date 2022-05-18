/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructStencilPrint( FILE *file, HYPRE_SStructStencil stencil )
{
   HYPRE_Int    ndim  = hypre_SStructStencilNDim(stencil);
   HYPRE_Int   *vars  = hypre_SStructStencilVars(stencil);
   hypre_Index *shape = hypre_SStructStencilShape(stencil);
   HYPRE_Int    size  = hypre_SStructStencilSize(stencil);

   HYPRE_Int    i;

   hypre_fprintf(file, "StencilCreate: %d %d", ndim, size);
   for (i = 0; i < size; i++)
   {
      hypre_fprintf(file, "\nStencilSetEntry: %d %d ", i, vars[i]);
      hypre_IndexPrint(file, ndim, shape[i]);
   }
   hypre_fprintf(file, "\n");

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructStencilRead( FILE *file, HYPRE_SStructStencil *stencil_ptr )
{
   HYPRE_SStructStencil    stencil;

   HYPRE_Int               var;
   hypre_Index             shape;
   HYPRE_Int               i, ndim;
   HYPRE_Int               entry, size;

   hypre_fscanf(file, "StencilCreate: %d %d", &ndim, &size);
   HYPRE_SStructStencilCreate(ndim, size, &stencil);

   for (i = 0; i < size; i++)
   {
      hypre_fscanf(file, "\nStencilSetEntry: %d %d ", &entry, &var);
      hypre_IndexRead(file, ndim, shape);

      HYPRE_SStructStencilSetEntry(stencil, entry, shape, var);
   }
   hypre_fscanf(file, "\n");

   *stencil_ptr = stencil;

   return hypre_error_flag;
}
