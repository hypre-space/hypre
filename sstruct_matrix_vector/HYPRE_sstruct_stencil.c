/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_SStructStencil interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructStencilCreate( int                   ndim,
                            int                   size,
                            HYPRE_SStructStencil *stencil_ptr )
{
   int  ierr = 0;

   hypre_SStructStencil  *stencil;
   hypre_StructStencil   *sstencil;
   int                   *vars;

   stencil = hypre_TAlloc(hypre_SStructStencil, 1);
   ierr = HYPRE_StructStencilCreate(ndim, size, &sstencil);
   vars = hypre_CTAlloc(int, hypre_StructStencilSize(sstencil));

   hypre_SStructStencilSStencil(stencil) = sstencil;
   hypre_SStructStencilVars(stencil)     = vars;
   hypre_SStructStencilRefCount(stencil) = 1;

   *stencil_ptr = stencil;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilDestroy
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructStencilDestroy( HYPRE_SStructStencil stencil )
{
   int  ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilSetEntry
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructStencilSetEntry( HYPRE_SStructStencil  stencil,
                              int                   entry,
                              int                  *offset,
                              int                   var )
{
   int                   ierr;
   hypre_StructStencil  *sstencil = hypre_SStructStencilSStencil(stencil);

   ierr = HYPRE_StructStencilSetElement(sstencil, entry, offset);
   hypre_SStructStencilVar(stencil, entry) = var;

   return ierr;
}


