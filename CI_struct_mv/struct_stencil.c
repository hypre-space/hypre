/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Constructors and destructors for stencil structure.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_NewStructStencil
 *--------------------------------------------------------------------------*/

hypre_StructStencil *
hypre_NewStructStencil( int dim,
		int size )
{
   hypre_StructStencil     *stencil;
   hypre_StructStencilElt  *shape;


   stencil = hypre_CTAlloc(hypre_StructStencil, 1);

   hypre_StructStencilShape(stencil) = hypre_CTAlloc(hypre_StructStencilElt, size);
   hypre_StructStencilDim(stencil)  = dim;
   hypre_StructStencilSize(stencil) = size;

   return stencil;
}

/*--------------------------------------------------------------------------
 * hypre_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
hypre_FreeStructStencil( hypre_StructStencil *stencil )
{
   if (stencil)
   {
      hypre_TFree(hypre_StructStencilShape(stencil));
      hypre_TFree(stencil);
   }
}

/*--------------------------------------------------------------------------
 * hypre_SetStructStencilElement
 *--------------------------------------------------------------------------*/

void 
hypre_SetStructStencilElement( hypre_StructStencil *stencil,
		       int          element_index,
		       int         *offset        )
{
   hypre_StructStencilElt  *shape;

   int          d;


   shape = hypre_StructStencilShape(stencil);
   for (d = 0; d < hypre_StructStencilDim(stencil); d++)
      shape[element_index][d] = offset[d];
}

