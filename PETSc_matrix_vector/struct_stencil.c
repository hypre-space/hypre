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
 * zzz_NewStructStencil
 *--------------------------------------------------------------------------*/

zzz_StructStencil *
zzz_NewStructStencil( int dim,
		int size )
{
   zzz_StructStencil     *stencil;
   zzz_StructStencilElt  *shape;


   stencil = talloc(zzz_StructStencil, 1);

   zzz_StructStencilShape(stencil) = ctalloc(zzz_StructStencilElt, size);
   zzz_StructStencilDim(stencil)  = dim;
   zzz_StructStencilSize(stencil) = size;

   return stencil;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
zzz_FreeStructStencil( zzz_StructStencil *stencil )
{
   if (stencil)
   {
      tfree(zzz_StructStencilShape(stencil));
      tfree(stencil);
   }
}

/*--------------------------------------------------------------------------
 * zzz_SetStructStencilElement
 *--------------------------------------------------------------------------*/

void 
zzz_SetStructStencilElement( zzz_StructStencil *stencil,
		       int          element_index,
		       int         *offset        )
{
   zzz_StructStencilElt  *shape;

   int          d;


   shape = zzz_StructStencilShape(stencil);
   for (d = 0; d < zzz_StructStencilDim(stencil); d++)
      shape[element_index][d] = offset[d];
}

