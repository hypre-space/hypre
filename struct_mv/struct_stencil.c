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
zzz_NewStructStencil( int         dim,
                      int         size,
                      zzz_Index **shape )
{
   zzz_StructStencil   *stencil;

   stencil = talloc(zzz_StructStencil, 1);

   zzz_StructStencilShape(stencil) = shape;
   zzz_StructStencilSize(stencil)  = size;
   zzz_StructStencilDim(stencil)   = dim;

   return stencil;
}

/*--------------------------------------------------------------------------
 * zzz_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
zzz_FreeStructStencil( zzz_StructStencil *stencil )
{
   int  i;

   if (stencil)
   {
      for (i = 0; i < zzz_StructStencilSize(stencil); i++)
         zzz_FreeIndex(zzz_StructStencilShape(stencil)[i]);
      tfree(zzz_StructStencilShape(stencil));
      tfree(stencil);
   }
}

