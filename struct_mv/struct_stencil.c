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

   stencil = zzz_TAlloc(zzz_StructStencil, 1);

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
      zzz_TFree(zzz_StructStencilShape(stencil));
      zzz_TFree(stencil);
   }
}

/*--------------------------------------------------------------------------
 * zzz_StructStencilElementRank
 *    Returns the rank of the `stencil_element' in `stencil'.
 *    If the element is not found, a -1 is returned.
 *--------------------------------------------------------------------------*/

int
zzz_StructStencilElementRank( zzz_StructStencil *stencil,
                              zzz_Index         *stencil_element )
{
   zzz_Index **stencil_shape;
   int         rank;
   int         i;

   rank = -1;
   stencil_shape = zzz_StructStencilShape(stencil);
   for (i = 0; i < zzz_StructStencilSize(stencil); i++)
   {
      if ((zzz_IndexX(stencil_shape[i]) == zzz_IndexX(stencil_element)) &&
          (zzz_IndexY(stencil_shape[i]) == zzz_IndexY(stencil_element)) &&
          (zzz_IndexZ(stencil_shape[i]) == zzz_IndexZ(stencil_element))   )
      {
         rank = i;
         break;
      }
   }

   return rank;
}

