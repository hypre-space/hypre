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
 * ZZZ_StructStencil interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * ZZZ_NewStructStencil
 *--------------------------------------------------------------------------*/

ZZZ_StructStencil
ZZZ_NewStructStencil( int dim,
                      int size )
{
   zzz_Index **shape;
   int         i;
 
   shape = zzz_CTAlloc(zzz_Index *, size);
   for (i = 0; i < size; i++)
   {
      shape[i] = zzz_NewIndex();
   }
 
   return ( (ZZZ_StructStencil) zzz_NewStructStencil( dim, size, shape ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructStencilElement
 *--------------------------------------------------------------------------*/

void 
ZZZ_SetStructStencilElement( ZZZ_StructStencil  stencil,
                             int                element_index,
                             int               *offset        )
{
   zzz_StructStencil  *new_stencil = (zzz_StructStencil *) stencil;
   zzz_Index         **shape;
   int                 d;
 
   shape = zzz_StructStencilShape(new_stencil);
   for (d = 0; d < zzz_StructStencilDim(new_stencil); d++)
   {
      zzz_IndexD(shape[element_index], d) = offset[d];
   }
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
ZZZ_FreeStructStencil( ZZZ_StructStencil stencil )
{
   zzz_FreeStructStencil( (zzz_StructStencil *) stencil );
}

