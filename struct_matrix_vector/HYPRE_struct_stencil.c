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
 * HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewStructStencil
 *--------------------------------------------------------------------------*/

int
HYPRE_NewStructStencil( int                  dim,
                        int                  size,
                        HYPRE_StructStencil *stencil )
{
   hypre_Index  *shape;
 
   shape = hypre_CTAlloc(hypre_Index, size);
 
   *stencil = ((HYPRE_StructStencil) hypre_NewStructStencil(dim, size,shape));

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructStencilElement
 *--------------------------------------------------------------------------*/

void 
HYPRE_SetStructStencilElement( HYPRE_StructStencil  stencil,
                               int                  element_index,
                               int                 *offset        )
{
   hypre_StructStencil  *new_stencil = (hypre_StructStencil *) stencil;
   hypre_Index          *shape;
   int                   d;
 
   shape = hypre_StructStencilShape(new_stencil);
   hypre_ClearIndex(shape[element_index]);
   for (d = 0; d < hypre_StructStencilDim(new_stencil); d++)
   {
      hypre_IndexD(shape[element_index], d) = offset[d];
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
HYPRE_FreeStructStencil( HYPRE_StructStencil stencil )
{
   hypre_FreeStructStencil( (hypre_StructStencil *) stencil );
}

