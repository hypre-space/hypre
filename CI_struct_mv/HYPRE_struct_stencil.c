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

HYPRE_StructStencil
HYPRE_NewStructStencil( int dim,
		int size )
{
   return ( (HYPRE_StructStencil) hypre_NewStructStencil( dim, size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructStencilElement
 *--------------------------------------------------------------------------*/

void 
HYPRE_SetStructStencilElement( HYPRE_StructStencil  stencil,
		       int          element_index,
		       int         *offset        )
{
   hypre_SetStructStencilElement( (hypre_StructStencil *) stencil, element_index, offset );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
HYPRE_FreeStructStencil( HYPRE_StructStencil stencil )
{
   hypre_FreeStructStencil( (hypre_StructStencil *) stencil );
}

