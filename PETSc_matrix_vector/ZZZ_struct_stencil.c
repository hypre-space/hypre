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
   return ( (ZZZ_StructStencil) zzz_NewStructStencil( dim, size ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_SetStructStencilElement
 *--------------------------------------------------------------------------*/

void 
ZZZ_SetStructStencilElement( ZZZ_StructStencil  stencil,
		       int          element_index,
		       int         *offset        )
{
   zzz_SetStructStencilElement( (zzz_StructStencil *) stencil, element_index, offset );
}

/*--------------------------------------------------------------------------
 * ZZZ_FreeStructStencil
 *--------------------------------------------------------------------------*/

void 
ZZZ_FreeStructStencil( ZZZ_StructStencil stencil )
{
   zzz_FreeStructStencil( (zzz_StructStencil *) stencil );
}

