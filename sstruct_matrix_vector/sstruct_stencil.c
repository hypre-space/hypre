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
 * Member functions for hypre_SStructStencil class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructStencilRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructStencilRef( hypre_SStructStencil  *stencil,
                         hypre_SStructStencil **stencil_ref )
{
   hypre_SStructStencilRefCount(stencil) ++;
   *stencil_ref = stencil;

   return 0;
}

