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
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_NewStructStencil
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_newstructstencil)( int      *dim,
                                         int      *size,
                                         long int *stencil,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_NewStructStencil( (int)                   *dim,
                                           (int)                   *size,
                                           (HYPRE_StructStencil *)  stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructStencilElement
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setstructstencilelement)( long int *stencil,
                                                int      *element_index,
                                                int      *offset,
                                                int      *ierr          )
{
   *ierr = (int) ( HYPRE_SetStructStencilElement( (HYPRE_StructStencil) *stencil,
                                                  (int)                 *element_index,
                                                  (int *)               offset         ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructStencil
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_freestructstencil)( long int *stencil,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_FreeStructStencil( (HYPRE_StructStencil) *stencil ) );
}
