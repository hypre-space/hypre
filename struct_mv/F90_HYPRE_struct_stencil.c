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
 * HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_createstructstencil)( int      *dim,
                                            int      *size,
                                            long int *stencil,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructStencilCreate( (int)                   *dim,
                                   (int)                   *size,
                                   (HYPRE_StructStencil *)  stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setstructstencilelement)( long int *stencil,
                                                int      *element_index,
                                                int      *offset,
                                                int      *ierr          )
{
   *ierr = (int)
      ( HYPRE_StructStencilSetElement( (HYPRE_StructStencil) *stencil,
                                       (int)                 *element_index,
                                       (int *)                offset       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_destroystructstencil)( long int *stencil,
                                             int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructStencilDestroy( (HYPRE_StructStencil) *stencil ) );
}
