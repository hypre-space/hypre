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
 * HYPRE_SStructStencil interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencilcreate, HYPRE_SSTRUCTSTENCILCREATE)
                                                                (int      *ndim,
                                                                 int      *size,
                                                                 long int *stencil_ptr,
                                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructStencilCreate( (int)                   *ndim,
                                              (int)                   *size,
                                              (HYPRE_SStructStencil *) stencil_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencildestroy, HYPRE_SSTRUCTSTENCILDESTROY)
                                                                (long int *stencil,
                                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructStencilDestroy( (HYPRE_SStructStencil) *stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilSetEntry
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencilsetentry, HYPRE_SSTRUCTSTENCILSETENTRY)
                                                                (long int *stencil,
                                                                 int      *entry,
                                                                 int      *offset,
                                                                 int      *var,
                                                                 int      *ierr)
{
   *ierr = (int) (HYPRE_SStructStencilSetEntry( (HYPRE_SStructStencil) *stencil,
                                                 (int)                 *entry,
                                                 (int *)                offset,
                                                 (int)                 *var ) );
}
