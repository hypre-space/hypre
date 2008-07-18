/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




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
