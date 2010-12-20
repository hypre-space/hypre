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
   (HYPRE_Int      *ndim,
    HYPRE_Int      *size,
    hypre_F90_Obj *stencil_ptr,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructStencilCreate(
                     (HYPRE_Int)                   *ndim,
                     (HYPRE_Int)                   *size,
                     (HYPRE_SStructStencil *) stencil_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencildestroy, HYPRE_SSTRUCTSTENCILDESTROY)
   (hypre_F90_Obj *stencil,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructStencilDestroy(
                     (HYPRE_SStructStencil) *stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilSetEntry
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencilsetentry, HYPRE_SSTRUCTSTENCILSETENTRY)
   (hypre_F90_Obj *stencil,
    HYPRE_Int      *entry,
    HYPRE_Int      *offset,
    HYPRE_Int      *var,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructStencilSetEntry(
                     (HYPRE_SStructStencil) *stencil,
                     (HYPRE_Int)                  *entry,
                     (HYPRE_Int *)                 offset,
                     (HYPRE_Int)                  *var ) );
}
