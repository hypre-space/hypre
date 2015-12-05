/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
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
   (hypre_F90_Int *ndim,
    hypre_F90_Int *size,
    hypre_F90_Obj *stencil_ptr,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructStencilCreate(
          hypre_F90_PassInt (ndim),
          hypre_F90_PassInt (size),
          hypre_F90_PassObjRef (HYPRE_SStructStencil, stencil_ptr) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencildestroy, HYPRE_SSTRUCTSTENCILDESTROY)
   (hypre_F90_Obj *stencil,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructStencilDestroy(
          hypre_F90_PassObj (HYPRE_SStructStencil, stencil) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructStencilSetEntry
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencilsetentry, HYPRE_SSTRUCTSTENCILSETENTRY)
   (hypre_F90_Obj *stencil,
    hypre_F90_Int *entry,
    hypre_F90_IntArray *offset,
    hypre_F90_Int *var,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructStencilSetEntry(
          hypre_F90_PassObj (HYPRE_SStructStencil, stencil),
          hypre_F90_PassInt (entry),
          hypre_F90_PassIntArray (offset),
          hypre_F90_PassInt (var) ) );
}
