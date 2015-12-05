/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/

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
hypre_F90_IFACE(hypre_structstencilcreate, HYPRE_STRUCTSTENCILCREATE)
   ( hypre_F90_Int *dim,
     hypre_F90_Int *size,
     hypre_F90_Obj *stencil,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int) HYPRE_StructStencilCreate(
      hypre_F90_PassInt (dim),
      hypre_F90_PassInt (size),
      hypre_F90_PassObjRef (HYPRE_StructStencil, stencil) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencilsetelement, HYPRE_STRUCTSTENCILSETELEMENT)
   ( hypre_F90_Obj *stencil,
     hypre_F90_Int *element_index,
     hypre_F90_IntArray *offset,
     hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int) HYPRE_StructStencilSetElement(
      hypre_F90_PassObj (HYPRE_StructStencil, stencil),
      hypre_F90_PassInt (element_index),
      hypre_F90_PassIntArray (offset)       );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencildestroy, HYPRE_STRUCTSTENCILDESTROY)
   ( hypre_F90_Obj *stencil,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int) HYPRE_StructStencilDestroy(
      hypre_F90_PassObj (HYPRE_StructStencil, stencil) );
}
