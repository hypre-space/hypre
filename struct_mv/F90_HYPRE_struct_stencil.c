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
 * HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencilcreate, HYPRE_STRUCTSTENCILCREATE)( HYPRE_Int      *dim,
                                            HYPRE_Int      *size,
                                            hypre_F90_Obj *stencil,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructStencilCreate( (HYPRE_Int)                   *dim,
                                   (HYPRE_Int)                   *size,
                                   (HYPRE_StructStencil *)  stencil ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencilsetelement, HYPRE_STRUCTSTENCILSETELEMENT)( hypre_F90_Obj *stencil,
                                                HYPRE_Int      *element_index,
                                                HYPRE_Int      *offset,
                                                HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructStencilSetElement( (HYPRE_StructStencil) *stencil,
                                       (HYPRE_Int)                 *element_index,
                                       (HYPRE_Int *)                offset       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structstencildestroy, HYPRE_STRUCTSTENCILDESTROY)( hypre_F90_Obj *stencil,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructStencilDestroy( (HYPRE_StructStencil) *stencil ) );
}
