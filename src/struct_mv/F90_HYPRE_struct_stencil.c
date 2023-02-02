/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
