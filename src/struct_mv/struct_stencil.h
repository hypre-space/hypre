/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for hypre_StructStencil data structures
 *
 *****************************************************************************/

#ifndef hypre_STRUCT_STENCIL_HEADER
#define hypre_STRUCT_STENCIL_HEADER

/*--------------------------------------------------------------------------
 * hypre_StructStencil
 *--------------------------------------------------------------------------*/

typedef struct hypre_StructStencil_struct
{
   hypre_Index   *shape;   /* Description of a stencil's shape */
   HYPRE_Int      size;    /* Number of stencil coefficients */

   HYPRE_Int      ndim;    /* Number of dimensions */

   HYPRE_Int      ref_count;
} hypre_StructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_StructStencil structure
 *--------------------------------------------------------------------------*/

#define hypre_StructStencilShape(stencil)      ((stencil) -> shape)
#define hypre_StructStencilSize(stencil)       ((stencil) -> size)
#define hypre_StructStencilNDim(stencil)       ((stencil) -> ndim)
#define hypre_StructStencilRefCount(stencil)   ((stencil) -> ref_count)
#define hypre_StructStencilElement(stencil, i) hypre_StructStencilShape(stencil)[i]

#endif
