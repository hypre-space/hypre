/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for hypre_SStructStencil data structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_STENCIL_HEADER
#define hypre_SSTRUCT_STENCIL_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructStencil
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructStencil_struct
{
   hypre_StructStencil  *sstencil;
   HYPRE_Int            *vars;

   HYPRE_Int             ref_count;

} hypre_SStructStencil;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_SStructStencil structure
 *--------------------------------------------------------------------------*/

#define hypre_SStructStencilSStencil(stencil)     ((stencil) -> sstencil)
#define hypre_SStructStencilVars(stencil)         ((stencil) -> vars)
#define hypre_SStructStencilVar(stencil, i)       ((stencil) -> vars[i])
#define hypre_SStructStencilRefCount(stencil)     ((stencil) -> ref_count)

#define hypre_SStructStencilShape(stencil) \
hypre_StructStencilShape( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilSize(stencil) \
hypre_StructStencilSize( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilNDim(stencil) \
hypre_StructStencilNDim( hypre_SStructStencilSStencil(stencil) )
#define hypre_SStructStencilEntry(stencil, i) \
hypre_StructStencilElement( hypre_SStructStencilSStencil(stencil), i )

#endif
