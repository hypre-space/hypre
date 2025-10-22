/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_SStructPMatrix *
hypre_SysPFMGCreateInterpOp( hypre_SStructPMatrix *A,
                             HYPRE_Int             cdir,
                             hypre_Index           stride )
{
   hypre_SStructPMatrix  *P;

   HYPRE_Int              rap_type = 0;
   hypre_StructMatrix    *sA;
   hypre_StructMatrix    *sP;

   HYPRE_Int              stencil_size;
   hypre_Index           *stencil_shape;

   HYPRE_Int              nvars, ndim, i, s, vi;
   hypre_SStructStencil **P_stencils;
   HYPRE_Int              centries[1] = {0};

   /* Create struct interpolation matrix sP first */
   sA = hypre_SStructPMatrixSMatrix(A, 0, 0);
   sP = hypre_PFMGCreateInterpOp(sA, cdir, stride, rap_type);
   hypre_StructMatrixInitializeShell(sP);  /* Don't allocate data */

   stencil_size  = hypre_StructStencilSize(hypre_StructMatrixStencil(sP));
   stencil_shape = hypre_StructStencilShape(hypre_StructMatrixStencil(sP));

   /* Set up P_stencils */
   nvars = hypre_SStructPMatrixNVars(A);
   ndim  = hypre_SStructPMatrixNDim(A);
   P_stencils = hypre_CTAlloc(hypre_SStructStencil *,  nvars, HYPRE_MEMORY_HOST);
   for (s = 0; s < nvars; s++)
   {
      HYPRE_SStructStencilCreate(ndim, stencil_size, &P_stencils[s]);
      for (i = 0; i < stencil_size; i++)
      {
         HYPRE_SStructStencilSetEntry(P_stencils[s], i, stencil_shape[i], s);
      }
   }

   /* Set up the P matrix */
   hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A),
                              hypre_SStructPMatrixPGrid(A), P_stencils, &P);
   hypre_SStructPMatrixSetDomainStride(P, stride);

   /* Make the diagonal constant */
   for (vi = 0; vi < nvars; vi++)
   {
      hypre_StructMatrixSetConstantEntries(hypre_SStructPMatrixSMatrix(P, vi, vi), 1, centries);
   }

   hypre_StructMatrixDestroy(sP);

   return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetupInterpOp( hypre_SStructPMatrix *P,
                            hypre_SStructPMatrix *A,
                            HYPRE_Int             cdir )
{
   HYPRE_Int  nvars = hypre_SStructPMatrixNVars(P);
   HYPRE_Int  vi;

   for (vi = 0; vi < nvars; vi++)
   {
      hypre_PFMGSetupInterpOp(hypre_SStructPMatrixSMatrix(P, vi, vi),
                              hypre_SStructPMatrixSMatrix(A, vi, vi), cdir);
   }

   return hypre_error_flag;
}
