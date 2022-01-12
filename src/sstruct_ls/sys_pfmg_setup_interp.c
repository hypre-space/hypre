/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_SStructPMatrix *
hypre_SysPFMGCreateInterpOp( hypre_SStructPMatrix *A,
                             hypre_SStructPGrid   *cgrid,
                             HYPRE_Int             cdir  )
{
   hypre_SStructPMatrix  *P;

   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;

   HYPRE_Int              ndim;

   HYPRE_Int              nvars;
   hypre_SStructStencil **P_stencils;

   HYPRE_Int              i, s;

   /* set up stencil_shape */
   stencil_size = 2;
   stencil_shape = hypre_CTAlloc(hypre_Index,  stencil_size, HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_SetIndex3(stencil_shape[i], 0, 0, 0);
   }
   hypre_IndexD(stencil_shape[0], cdir) = -1;
   hypre_IndexD(stencil_shape[1], cdir) =  1;

   /* set up P_stencils */
   ndim = hypre_StructStencilNDim(hypre_SStructPMatrixSStencil(A, 0, 0));
   nvars = hypre_SStructPMatrixNVars(A);
   P_stencils = hypre_CTAlloc(hypre_SStructStencil *,  nvars, HYPRE_MEMORY_HOST);
   for (s = 0; s < nvars; s++)
   {
      HYPRE_SStructStencilCreate(ndim, stencil_size, &P_stencils[s]);
      for (i = 0; i < stencil_size; i++)
      {
         HYPRE_SStructStencilSetEntry(P_stencils[s], i,
                                      stencil_shape[i], s);
      }
   }

   /* create interpolation matrix */
   hypre_SStructPMatrixCreate(hypre_SStructPMatrixComm(A), cgrid,
                              P_stencils, &P);

   hypre_TFree(stencil_shape, HYPRE_MEMORY_HOST);

   return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetupInterpOp( hypre_SStructPMatrix *A,
                            HYPRE_Int             cdir,
                            hypre_Index           findex,
                            hypre_Index           stride,
                            hypre_SStructPMatrix *P      )
{
   HYPRE_Int              nvars;
   hypre_StructMatrix    *A_s;
   hypre_StructMatrix    *P_s;
   HYPRE_Int              vi;

   nvars = hypre_SStructPMatrixNVars(A);

   for (vi = 0; vi < nvars; vi++)
   {
      A_s = hypre_SStructPMatrixSMatrix(A, vi, vi);
      P_s = hypre_SStructPMatrixSMatrix(P, vi, vi);
      hypre_PFMGSetupInterpOp(A_s, cdir, findex, stride, P_s, 0);
   }

   return hypre_error_flag;
}
