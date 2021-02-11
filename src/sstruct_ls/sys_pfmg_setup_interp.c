/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#if 0
hypre_SStructPMatrix*
hypre_SysPFMGCreateInterpOp( hypre_SStructPMatrix *A,
                             hypre_SStructPGrid   *cgrid,
                             HYPRE_Int             cdir  )
{
   MPI_Comm               comm  = hypre_SStructPMatrixComm(A);
   HYPRE_Int              ndim  = hypre_SStructPMatrixNDim(A);
   HYPRE_Int              nvars = hypre_SStructPMatrixNVars(A);
   hypre_SStructPMatrix  *P;

   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size_P;
   HYPRE_Int              stencil_size_A;
   hypre_SStructStencil **P_stencils;

   hypre_StructMatrix    *A_s;
   hypre_StructStencil   *stencil;

   HYPRE_Int              centries[3] = {0, 1, 2};
   HYPRE_Int             *num_centries;

   HYPRE_Int              i, s, vi;
   hypre_Index            cstride;
   hypre_Index            fstride;

   /* Coarsening in direction cdir by a factor of 2 */
   hypre_SetIndex(fstride, 1);
   hypre_SetIndex(cstride, 1);
   hypre_IndexD(cstride, cdir) = 2;

   /* Set up the stencil for P(vi, vi) */
   stencil_size_P = 3;
   num_centries  = hypre_CTAlloc(HYPRE_Int, stencil_size_P, HYPRE_MEMORY_HOST);
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size_P, HYPRE_MEMORY_HOST);
   P_stencils    = hypre_CTAlloc(hypre_SStructStencil *, nvars, HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size_P; i++)
   {
      hypre_SetIndex(stencil_shape[i], 0);
   }
   hypre_IndexD(stencil_shape[1], cdir) = -1;
   hypre_IndexD(stencil_shape[2], cdir) =  1;

   for (vi = 0; vi < nvars; vi++)
   {
      A_s = hypre_SStructPMatrixSMatrix(A, vi, vi);
      stencil = hypre_StructMatrixStencil(A_s);
      stencil_size_A = hypre_StructStencilSize(stencil);

      /* Create stencil for P(vi) */
      HYPRE_SStructStencilCreate(ndim, stencil_size_P, &P_stencils[vi]);
      for (s = 0; s < stencil_size_P; s++)
      {
         HYPRE_SStructStencilSetEntry(P_stencils[vi], s, stencil_shape[s], vi);
      }

      /* Figure out which entries to make constant (ncentries) */
      num_centries[vi] = 3;
      for (i = 0; i < stencil_size_A; i++)
      {
         /* Check for entries in A in direction cdir that are variable */
         if (hypre_IndexD(hypre_StructStencilOffset(stencil, i), cdir) != 0)
         {
            if (!hypre_StructMatrixConstEntry(A_s, i))
            {
               num_centries[nvars] = 1; /* Make only the diagonal of P constant */
               break;
            }
         }
      }
   }

   hypre_SStructPMatrixCreate(comm, cgrid, P_stencils, &P);
   for (vi = 0; vi < nvars; vi++)
   {
      hypre_SStructPMatrixSetSymmetric(P, vi, vi, 0);
      hypre_SStructPMatrixSetCEntries(P, vi, vi, num_centries[vi], centries);
   }
   hypre_SStructPMatrixSetDomainStride(P, cstride);
   hypre_SStructPMatrixSetRangeStride(P, fstride);
   hypre_SStructPMatrixInitialize(P);
   hypre_SStructPMatrixAssemble(P);

   /* Free memory */
   hypre_TFree(num_centries, HYPRE_MEMORY_HOST);
   hypre_TFree(stencil_shape, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      hypre_TFree(P_stencils[vi], HYPRE_MEMORY_HOST);
   }

   return P;
}
#else
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

   HYPRE_Int              i,s;

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
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetupInterpOp( hypre_SStructPMatrix *A,
                            HYPRE_Int             cdir,
                            hypre_Index           findex,
                            hypre_Index           stride,
                            hypre_SStructPMatrix *P      )
{
   HYPRE_Int              nvars = hypre_SStructPMatrixNVars(A);
   hypre_StructMatrix    *A_s;
   hypre_StructMatrix    *P_s;
   HYPRE_Int              vi;

   for (vi = 0; vi < nvars; vi++)
   {
      A_s = hypre_SStructPMatrixSMatrix(A, vi, vi);
      P_s = hypre_SStructPMatrixSMatrix(P, vi, vi);
      hypre_PFMGSetupInterpOp(A_s, cdir, findex, stride, P_s, 0);
      //hypre_zPFMGSetupInterpOp(P_s, A_s, cdir);
   }

   return hypre_error_flag;
}
