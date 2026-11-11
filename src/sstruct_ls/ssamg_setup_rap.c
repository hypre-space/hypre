/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

static HYPRE_Int
hypre_SSAMGStencilIsPFMG( hypre_StructMatrix *A )
{
   hypre_StructStencil *stencil    = hypre_StructMatrixUserStencil(A);
   HYPRE_Int            ndim       = hypre_StructStencilNDim(stencil);
   HYPRE_Int            st_size    = hypre_StructStencilSize(stencil);
   HYPRE_Int            axis[2 * HYPRE_MAXDIM] = {0};
   HYPRE_Int            num_center = 0;
   HYPRE_Int            s, d, offset, entry_dir, entry_side;

   if (((ndim == 2) && (st_size != 5)) ||
       ((ndim == 3) && (st_size != 7)))
   {
      return 0;
   }

   for (s = 0; s < st_size; s++)
   {
      entry_dir  = -1;
      entry_side = -1;
      for (d = 0; d < ndim; d++)
      {
         offset = hypre_IndexD(hypre_StructStencilOffset(stencil, s), d);
         if (offset != 0)
         {
            if ((entry_dir > -1) || ((offset != -1) && (offset != 1)))
            {
               return 0;
            }
            entry_dir  = d;
            entry_side = (offset > 0) ? 1 : 0;
         }
      }

      if (entry_dir < 0)
      {
         num_center++;
      }
      else
      {
         axis[2 * entry_dir + entry_side]++;
      }
   }

   if (num_center != 1)
   {
      return 0;
   }
   for (d = 0; d < ndim; d++)
   {
      if ((axis[2 * d] != 1) || (axis[2 * d + 1] != 1))
      {
         return 0;
      }
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeRAP
 *
 *   Wrapper for 2D and 3D RAP routines which sets up new coarse
 *   grid structures for SSAMG.
 *
 *   if the non_galerkin option is turned on, then use the PARFLOW formula
 *   for computing the coarse grid operator (works only with 5pt stencils in
 *   2D and 7pt stencils in 3D). If non_galerkin is turned off, then it uses
 *   the general purpose matrix-matrix multiplication function (SStructMatmult)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGComputeRAP( hypre_SStructMatrix    *A,
                       hypre_SStructMatrix    *P,
                       hypre_SStructGrid     **cgrid,
                       HYPRE_Int              *cdir_p,
                       HYPRE_Int               non_galerkin,
                       hypre_SStructMatrix   **Ac_ptr )
{
   hypre_SStructMatrix *Ac;
   hypre_SStructGraph  *graph;
   hypre_SStructGrid   *grid;

   if (non_galerkin)
   {
      hypre_SSAMGComputeRAPNonGlk(A, P, cdir_p, &Ac);
   }
   else
   {
      hypre_SStructMatrixPtAP(A, P, &Ac);
   }

   /* Update grid object */
   graph = hypre_SStructMatrixGraph(Ac);
   grid  = hypre_SStructGraphGrid(graph);
   HYPRE_SStructGridDestroy(*cgrid);
   hypre_SStructGridRef(grid, cgrid);

   /* Update pointer to resulting matrix */
   *Ac_ptr = Ac;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeRAPNonGlk
 *
 * Notes:
 *        1) The PFMG formula is applied only to supported 5/7-point structured
 *           diagonal blocks with a valid coarsening direction. Other structured
 *           blocks retain the general SStruct product.
 *        2) Multivariable off-diagonal blocks retain the general PtAP result.
 *        3) The unstructured component is computed after replacing structured
 *           blocks so overlapping entries are compressed into the final S part.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SSAMGComputeRAPNonGlk( hypre_SStructMatrix  *A,
                             hypre_SStructMatrix  *P,
                             HYPRE_Int            *cdir_p,
                             hypre_SStructMatrix **Ac_ptr )
{
   hypre_SStructMatmultData *mmdata;
   hypre_SStructMatrix      *matrices[2] = {A, P};
   HYPRE_Int                 terms[3]    = {1, 0, 1};
   HYPRE_Int                 trans[3]    = {1, 0, 0};
   hypre_SStructMatrix      *Ac;
   hypre_SStructPMatrix     *pA, *pP, *pAc;
   hypre_StructMatrix       *sA, *sP, *sAc, *sAc_old;
   hypre_StructStencil      *stencil_Ac, *stencil_old;
   hypre_SStructPGrid       *pcgrid;
   hypre_StructGrid         *scgrid;
   hypre_Index               cindex;
   hypre_Index               cstride;
   HYPRE_Int                 nparts, part, nvars, vi, cdir;
   HYPRE_Int                 s, d, stencil_matches;

   /* Build the general structured product first. The U component is computed
    * after the optimized blocks below are installed. */
   hypre_SStructMatmultCreate(2, matrices, 3, terms, trans, &mmdata);
   hypre_SStructMatmultInitialize(mmdata, &Ac);
   hypre_SStructMatmultCommunicate(mmdata);
   hypre_SStructMatmultComputeS(mmdata, Ac);

   hypre_SetIndex(cindex, 0);
   nparts = hypre_SStructMatrixNParts(A);
   for (part = 0; part < nparts; part++)
   {
      cdir = cdir_p[part];
      if (cdir < 0)
      {
         continue;
      }

      pA    = hypre_SStructMatrixPMatrix(A, part);
      pP    = hypre_SStructMatrixPMatrix(P, part);
      pAc   = hypre_SStructMatrixPMatrix(Ac, part);
      nvars = hypre_SStructPMatrixNVars(pA);
      pcgrid = hypre_SStructPMatrixPGrid(pAc);

      hypre_SetIndex(cstride, 1);
      hypre_IndexD(cstride, cdir) = 2;

      for (vi = 0; vi < nvars; vi++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sP = hypre_SStructPMatrixSMatrix(pP, vi, vi);
         if (!hypre_SSAMGStencilIsPFMG(sA))
         {
            continue;
         }

         scgrid = hypre_SStructPGridSGrid(pcgrid, vi);
         sAc = hypre_PFMGCreateRAPOp(sP, sA, sP, scgrid, cdir, 1);
         if (!sAc)
         {
            continue;
         }
         hypre_StructMatrixInitialize(sAc);
         hypre_PFMGSetupRAPOp(sP, sA, sP, cdir, cindex, cstride, 1, sAc);

         /* Keep the general product block if the optimized operator changes the
          * coarse stencil layout. This should not happen for supported PFMG
          * stencils, but avoids attaching an incompatible matrix. */
         sAc_old = hypre_SStructPMatrixSMatrix(pAc, vi, vi);
         stencil_Ac = hypre_StructMatrixStencil(sAc);
         stencil_old = hypre_StructMatrixStencil(sAc_old);
         stencil_matches =
            (hypre_StructStencilSize(stencil_Ac) ==
             hypre_StructStencilSize(stencil_old));
         if (stencil_matches)
         {
            for (s = 0; s < hypre_StructStencilSize(stencil_Ac); s++)
            {
               for (d = 0; d < hypre_SStructMatrixNDim(A); d++)
               {
                  if (hypre_IndexD(hypre_StructStencilOffset(stencil_Ac, s), d) !=
                      hypre_IndexD(hypre_StructStencilOffset(stencil_old, s), d))
                  {
                     stencil_matches = 0;
                  }
               }
            }
         }

         if (stencil_matches)
         {
            hypre_StructMatrixDestroy(sAc_old);
            hypre_SStructPMatrixSMatrix(pAc, vi, vi) = sAc;
            hypre_SStructPMatrixAccumulated(pAc) = 0;
            sAc = NULL;
         }
         else
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "PFMG RAP stencil differs from SStruct PtAP stencil");
         }
         if (sAc)
         {
            hypre_StructMatrixDestroy(sAc);
         }
      }
   }

   /* Move overlapping U entries into the final structured representation. */
   hypre_SStructMatmultComputeU(mmdata, Ac);
   hypre_SStructMatmultDestroy(mmdata);
   HYPRE_SStructMatrixAssemble(Ac);
   *Ac_ptr = Ac;

   return hypre_error_flag;
}
