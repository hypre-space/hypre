/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

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
 *        1) Multivariable version not implemented.
 *        2) Needs debugging.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SSAMGComputeRAPNonGlk( hypre_SStructMatrix  *A,
                             hypre_SStructMatrix  *P,
                             HYPRE_Int            *cdir_p,
                             hypre_SStructMatrix **Ac_ptr )
{
   MPI_Comm                 comm   = hypre_SStructMatrixComm(A);
   HYPRE_Int                ndim   = hypre_SStructMatrixNDim(A);
   hypre_SStructGraph      *graph  = hypre_SStructMatrixGraph(P);
   HYPRE_Int                nparts = hypre_SStructGraphNParts(graph);
   hypre_SStructGrid       *cgrid  = hypre_SStructGraphDomGrid(graph);

   hypre_StructStencil     *stencil;
   hypre_SStructGraph      *cgraph;
   hypre_SStructPGrid      *pcgrid;
   hypre_StructGrid        *scgrid;
   hypre_StructMatrix   ****sAc;
   hypre_SStructStencil    *st_Ac;
   hypre_StructMatrix      *sA, *sP;
   hypre_SStructPMatrix    *pA, *pP, *pAc;
   hypre_SStructMatrix     *Ac;
   hypre_IJMatrix          *ij_Ac;
   //hypre_ParCSRMatrix      *parcsr_uAc;

   //hypre_SStructMatrix     *ssmatrices[3] = {A, P, P};
   HYPRE_Int                terms[3] = {1, 0, 1};
   HYPRE_Int                trans[3] = {1, 0, 0};

   hypre_Index              cindex;
   hypre_Index              cstride;
   hypre_Index             *st_shape;
   HYPRE_Int                cdir;
   HYPRE_Int                nvars;
   HYPRE_Int                st_size;
   HYPRE_Int                part, s, vi, vj;

   /* Allocate memory */
   sAc = hypre_TAlloc(hypre_StructMatrix ***, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pcgrid = hypre_SStructGridPGrid(cgrid, part);
      nvars  = hypre_SStructPGridNVars(pcgrid);

      sAc[part] = hypre_TAlloc(hypre_StructMatrix **, nvars, HYPRE_MEMORY_HOST);
      for (vi = 0; vi < nvars; vi++)
      {
         sAc[part][vi] = hypre_TAlloc(hypre_StructMatrix *, nvars, HYPRE_MEMORY_HOST);
      }
   }

   /* Create SStructGraph of Ac */
   HYPRE_SStructGraphCreate(comm, cgrid, &cgraph);
   HYPRE_SStructGraphSetObjectType(cgraph, HYPRE_SSTRUCT);

   /* Compute struct component of Ac */
   hypre_SetIndex(cindex, 0);
   for (part = 0; part < nparts; part++)
   {
      cdir   = cdir_p[part];
      pA     = hypre_SStructMatrixPMatrix(A, part);
      pP     = hypre_SStructMatrixPMatrix(P, part);
      nvars  = hypre_SStructPMatrixNVars(pA);
      pcgrid = hypre_SStructGridPGrid(cgrid, part);

      hypre_SetIndex(cstride, 1);
      hypre_IndexD(cstride, cdir) = 2;
      for (vi = 0; vi < nvars; vi++)
      {
         sA      = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sP      = hypre_SStructPMatrixSMatrix(pP, vi, vi);
         scgrid  = hypre_SStructPGridSGrid(pcgrid, vi);
         stencil = hypre_StructMatrixUserStencil(sA);
         st_size = hypre_StructStencilSize(stencil);

         if (st_size == 5 || st_size == 7)
         {
            sAc[part][vi][vi] = hypre_PFMGCreateRAPOp(sP, sA, sP, scgrid, cdir, 1);
            hypre_StructMatrixInitialize(sAc[part][vi][vi]);
            hypre_PFMGSetupRAPOp(sP, sA, sP, cdir, cindex, cstride, 1, sAc[part][vi][vi]);
         }
         else
         {
            /* Use generic StructMatmult */
            hypre_StructMatrix *smatrices[3] = {sA, sP, sP};

            hypre_StructMatmult(3, smatrices, 3, terms, trans, &sAc[part][vi][vi]);
         }

         /* Create SStructStencil object for M */
         stencil  = hypre_StructMatrixStencil(sAc[part][vi][vi]);
         st_size  = hypre_StructStencilSize(stencil);
         st_shape = hypre_StructStencilShape(stencil);

         HYPRE_SStructStencilCreate(ndim, st_size, &st_Ac);
         for (s = 0; s < st_size; s++)
         {
            HYPRE_SStructStencilSetEntry(st_Ac, s, st_shape[s], vi);
         }
         HYPRE_SStructGraphSetStencil(cgraph, part, vi, st_Ac);
         HYPRE_SStructStencilDestroy(st_Ac);
      }
   }

   /* Compute unstructured component of Ac */
   //call to hypre_SStructMatrixMultComputeU

   /* Assemble SStructGraph */
   HYPRE_SStructGraphAssemble(cgraph);

   /* Create Ac */
   HYPRE_SStructMatrixCreate(comm, cgraph, &Ac);
   HYPRE_SStructMatrixInitialize(Ac);
   for (part = 0; part < nparts; part++)
   {
      pAc = hypre_SStructMatrixPMatrix(Ac, part);
      for (vi = 0; vi < nvars; vi++)
      {
         hypre_StructMatrixDestroy(hypre_SStructPMatrixSMatrix(pAc, vi, vi));
         hypre_SStructPMatrixSMatrix(pAc, vi, vi) = hypre_StructMatrixRef(sAc[part][vi][vi]);
      }
   }
   ij_Ac = hypre_SStructMatrixIJMatrix(Ac);
   hypre_IJMatrixDestroyParCSR(ij_Ac);
   hypre_IJMatrixObject(ij_Ac) = NULL;
   hypre_IJMatrixTranslator(ij_Ac) = NULL;
   hypre_IJMatrixAssembleFlag(ij_Ac) = 1;
   //hypre_IJMatrixSetObject(ij_Ac, parcsr_uAc);
   HYPRE_SStructMatrixAssemble(Ac);

   /* Set pointer to Ac */
   *Ac_ptr = Ac;

   /* Free memory */
   HYPRE_SStructGraphDestroy(cgraph);
   for (part = 0; part < nparts; part++)
   {
      pcgrid = hypre_SStructGridPGrid(cgrid, part);
      nvars  = hypre_SStructPGridNVars(pcgrid);

      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            hypre_StructMatrixDestroy(sAc[part][vi][vj]);
         }
         hypre_TFree(sAc[part][vi], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(sAc[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(sAc, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
