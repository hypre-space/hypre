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
hypre_SSAMGComputeRAP( hypre_SStructMatrix   *A,
                       hypre_SStructMatrix   *P,
                       hypre_SStructGrid     *cgrid,
                       HYPRE_Int             *cdir_p,
                       HYPRE_Int              non_galerkin,
                       hypre_SStructMatrix  **Ac_ptr )
{
   hypre_SStructMatrix *Ac;

   if (non_galerkin)
   {
      hypre_SSAMGComputeRAPNonGlk(A, P, cdir_p, &Ac);
   }
   else
   {
      hypre_SStructMatPtAP(A, P, &Ac);
   }

   *Ac_ptr = Ac;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGComputeRAPNonGlk
 *
 * Notes:
 *        1) Multivariable version not implemented.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SSAMGComputeRAPNonGlk( hypre_SStructMatrix  *A,
                             hypre_SStructMatrix  *P,
                             HYPRE_Int            *cdir_p,
                             hypre_SStructMatrix **Ac_ptr )
{
   MPI_Comm                 comm   = hypre_SStructMatrixComm(A);
   hypre_SStructGraph      *graph  = hypre_SStructMatrixGraph(P);
   HYPRE_Int                nparts = hypre_SStructGraphNParts(graph);
   hypre_SStructGrid       *cgrid  = hypre_SStructGraphDomGrid(graph);

   hypre_StructStencil     *stencil;
   hypre_SStructGraph      *cgraph;
   hypre_SStructPGrid      *pcgrid;
   hypre_StructGrid        *scgrid;
   hypre_StructMatrix   ****sAc;
   hypre_StructMatrix      *sA, *sP;
   hypre_SStructPMatrix    *pA, *pP, *pAc;
   hypre_SStructMatrix     *Ac;
   hypre_IJMatrix          *ij_Ac;
   hypre_ParCSRMatrix      *parcsr_uAc;

   hypre_SStructMatrix     *ssmatrices[3] = {A, P, P};
   HYPRE_Int                terms[3] = {1, 0, 1};
   HYPRE_Int                trans[3] = {1, 0, 0};

   hypre_Index              cindex;
   hypre_Index              cstride;
   HYPRE_Int                cdir;
   HYPRE_Int                nvars;
   HYPRE_Int                st_size;
   HYPRE_Int                part, vi, vj;

   /* Create SStructGraph of Ac */
   HYPRE_SStructGraphCreate(comm, cgrid, &cgraph);

   /* Allocate memory */
   sAc = hypre_TAlloc(hypre_StructMatrix ***, nparts);
   for (part = 0; part < nparts; part++)
   {
      pcgrid = hypre_SStructGridPGrid(cgrid, part);
      nvars  = hypre_SStructPGridNVars(pcgrid);

      sAc[part] = hypre_TAlloc(hypre_StructMatrix **, nvars);
      for (vi = 0; vi < nvars; vi++)
      {
         sAc[part][vi] = hypre_TAlloc(hypre_StructMatrix *, nvars);
      }
   }

   /* Compute struct component of Ac */
   hypre_SetIndex(cindex, 0);
   for (part = 0; part < nparts; part++)
   {
      cdir   = cdir_p[part];
      pA     = hypre_SStructMatrixPMatrix(A, part);
      pP     = hypre_SStructMatrixPMatrix(P, part);
      nvars  = hypre_SStructPMatrixNVars(pA);
      pcgrid = hypre_SStructGridPGrid(cgrid, part);

      if (cdir > -1)
      {
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
               hypre_PFMGSetupRAPOp(sP, sA, sP, cdir, cindex, cstride, 1, sAc[part][vi][vi]);
            }
            else
            {
               /* Use generic StructMatmult */
               hypre_StructMatrix *smatrices[3] = {sA, sP, sP};

               hypre_StructMatmult(3, smatrices, 3, terms, trans, NULL,
                                   &sAc[part][vi][vi]);
            }
         }
      }
   }

   /* Compute unstructured component of Ac */
   hypre_SStructMatmultU(3, ssmatrices, 3, terms, trans, &parcsr_uAc);

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
   hypre_IJMatrixSetObject(ij_Ac, parcsr_uAc);
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
         hypre_TFree(sAc[part][vi]);
      }
      hypre_TFree(sAc[part]);
   }
   hypre_TFree(sAc);

   return hypre_error_flag;
}
