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
 * TODO:
 *       1) Consider inter-variable coupling
 *       2) Do we need to Destroy SStructStencil?
 *--------------------------------------------------------------------------*/

hypre_SStructMatrix *
hypre_SSAMGCreateInterpOp( hypre_SStructMatrix  *A,
                           hypre_SStructGrid    *cgrid,
                           HYPRE_Int            *cdir_p)
{
   MPI_Comm                 comm;
   hypre_SStructMatrix     *P;
   hypre_SStructPMatrix    *pmatrix;
   hypre_StructMatrix      *smatrix;
   hypre_SStructGraph      *graph_A;
   hypre_SStructGraph      *graph_P;
   hypre_SStructGrid       *grid_A;
   hypre_SStructStencil  ***stencils_P;
   hypre_StructStencil     *stencil;

   hypre_Index             *dom_strides;
   HYPRE_Int                cdir;
   HYPRE_Int                i, s, part, vi, ndim;
   HYPRE_Int                nparts, nvars;

   HYPRE_Int                centries[3] = {0, 1, 2};
   HYPRE_Int               *num_centries;

   HYPRE_Int                stencil_size_A;
   HYPRE_Int                stencil_size_P;
   hypre_Index            **st_shape;

   /*-------------------------------------------------------
    * Initialize some variables
    *-------------------------------------------------------*/
   comm     = hypre_SStructMatrixComm(A);
   ndim     = hypre_SStructMatrixNDim(A);
   nparts   = hypre_SStructMatrixNParts(A);
   graph_A  = hypre_SStructMatrixGraph(A);
   grid_A   = hypre_SStructGraphGrid(graph_A);

   st_shape     = hypre_CTAlloc(hypre_Index *, nparts);
   dom_strides  = hypre_CTAlloc(hypre_Index, nparts);
   num_centries = hypre_CTAlloc(HYPRE_Int, nparts);

   /*-------------------------------------------------------
    * Create SStructGraph data structure for P
    *-------------------------------------------------------*/
   HYPRE_SStructGraphCreate(comm, cgrid, grid_A, &graph_P);
   stencils_P = hypre_SStructGraphStencils(graph_P);

   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(A, part);
      nvars   = hypre_SStructPMatrixNVars(pmatrix);
      cdir    = cdir_p[part];

      hypre_SetIndex(dom_strides[part], 1);
      if (cdir > -1)
      {
         // Coarsening in direction cdir by a factor of 2
         hypre_IndexD(dom_strides[part], cdir) = 2;

         stencil_size_P     = 3;
         st_shape[part]     = hypre_CTAlloc(hypre_Index, stencil_size_P);
         num_centries[part] = stencil_size_P;
         for (vi = 0; vi < nvars; vi++)
         {
            smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vi);
            stencil = hypre_StructMatrixStencil(smatrix);
            stencil_size_A = hypre_StructStencilSize(stencil);

            /* Set up the stencil for P(part, vi, vi) */
            for (i = 0; i < stencil_size_P; i++)
            {
               hypre_SetIndex(st_shape[part][i], 0);
            }
            hypre_IndexD(st_shape[part][1], cdir) = -1;
            hypre_IndexD(st_shape[part][2], cdir) =  1;

            /* Create stencil for P(part, vi) */
            HYPRE_SStructStencilCreate(ndim, stencil_size_P, &stencils_P[part][vi]);
            for (s = 0; s < stencil_size_P; s++)
            {
               HYPRE_SStructStencilSetEntry(stencils_P[part][vi], s, st_shape[part][s], vi);
            }

            /* Figure out which entries to make constant (ncentries) */
            for (i = 0; i < stencil_size_A; i++)
            {
               /* Check for entries in A in direction cdir that are variable */
               if (hypre_IndexD(hypre_StructStencilOffset(stencil, i), cdir) != 0)
               {
                  if (!hypre_StructMatrixConstEntry(smatrix, i))
                  {
                     num_centries[part] = 1; /* Make only the diagonal of P constant */
                     break;
                  }
               }
            }
         }
      }
      else
      {
         stencil_size_P     = 1;
         st_shape[part]     = hypre_CTAlloc(hypre_Index, stencil_size_P);
         num_centries[part] = stencil_size_P;
         for (vi = 0; vi < nvars; vi++)
         {
            /* Create stencil for P(part, vi) */
            HYPRE_SStructStencilCreate(ndim, stencil_size_P, &stencils_P[part][vi]);
            for (s = 0; s < stencil_size_P; s++)
            {
               HYPRE_SStructStencilSetEntry(stencils_P[part][vi], s, st_shape[part][s], vi);
            }
         }
      }
   }

   HYPRE_SStructGraphAssemble(graph_P);
   HYPRE_SStructMatrixCreate(comm, graph_P, &P);
   for (part = 0; part < nparts; part++)
   {
      HYPRE_SStructMatrixSetDomainStride(P, part, dom_strides[part]);
      HYPRE_SStructMatrixSetConstantEntries(P, part, -1, -1, num_centries[part], centries);
   }
   HYPRE_SStructMatrixInitialize(P);
   HYPRE_SStructMatrixAssemble(P);

   /* Free memory */
   hypre_TFree(dom_strides);
   hypre_TFree(num_centries);
   for (part = 0; part < nparts; part++)
   {
      hypre_TFree(st_shape[part]);
   }
   hypre_TFree(st_shape);

   return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetupInterpOp( hypre_SStructMatrix  *A,
                          HYPRE_Int            *cdir_p,
                          hypre_SStructMatrix  *P)
{
   hypre_SStructPMatrix    *A_p, *P_p;
   hypre_StructMatrix      *A_s, *P_s;

   HYPRE_Real              *Pcoef;
   HYPRE_Int                nparts, nvars;
   HYPRE_Int                cdir, part, vi;

   /*-------------------------------------------------------
    * Initialize some variables
    *-------------------------------------------------------*/
   nparts = hypre_SStructMatrixNParts(A);

   for (part = 0; part < nparts; part++)
   {
      A_p   = hypre_SStructMatrixPMatrix(A, part);
      P_p   = hypre_SStructMatrixPMatrix(P, part);
      nvars = hypre_SStructPMatrixNVars(P_p);
      cdir  = cdir_p[part];

      if (cdir > -1)
      {
         for (vi = 0; vi < nvars; vi++)
         {
            A_s = hypre_SStructPMatrixSMatrix(A_p, vi, vi);
            P_s = hypre_SStructPMatrixSMatrix(P_p, vi, vi);
            hypre_zPFMGSetupInterpOp(P_s, A_s, cdir);
         }
      }
      else
      {
         for (vi = 0; vi < nvars; vi++)
         {
            A_s = hypre_SStructPMatrixSMatrix(A_p, vi, vi);
            P_s = hypre_SStructPMatrixSMatrix(P_p, vi, vi);

            /* Set center coefficient to 1 */
            Pcoef = hypre_StructMatrixConstData(P_s, 0);
            Pcoef[0] = 1.0;

            /* Assemble matrix */
            hypre_StructMatrixAssemble(P_s);
         }
      }
   }

   return hypre_error_flag;
}
