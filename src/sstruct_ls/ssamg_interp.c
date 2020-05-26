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
   hypre_SStructGrid       *ran_grid;
   hypre_SStructStencil  ***stencils_P;
   hypre_StructStencil     *stencil;
   HYPRE_Int               *fids;
   HYPRE_Int               *cids;

   hypre_Index             *dom_strides;
   HYPRE_Int                cdir;
   HYPRE_Int                part_id, fpart, cpart;
   HYPRE_Int                fnparts, cnparts;
   HYPRE_Int                i, s, vi, ndim;
   HYPRE_Int                nvars;

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
   graph_A  = hypre_SStructMatrixGraph(A);

   ran_grid = hypre_SStructGraphGrid(graph_A);
   fnparts  = hypre_SStructGridNParts(ran_grid);
   fids     = hypre_SStructGridPartIDs(ran_grid);

   cnparts  = hypre_SStructGridNParts(cgrid);
   cids     = hypre_SStructGridPartIDs(cgrid);

   st_shape     = hypre_CTAlloc(hypre_Index *, cnparts);
   dom_strides  = hypre_CTAlloc(hypre_Index, cnparts);
   num_centries = hypre_CTAlloc(HYPRE_Int, cnparts);

   /*-------------------------------------------------------
    * Create SStructGraph data structure for P
    *-------------------------------------------------------*/
   HYPRE_SStructGraphCreate(comm, cgrid, ran_grid, &graph_P);
   stencils_P = hypre_SStructGraphStencils(graph_P);

   for (cpart = 0; cpart < cnparts; cpart++)
   {
      part_id = cids[cpart];
      fpart   = hypre_BinarySearch(fids, part_id, fnparts);
      cdir    = cdir_p[part_id];

      pmatrix = hypre_SStructMatrixPMatrix(A, fpart);
      nvars   = hypre_SStructPMatrixNVars(pmatrix);

      // Coarsening in direction cdir by a factor of 2
      hypre_SetIndex(dom_strides[cpart], 1);
      hypre_IndexD(dom_strides[cpart], cdir) = 2;

      stencil_size_P      = 3;
      st_shape[cpart]     = hypre_CTAlloc(hypre_Index, stencil_size_P);
      num_centries[cpart] = stencil_size_P;
      for (vi = 0; vi < nvars; vi++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vi);
         stencil = hypre_StructMatrixStencil(smatrix);
         stencil_size_A = hypre_StructStencilSize(stencil);

         /* Set up the stencil for P(cpart, vi, vi) */
         for (i = 0; i < stencil_size_P; i++)
         {
            hypre_SetIndex(st_shape[cpart][i], 0);
         }
         hypre_IndexD(st_shape[cpart][1], cdir) = -1;
         hypre_IndexD(st_shape[cpart][2], cdir) =  1;

         /* Create stencil for P(cpart, vi) */
         HYPRE_SStructStencilCreate(ndim, stencil_size_P, &stencils_P[cpart][vi]);
         for (s = 0; s < stencil_size_P; s++)
         {
            HYPRE_SStructStencilSetEntry(stencils_P[cpart][vi], s, st_shape[cpart][s], vi);
         }

         /* Figure out which entries to make constant (ncentries) */
         for (i = 0; i < stencil_size_A; i++)
         {
            /* Check for entries in A in direction cdir that are variable */
            if (hypre_IndexD(hypre_StructStencilOffset(stencil, i), cdir) != 0)
            {
               if (!hypre_StructMatrixConstEntry(smatrix, i))
               {
                  num_centries[cpart] = 1; /* Make only the diagonal of P constant */
                  break;
               }
            }
         }
      }
   }

   HYPRE_SStructGraphAssemble(graph_P);
   HYPRE_SStructMatrixCreate(comm, graph_P, &P);
   for (cpart = 0; cpart < cnparts; cpart++)
   {
      HYPRE_SStructMatrixSetDomainStride(P, cpart, dom_strides[cpart]);
      HYPRE_SStructMatrixSetConstantEntries(P, cpart, -1, -1, num_centries[cpart], centries);
   }
   HYPRE_SStructMatrixInitialize(P);
   HYPRE_SStructMatrixAssemble(P);

   /* Free memory */
   hypre_TFree(dom_strides);
   hypre_TFree(num_centries);
   for (cpart = 0; cpart < cnparts; cpart++)
   {
      hypre_TFree(st_shape[cpart]);
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
   HYPRE_Int                ndim     = hypre_SStructMatrixNDim(P);
   hypre_SStructGraph      *graph    = hypre_SStructMatrixGraph(P);
   hypre_SStructGrid       *dom_grid = hypre_SStructGraphDomGrid(graph);
   hypre_SStructGrid       *ran_grid = hypre_SStructGraphRanGrid(graph);
   HYPRE_Int               *cids     = hypre_SStructGridPartIDs(dom_grid);
   HYPRE_Int               *fids     = hypre_SStructGridPartIDs(ran_grid);
   HYPRE_Int                cnparts  = hypre_SStructGridNParts(dom_grid);
   HYPRE_Int                fnparts  = hypre_SStructGridNParts(ran_grid);

   hypre_SStructPMatrix    *A_p, *P_p;
   hypre_StructMatrix      *A_s, *P_s;
   hypre_SStructPGrid      *pgrid;
   hypre_BoxArray          *compute_boxes;
   hypre_BoxArray          *grid_boxes;
   hypre_BoxArray          *pbnd_boxa;
   hypre_Box               *compute_box;
   hypre_Box               *A_dbox;
   hypre_Box               *P_dbox;

   hypre_StructStencil     *A_stencil;
   hypre_Index             *A_stencil_shape;
   HYPRE_Int                A_stencil_size;
   hypre_StructStencil     *P_stencil;
   hypre_Index             *P_stencil_shape;
   HYPRE_Int               *ventries, nventries;

   HYPRE_Real              *Ap, *Pp0, *Pp1, *Pp2, *Pcoef;
   HYPRE_Real               Pconst[3], center;

   HYPRE_Int                Astenc, Pstenc1, Pstenc2;
   hypre_Index              Astart, Astride, Pstart, Pstride;
   hypre_Index              origin, stride, loop_size, index;

   HYPRE_Int                cdir;
   HYPRE_Int                part_id, fpart, cpart;
   HYPRE_Int                i, j, si, vi, Ai, Pi;
   HYPRE_Int                nvars;

   /*-------------------------------------------------------
    * Set prolongation coefficients for each part
    *-------------------------------------------------------*/
   for (cpart = 0; cpart < cnparts; cpart++)
   {
      part_id = cids[cpart];
      fpart   = hypre_BinarySearch(fids, part_id, fnparts);
      cdir    = cdir_p[part_id];

      pgrid = hypre_SStructGridPGrid(ran_grid, fpart);
      A_p   = hypre_SStructMatrixPMatrix(A, fpart);
      P_p   = hypre_SStructMatrixPMatrix(P, cpart);
      nvars = hypre_SStructPMatrixNVars(P_p);

      for (vi = 0; vi < nvars; vi++)
      {
         A_s = hypre_SStructPMatrixSMatrix(A_p, vi, vi);
         P_s = hypre_SStructPMatrixSMatrix(P_p, vi, vi);

         pbnd_boxa       = hypre_SStructPGridPBndBoxArray(pgrid, vi);

         A_stencil       = hypre_StructMatrixStencil(A_s);
         A_stencil_shape = hypre_StructStencilShape(A_stencil);
         A_stencil_size  = hypre_StructStencilSize(A_stencil);

         P_stencil       = hypre_StructMatrixStencil(P_s);
         P_stencil_shape = hypre_StructStencilShape(P_stencil);

         /* Set center coefficient to 1 */
         Pp0 = hypre_StructMatrixConstData(P_s, 0);
         Pp0[0] = 1;

         if (hypre_StructMatrixConstEntry(P_s, 1))
         {
            /* Off-diagonal entries are constant */
            Pp1 = hypre_StructMatrixConstData(P_s, 1);
            Pp2 = hypre_StructMatrixConstData(P_s, 2);

            Pp1[0] = 0.5;
            Pp2[0] = 0.5;
         }
         else
         {
            /* Off-diagonal entries are variable */
            compute_box = hypre_BoxCreate(ndim);

            Pstenc1 = hypre_IndexD(P_stencil_shape[1], cdir);
            Pstenc2 = hypre_IndexD(P_stencil_shape[2], cdir);

            /* Compute the constant part of the stencil collapse */
            ventries = hypre_TAlloc(HYPRE_Int, A_stencil_size);
            nventries = 0;
            Pconst[0] = 0.0;
            Pconst[1] = 0.0;
            Pconst[2] = 0.0;
            for (si = 0; si < A_stencil_size; si++)
            {
               if (hypre_StructMatrixConstEntry(A_s, si))
               {
                  Ap = hypre_StructMatrixConstData(A_s, si);
                  Astenc = hypre_IndexD(A_stencil_shape[si], cdir);

                  if (Astenc == 0)
                  {
                     Pconst[0] += Ap[0];
                  }
                  else if (Astenc == Pstenc1)
                  {
                     Pconst[1] -= Ap[0];
                  }
                  else if (Astenc == Pstenc2)
                  {
                     Pconst[2] -= Ap[0];
                  }
               }
               else
               {
                  ventries[nventries++] = si;
               }
            }

            /* Get stencil space on base grid for entry 1 of P (valid also for entry 2) */
            hypre_StructMatrixGetStencilSpace(P_s, 1, 0, origin, stride);

            hypre_CopyToIndex(stride, ndim, Astride);
            hypre_StructMatrixMapDataStride(A_s, Astride);
            hypre_CopyToIndex(stride, ndim, Pstride);
            hypre_StructMatrixMapDataStride(P_s, Pstride);

            grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A_s));
            compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P_s));
            hypre_ForBoxI(i, compute_boxes)
            {
               hypre_CopyBox(hypre_BoxArrayBox(compute_boxes, i), compute_box);
               hypre_ProjectBox(compute_box, origin, stride);
               hypre_CopyToIndex(hypre_BoxIMin(compute_box), ndim, Astart);
               hypre_StructMatrixMapDataIndex(A_s, Astart);
               hypre_CopyToIndex(hypre_BoxIMin(compute_box), ndim, Pstart);
               hypre_StructMatrixMapDataIndex(P_s, Pstart);

               A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_s), i);
               P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P_s), i);

               Pp1 = hypre_StructMatrixBoxData(P_s, i, 1);
               Pp2 = hypre_StructMatrixBoxData(P_s, i, 2);

               hypre_BoxGetStrideSize(compute_box, stride, loop_size);

               hypre_BoxLoop2Begin(ndim, loop_size,
                                   A_dbox, Astart, Astride, Ai,
                                   P_dbox, Pstart, Pstride, Pi);
#ifdef HYPRE_USING_OPENMP
#praga omp parallel for private(HYPRE_BOX_PRIVATE,Ai,Pi,si,center,Ap,Astenc) HYPRE_SMP_SCHEDULE
#endif
               hypre_BoxLoop2For(Ai, Pi)
               {
                  center  = Pconst[0];
                  Pp1[Pi] = Pconst[1];
                  Pp2[Pi] = Pconst[2];
                  for (vi = 0; vi < nventries; vi++)
                  {
                     si = ventries[vi];
                     Ap = hypre_StructMatrixBoxData(A_s, i, si);
                     Astenc = hypre_IndexD(A_stencil_shape[si], cdir);

                     if (Astenc == 0)
                     {
                        center += Ap[Ai];
                     }
                     else if (Astenc == Pstenc1)
                     {
                        Pp1[Pi] -= Ap[Ai];
                     }
                     else if (Astenc == Pstenc2)
                     {
                        Pp2[Pi] -= Ap[Ai];
                     }

                  }

                  if (center)
                  {
                     Pp1[Pi] /= center;
                     Pp2[Pi] /= center;
                  }
                  else
                  {
                     /* For some reason the interpolation coefficients sum to zero */
                     Pp1[Pi] = 0.0;
                     Pp2[Pi] = 0.0;
                  }
               }
               hypre_BoxLoop2End(Ai, Pi);
            } /* loop on compute_boxes */

            /* Adjust weights at part boundaries */
            hypre_ForBoxI(i, pbnd_boxa)
            {
               hypre_CopyBox(hypre_BoxArrayBox(pbnd_boxa, i), compute_box);
               hypre_ProjectBox(compute_box, origin, stride);
               hypre_CopyToIndex(hypre_BoxIMin(compute_box), ndim, Pstart);
               hypre_StructMatrixMapDataIndex(P_s, Pstart);

               /* TODO: Fix 0 */
               Pp1 = hypre_StructMatrixBoxData(P_s, 0, 1);
               Pp2 = hypre_StructMatrixBoxData(P_s, 0, 2);

               hypre_BoxGetStrideSize(compute_box, stride, loop_size);

               hypre_BoxLoop1Begin(ndim, loop_size, P_dbox, Pstart, Pstride, Pi);
               hypre_BoxLoop1For(Pi)
               {
                  center = Pp1[Pi] + Pp2[Pi];
                  if (center)
                  {
                     Pp1[Pi] /= center;
                     Pp2[Pi] /= center;
                  }
               }
               hypre_BoxLoop1End(Pi);
            } /* loop on pbnd_boxa */

            hypre_TFree(ventries);
            hypre_BoxDestroy(compute_box);
         } /* if constant variables*/

         hypre_StructMatrixAssemble(P_s);
         /* The following call is needed to prevent cases where interpolation reaches
          * outside the boundary with nonzero coefficient */
         hypre_StructMatrixClearBoundary(P_s);
      }
   }

   return hypre_error_flag;
}
