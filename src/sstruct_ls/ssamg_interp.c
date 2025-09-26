/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 * TODO:
 *       1) Consider inter-variable coupling
 *       2) Do we need to Destroy SStructStencil?
 *--------------------------------------------------------------------------*/

hypre_SStructMatrix *
hypre_SSAMGCreateInterpOp( hypre_SStructMatrix  *A,
                           hypre_SStructGrid    *dom_grid,
                           HYPRE_Int            *cdir_p)
{
   MPI_Comm                 comm   = hypre_SStructMatrixComm(A);
   HYPRE_Int                ndim   = hypre_SStructMatrixNDim(A);
   hypre_SStructGraph      *graph  = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid       *grid   = hypre_SStructGraphGrid(graph);
   HYPRE_Int                nparts = hypre_SStructGridNParts(grid);

   hypre_SStructMatrix     *P;
   hypre_SStructPMatrix    *pmatrix;
   hypre_StructMatrix      *smatrix;
   hypre_SStructGraph      *graph_P;
   hypre_SStructStencil  ***stencils_P;
   hypre_StructStencil     *stencil;

   HYPRE_Int                centries[3] = {0, 1, 2};
   HYPRE_Int               *num_centries;
   HYPRE_Int                stencil_size_A;
   HYPRE_Int                stencil_size_P;
   hypre_Index            **st_shape;
   hypre_Index             *strides;

   HYPRE_Int                cdir;
   HYPRE_Int                part, nvars;
   HYPRE_Int                i, s, vi;

   /*-------------------------------------------------------
    * Allocate data
    *-------------------------------------------------------*/
   st_shape     = hypre_CTAlloc(hypre_Index *, nparts, HYPRE_MEMORY_HOST);
   strides      = hypre_CTAlloc(hypre_Index, nparts, HYPRE_MEMORY_HOST);
   num_centries = hypre_CTAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);

   /*-------------------------------------------------------
    * Create SStructGraph data structure for P
    *-------------------------------------------------------*/
   HYPRE_SStructGraphCreate(comm, grid, &graph_P);

   /* Set domain grid. This is needed in order to SStructUMatrixSetBoxValues be
      able to compute the correct column indices in case of rectangular matrices */
   HYPRE_SStructGraphSetDomainGrid(graph_P, dom_grid);

   /* Set stencil entries */
   stencils_P = hypre_SStructGraphStencils(graph_P);
   for (part = 0; part < nparts; part++)
   {
      cdir    = cdir_p[part];
      pmatrix = hypre_SStructMatrixPMatrix(A, part);
      nvars   = hypre_SStructPMatrixNVars(pmatrix);

      hypre_SetIndex(strides[part], 1);
      if (cdir > -1)
      {
         /* Coarsening in direction cdir by a factor of 2 */
         hypre_IndexD(strides[part], cdir) = 2;

         stencil_size_P     = 3;
         st_shape[part]     = hypre_CTAlloc(hypre_Index, stencil_size_P, HYPRE_MEMORY_HOST);
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
         /* This part is not coarsened */
         hypre_IndexD(strides[part], 0) = 2;
         stencil_size_P     = 1;
         st_shape[part]     = hypre_CTAlloc(hypre_Index, 1, HYPRE_MEMORY_HOST);
         num_centries[part] = stencil_size_P;
         for (vi = 0; vi < nvars; vi++)
         {
            /* Set up the stencil for P(part, vi, vi) */
            hypre_SetIndex(st_shape[part][0], 0);

            /* Create stencil for P(part, vi) */
            HYPRE_SStructStencilCreate(ndim, stencil_size_P, &stencils_P[part][vi]);
            HYPRE_SStructStencilSetEntry(stencils_P[part][vi], 0, st_shape[part][0], vi);
         }
      }
   }

   HYPRE_SStructGraphAssemble(graph_P);
   HYPRE_SStructMatrixCreate(comm, graph_P, &P);
   for (part = 0; part < nparts; part++)
   {
      HYPRE_SStructMatrixSetDomainStride(P, part, strides[part]);
      HYPRE_SStructMatrixSetConstantEntries(P, part, -1, -1, num_centries[part], centries);
   }

   /* Free memory */
   HYPRE_SStructGraphDestroy(graph_P);
   hypre_TFree(strides, HYPRE_MEMORY_HOST);
   hypre_TFree(num_centries, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      hypre_TFree(st_shape[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(st_shape, HYPRE_MEMORY_HOST);

   return P;
}

/*--------------------------------------------------------------------------
 * Sets up structured interpolation coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetupSInterpOp( hypre_SStructMatrix  *A,
                           HYPRE_Int            *cdir_p,
                           hypre_SStructMatrix  *P,
                           HYPRE_Int             interp_type)
{
   HYPRE_Int                ndim       = hypre_SStructMatrixNDim(P);
   hypre_SStructGraph      *graph      = hypre_SStructMatrixGraph(P);
   hypre_SStructGrid       *grid       = hypre_SStructGraphGrid(graph);
   HYPRE_Int                nparts     = hypre_SStructGridNParts(grid);

   hypre_SStructPMatrix    *A_p, *P_p;
   hypre_StructMatrix      *A_s, *P_s;
   hypre_SStructPGrid      *pgrid;
   hypre_StructGrid        *sgrid;
   hypre_BoxArray          *compute_boxes;
   hypre_BoxArray          *pbnd_boxa;
   hypre_BoxArrayArray     *pbnd_boxaa;
   hypre_Box               *compute_box;
   hypre_Box               *tmp_box;
   hypre_Box               *P_dbox;

   hypre_StructStencil     *P_stencil;
   HYPRE_Real              *Pp1, *Pp2;
   HYPRE_Real               Pconst0, Pconst1, Pconst2;

   hypre_Index              Pstart, Pstride;
   hypre_Index              origin, stride, loop_size;
   hypre_Box               *shrink_box;

   HYPRE_Int                cdir;
   HYPRE_Int                part, nvars;
   HYPRE_Int                box_id;
   HYPRE_Int                d, i, ii, j, vi;
   HYPRE_Real               one  = 1.0;
   HYPRE_Real               half = 0.5;

   HYPRE_MemoryLocation     memory_location_P = hypre_SStructMatrixMemoryLocation(P);

   /*-------------------------------------------------------
    * Create temporary boxes
    *-------------------------------------------------------*/

   tmp_box     = hypre_BoxCreate(ndim);
   shrink_box  = hypre_BoxCreate(ndim);
   compute_box = hypre_BoxCreate(ndim);

   /*-------------------------------------------------------
    * Setup structured prolongation component
    *-------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      cdir  = cdir_p[part];
      A_p   = hypre_SStructMatrixPMatrix(A, part);
      P_p   = hypre_SStructMatrixPMatrix(P, part);
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPMatrixNVars(P_p);

      for (vi = 0; vi < nvars; vi++)
      {
         pbnd_boxaa = hypre_SStructPGridPBndBoxArrayArray(pgrid, vi);
         A_s        = hypre_SStructPMatrixSMatrix(A_p, vi, vi);
         P_s        = hypre_SStructPMatrixSMatrix(P_p, vi, vi);
         sgrid      = hypre_StructMatrixGrid(P_s);
         P_stencil  = hypre_StructMatrixStencil(P_s);

         /* Set center coefficient to 1 */
         hypre_TMemcpy(hypre_StructMatrixConstData(P_s, 0), &one, HYPRE_Real, 1,
                       memory_location_P, HYPRE_MEMORY_HOST);

         /* If there are no off-diagonal entries, assemble the matrix and continue outer loop */
         if (hypre_StructStencilSize(P_stencil) <= 1)
         {
            /* Assemble structured component of prolongation matrix */
            hypre_StructMatrixAssemble(P_s);

            /* The following call is needed to prevent cases where interpolation reaches
            * outside the boundary with nonzero coefficient */
            hypre_StructMatrixClearBoundary(P_s);

            continue;
         }

         if (hypre_StructMatrixConstEntry(P_s, 1))
         {
            /* Off-diagonal entries are constant */
            hypre_TMemcpy(hypre_StructMatrixConstData(P_s, 1), &half, HYPRE_Real, 1,
                          memory_location_P, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(hypre_StructMatrixConstData(P_s, 2), &half, HYPRE_Real, 1,
                          memory_location_P, HYPRE_MEMORY_HOST);
         }
         else
         {
            /* Set prolongation entries derived from constant coefficients in A */
            hypre_PFMGSetupInterpOp_core_CC(P_s, A_s, cdir, &Pconst0, &Pconst1, &Pconst2);

            /* Set prolongation entries derived from variable coefficients in A */
            hypre_PFMGSetupInterpOp_core_VC(P_s, A_s, cdir, Pconst0, Pconst1, Pconst2);

            /* Get the stencil space on the base grid for entry 1 of P (valid also for entry 2) */
            hypre_StructMatrixGetStencilSpace(P_s, 1, 0, origin, stride);

            /* Get grid boxes for P_s */
            compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P_s));

            /* Adjust weights at part boundaries */
            hypre_ForBoxArrayI(i, pbnd_boxaa)
            {
               pbnd_boxa = hypre_BoxArrayArrayBoxArray(pbnd_boxaa, i);
               box_id = hypre_BoxArrayArrayID(pbnd_boxaa, i);
               ii = hypre_BinarySearch(hypre_StructGridIDs(sgrid),
                                       box_id,
                                       hypre_StructGridNumBoxes(sgrid));

               if (ii > -1)
               {
                  hypre_ForBoxI(j, pbnd_boxa)
                  {
                     hypre_CopyBox(hypre_BoxArrayBox(pbnd_boxa, j), compute_box);
                     hypre_ProjectBox(compute_box, origin, stride);
                     hypre_CopyToIndex(hypre_BoxIMin(compute_box), ndim, Pstart);
                     hypre_StructMatrixMapDataIndex(P_s, Pstart);
                     hypre_CopyToIndex(stride, ndim, Pstride);
                     hypre_StructMatrixMapDataStride(P_s, Pstride);

                     Pp1 = hypre_StructMatrixBoxData(P_s, ii, 1);
                     Pp2 = hypre_StructMatrixBoxData(P_s, ii, 2);
                     P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P_s), ii);

                     /* Update compute_box */
                     for (d = 0; d < ndim; d++)
                     {
                        if ((d != cdir) && (hypre_BoxSizeD(compute_box, d) > 1))
                        {
                           hypre_IndexD(hypre_BoxIMin(compute_box), d) -= 1;
                           hypre_IndexD(hypre_BoxIMax(compute_box), d) += 1;
                        }
                     }
                     hypre_CopyBox(compute_box, tmp_box);
                     hypre_IntersectBoxes(tmp_box,
                                          hypre_BoxArrayBox(compute_boxes, ii),
                                          compute_box);
                     hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                     /* Only do the renormalization at boundaries if interp_type < 0
                      * (that is, if no unstructured interpolation is used) */
                     if (interp_type < 0)
                     {
                        if (hypre_IndexD(loop_size, cdir) > 1)
                        {
                           hypre_BoxLoop1Begin(ndim, loop_size, P_dbox, Pstart, Pstride, Pi);
                           {
                              HYPRE_Real center = Pp1[Pi] + Pp2[Pi];

                              if (center)
                              {
                                 Pp1[Pi] /= center;
                                 Pp2[Pi] /= center;
                              }
                           }
                           hypre_BoxLoop1End(Pi);
                        }
                        else
                        {
                           hypre_BoxLoop1Begin(ndim, loop_size, P_dbox, Pstart, Pstride, Pi);
                           {
                              if (Pp1[Pi] > Pp2[Pi])
                              {
                                 Pp1[Pi] = 1.0;
                                 Pp2[Pi] = 0.0;
                              }
                              else if (Pp2[Pi] > Pp1[Pi])
                              {
                                 Pp1[Pi] = 0.0;
                                 Pp2[Pi] = 1.0;
                              }
                           }
                           hypre_BoxLoop1End(Pi);
                        } /* if loop_size[cdir] > 1 */
                     } /* if interp_type < 0 */
                  } /* loop on pbnd_boxa */
               } /* if ii > -1 */
            } /* loop on pbnd_boxaa */
         } /* if constant coefficients */

         /* Assemble structured component of prolongation matrix */
         hypre_StructMatrixAssemble(P_s);

         /* The following call is needed to prevent cases where interpolation reaches
          * outside the boundary with nonzero coefficient */
         hypre_StructMatrixClearBoundary(P_s);
      }
   }

   /* Free memory */
   hypre_BoxDestroy(tmp_box);
   hypre_BoxDestroy(shrink_box);
   hypre_BoxDestroy(compute_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Sets up interpolation coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetupInterpOp( hypre_SStructMatrix  *A,
                          HYPRE_Int            *cdir_p,
                          hypre_SStructMatrix  *P,
                          HYPRE_Int             interp_type)
{
   /* Setup structured interpolation component */
   hypre_SSAMGSetupSInterpOp(A, cdir_p, P, interp_type);

   /* Setup unstructured interpolation component */
   hypre_SSAMGSetupUInterpOp(A, cdir_p, P, interp_type);

   return hypre_error_flag;
}
