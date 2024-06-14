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
   HYPRE_SStructMatrixInitialize(P);
   HYPRE_SStructMatrixAssemble(P);

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
 * hypre_SSAMGSetupInterpOp
 *
 * Sets up interpolation coefficients
 *
 * TODO: Add DEVICE_VAR to boxloops
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetupInterpOp( hypre_SStructMatrix  *A,
                          HYPRE_Int            *cdir_p,
                          hypre_SStructMatrix  *P,
                          HYPRE_Int            interp_type)
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
   hypre_Box               *A_dbox;
   hypre_Box               *P_dbox;

   hypre_StructStencil     *A_stencil;
   hypre_Index             *A_stencil_shape;
   HYPRE_Int                A_stencil_size;
   hypre_StructStencil     *P_stencil;
   hypre_Index             *P_stencil_shape;
   HYPRE_Int                P_stencil_size;
   HYPRE_Int               *ventries, nventries;

   HYPRE_Real              *Ap, *Pp0, *Pp1, *Pp2;
   HYPRE_Real               Pconst[3];

   HYPRE_Int                Astenc, Pstenc1, Pstenc2;
   hypre_Index              Astart, Astride, Pstart, Pstride;
   hypre_Index              origin, stride, loop_size;

   HYPRE_Int                cdir;
   HYPRE_Int                part, nvars;
   HYPRE_Int                box_id;
   HYPRE_Int                d, i, ii, j, si, vi;

   /*-------------------------------------------------------
    * Set prolongation coefficients for each part
    *-------------------------------------------------------*/
   tmp_box = hypre_BoxCreate(ndim);
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

         A_s = hypre_SStructPMatrixSMatrix(A_p, vi, vi);
         A_stencil       = hypre_StructMatrixStencil(A_s);
         A_stencil_shape = hypre_StructStencilShape(A_stencil);
         A_stencil_size  = hypre_StructStencilSize(A_stencil);

         P_s = hypre_SStructPMatrixSMatrix(P_p, vi, vi);
         sgrid = hypre_StructMatrixGrid(P_s);
         P_stencil       = hypre_StructMatrixStencil(P_s);
         P_stencil_shape = hypre_StructStencilShape(P_stencil);
         P_stencil_size  = hypre_StructStencilSize(P_stencil);

         /* Set center coefficient to 1 */
         Pp0 = hypre_StructMatrixConstData(P_s, 0);
         Pp0[0] = 1;

         if (P_stencil_size > 1)
         {
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
               ventries = hypre_TAlloc(HYPRE_Int, A_stencil_size, HYPRE_MEMORY_HOST);
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

               compute_boxes = hypre_StructGridBoxes(sgrid);
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
                  {
                     HYPRE_Int    ei, entry;
                     HYPRE_Int    Astenc;
                     HYPRE_Real   center;
                     HYPRE_Real  *Ap;

                     center  = Pconst[0];
                     Pp1[Pi] = Pconst[1];
                     Pp2[Pi] = Pconst[2];
                     for (ei = 0; ei < nventries; ei++)
                     {
                        entry = ventries[ei];
                        Ap = hypre_StructMatrixBoxData(A_s, i, entry);
                        Astenc = hypre_IndexD(A_stencil_shape[entry], cdir);

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
                        hypre_IntersectBoxes(tmp_box, hypre_BoxArrayBox(compute_boxes, ii), compute_box);

                        hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                        /* Only do the renormalization at boundaries if interp_type < 0
                         * (that is, if no unstructured interpolation is used) */
                        if (interp_type >= 0)
                        {
                           /* If handling all interpolation at part boundaries with unstructured, zero out the structured weights */
                           /* WM: todo - this zeros out the boundary weights, but we should not rely on a halo here...
                            *            Move this later, and loop over the same boxes that get converted for A_u_aug */
                           hypre_BoxLoop1Begin(ndim, loop_size, P_dbox, Pstart, Pstride, Pi);
                           {
                              Pp1[Pi] = 0.0;
                              Pp2[Pi] = 0.0;
                           }
                           hypre_BoxLoop1End(Pi);
                        }
                        else
                        {

                           if (hypre_IndexD(loop_size, cdir) > 1)
                           {
                              hypre_BoxLoop1Begin(ndim, loop_size, P_dbox, Pstart, Pstride, Pi);
                              {
                                 HYPRE_Real center;

                                 center = Pp1[Pi] + Pp2[Pi];
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
                           }
                        }
                     } /* loop on pbnd_boxa */
                  }
               } /* loop on pbnd_boxaa */

               hypre_TFree(ventries, HYPRE_MEMORY_HOST);
               hypre_BoxDestroy(compute_box);
            } /* if constant coefficients */
         } /* if (P_stencil_size > 1)*/

         hypre_StructMatrixAssemble(P_s);
         /* The following call is needed to prevent cases where interpolation reaches
          * outside the boundary with nonzero coefficient */
         hypre_StructMatrixClearBoundary(P_s);

      }
   }

   hypre_BoxDestroy(tmp_box);

   /* Unstructured interpolation */
   if (interp_type >= 0)
   {
      hypre_ParCSRMatrix *A_u = hypre_SStructMatrixParCSRMatrix(A);
      hypre_ParCSRMatrix *A_u_aug;
      hypre_ParCSRMatrix *P_u;

      /* Convert boundary of A to IJ matrix */
      /* WM: todo - don't rely on a halo here but rather connections in A_u */
      hypre_IJMatrix *A_struct_bndry_ij;
      hypre_SStructMatrixHaloToUMatrix(A, grid, &A_struct_bndry_ij, 1);
      hypre_ParCSRMatrix *A_struct_bndry = hypre_IJMatrixObject(A_struct_bndry_ij);

      /* Add structured boundary portion to unstructured portion */
      hypre_ParCSRMatrix *A_bndry;
      hypre_ParCSRMatrixAdd(1.0, A_struct_bndry, 1.0, A_u, &A_bndry);

      /* WM: todo - I'm adding a zero diagonal here because if BoomerAMG interpolation gets a */
      /* totally empty matrix (no nonzeros and a NULL data array) you run into seg faults... this is kind of a dirty fix for now */
      hypre_ParCSRMatrix *zero = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A_u),
                                                            hypre_ParCSRMatrixGlobalNumRows(A_u),
                                                            hypre_ParCSRMatrixGlobalNumCols(A_u),
                                                            hypre_ParCSRMatrixRowStarts(A_u),
                                                            hypre_ParCSRMatrixRowStarts(A_u),
                                                            0,
                                                            hypre_ParCSRMatrixNumRows(A_u),
                                                            0);
      hypre_ParCSRMatrixInitialize(zero);
      hypre_CSRMatrix *zero_diag = hypre_ParCSRMatrixDiag(zero);
      for (i = 0; i < hypre_CSRMatrixNumRows(zero_diag); i++)
      {
         hypre_CSRMatrixI(zero_diag)[i] = i;
         hypre_CSRMatrixJ(zero_diag)[i] = i;
         hypre_CSRMatrixData(zero_diag)[i] = 0.0;
      }
      hypre_CSRMatrixI(zero_diag)[ hypre_CSRMatrixNumRows(zero_diag) ] = hypre_CSRMatrixNumRows(zero_diag);
      hypre_ParCSRMatrixAdd(1.0, zero, 1.0, A_bndry, &A_u_aug);
      hypre_ParCSRMatrixDestroy(zero);

      /* Get CF splitting */
      HYPRE_Int *CF_marker = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A_u), HYPRE_MEMORY_DEVICE);

      /* Initialize CF_marker to all C-point (F-points marked below) */
      for (i = 0; i < hypre_ParCSRMatrixNumRows(A_u); i++)
      {
         CF_marker[i] = 1;
      }

      /* Loop over parts */
      /* WM: todo - re-work CF splitting stuff below... I don't think what I have is general. */
      HYPRE_Int box_start_index = 0;
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);

         A_p   = hypre_SStructMatrixPMatrix(A, part);
         cdir  = cdir_p[part];

         /* Loop over variables */
         for (vi = 0; vi < nvars; vi++)
         {
            A_s = hypre_SStructPMatrixSMatrix(A_p, vi, vi);
            sgrid = hypre_StructMatrixGrid(A_s);
            compute_boxes = hypre_StructGridBoxes(sgrid);
            /* compute_boxes = hypre_StructMatrixDataSpace(A_s); */

            /* Loop over boxes */
            hypre_ForBoxI(i, compute_boxes)
            {
               /* WM: how to set loop_size, box, start, stride?
                *     I guess cdir will be used to set the stride? */
               /* WM: do I need to worry about ghost zones here or anything? */
               compute_box = hypre_BoxClone(hypre_BoxArrayBox(compute_boxes, i));
               /* compute_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_s), i); */
               hypre_Box *shrink_box = hypre_BoxClone(compute_box);

               /* Grow the compute box to include ghosts */
               /* WM: todo - use the below instead? I guess sometimes the number of ghosts is not 1 in all directions? */
               /* HYPRE_Int *num_ghost = hypre_StructGridNumGhost(sgrid); */
               /* hypre_BoxGrowByArray(compute_box, num_ghost); */

               hypre_Index grow_index;
               hypre_IndexX(grow_index) = 1;
               hypre_IndexY(grow_index) = 1;
               hypre_IndexZ(grow_index) = 1;
               hypre_BoxGrowByIndex(compute_box, grow_index);

               /* Don't add ghosts to the shrink box in the coarseining direction */
               /* WM: is this right??? What if num_ghosts is not 1??? */
               /* num_ghost[2 * cdir] -= 1; */
               /* num_ghost[2 * cdir + 1] -= 1; */
               /* hypre_BoxGrowByArray(shrink_box, num_ghost); */

               hypre_IndexD(grow_index, cdir) = 0;
               hypre_BoxGrowByIndex(shrink_box, grow_index);
               hypre_IndexRef shrink_start = hypre_BoxIMin(shrink_box);

               /* WM: define the start by even/odd coordinate... is this right? */
               if (hypre_IndexD(shrink_start, cdir) % 2 == 0)
               {
                  hypre_IndexD(shrink_start, cdir)++;
               }

               /* Set the stride to 2 in the coarsening direction (1 otherwise) */
               hypre_IndexX(stride) = 1;
               hypre_IndexY(stride) = 1;
               hypre_IndexZ(stride) = 1;
               hypre_IndexD(stride, cdir) = 2;

               /* Get the loop size */
               /* WM: todo - what if there are multiple boxes per part??? Does this approach still work? */
               hypre_BoxGetStrideSize(shrink_box, stride, loop_size);

               /* Loop over dofs */
               hypre_BoxLoop1Begin(ndim, loop_size, compute_box, shrink_start, stride, ii);
               {
                  CF_marker[box_start_index + ii] = -1;
               }
               hypre_BoxLoop1End(ii);

               /* Increment box start index */
               box_start_index += hypre_BoxVolume(compute_box);
            }
         }
      }

      /* WM: todo - this is kind of a hacky workaround for now... just looping through */
      /* WM: todo - test without this now that I have compressSToU */
      /* A_u_aug and looking for missing/negative diagonals and removing those rows */
      HYPRE_Int *A_u_aug_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A_u_aug));
      HYPRE_Int *A_u_aug_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A_u_aug));
      HYPRE_Int *A_u_aug_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A_u_aug));
      HYPRE_Complex *A_u_aug_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A_u_aug));
      HYPRE_Complex *A_u_aug_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A_u_aug));
      HYPRE_Int zero_this_row;

      for (i = 0; i < hypre_ParCSRMatrixNumRows(A_u_aug); i++)
      {
         /* Zero out the row unless you find a positive diagonal element */
         zero_this_row = 1;
         for (j = A_u_aug_diag_i[i]; j < A_u_aug_diag_i[i+1]; j++)
         {
            if (A_u_aug_diag_j[j] == i && A_u_aug_diag_data[j] > 0.0)
            {
               zero_this_row = 0;
            }
         }
         if (zero_this_row)
         {
            for (j = A_u_aug_diag_i[i]; j < A_u_aug_diag_i[i+1]; j++)
            {
               A_u_aug_diag_data[j] = 0.0;
            }
            for (j = A_u_aug_offd_i[i]; j < A_u_aug_offd_i[i+1]; j++)
            {
               A_u_aug_offd_data[j] = 0.0;
            }
         }
      }

      /* Generate unstructured interpolation */
      /* WM: todo - experiment with strenght matrix that counts only the P_s stencil entries and all inter-part connections as strong; */
      /*            this keeps the same sparsity pattern inside the structured part */
      /* WM: todo - add other interpolation options (align interp_type parameter with BoomerAMG numbering) */
      HYPRE_Int debug_flag = 0;
      HYPRE_Real trunc_factor = 0.0;
      HYPRE_Int max_elmts = 4;
      hypre_BoomerAMGBuildInterp(A_u_aug,
                                 CF_marker,
                                 A_u_aug, /* WM: todo - do I need to do any strength measure here? */
                                 hypre_ParCSRMatrixColStarts(hypre_SStructMatrixParCSRMatrix(P)),
                                 1,
                                 NULL,
                                 debug_flag,
                                 trunc_factor,
                                 max_elmts,
                                 &P_u);

      /* WM: postprocess P_u to remove injection entries. These should already be accounted for in P_s. Is this the best way to do this? */
      /* WM: todo - yeah, once I have the functionality for compressing unstructured -> structured entries, I can use that here */
      for (i = 0; i < hypre_ParCSRMatrixNumRows(P_u); i++)
      {
         if (CF_marker[i] == 1)
         {
            hypre_CSRMatrixData( hypre_ParCSRMatrixDiag(P_u) )[ hypre_CSRMatrixI( hypre_ParCSRMatrixDiag(P_u) )[i] ] = 0.0;
         }
      }

      /* WM: todo - should I do this here? What tolerance? Smarter way to avoid a bunch of zero entries? */
      hypre_CSRMatrix *delete_zeros = hypre_CSRMatrixDeleteZeros(hypre_ParCSRMatrixDiag(P_u), 1e-20);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(P_u));
         hypre_ParCSRMatrixDiag(P_u) = delete_zeros;
      }
      delete_zeros = hypre_CSRMatrixDeleteZeros(hypre_ParCSRMatrixOffd(P_u), 1e-20);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(P_u));
         hypre_ParCSRMatrixOffd(P_u) = delete_zeros;
      }
      hypre_ParCSRMatrixSetNumNonzeros(P_u);

      /* WM: question - is this the right way to set the U matrix? */
      hypre_IJMatrixDestroyParCSR(hypre_SStructMatrixIJMatrix(P));
      hypre_SStructMatrixParCSRMatrix(P) = P_u;
      hypre_IJMatrixSetObject(hypre_SStructMatrixIJMatrix(P), P_u);

      /* Clean up */
      HYPRE_IJMatrixDestroy(A_struct_bndry_ij);
      hypre_ParCSRMatrixDestroy(A_u_aug);
   }

   return hypre_error_flag;
}
