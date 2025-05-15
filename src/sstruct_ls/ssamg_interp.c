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
 * hypre_SSAMGSetupInterpOp
 *
 * Sets up interpolation coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetupInterpOp( hypre_SStructMatrix  *A,
                          HYPRE_Int            *cdir_p,
                          hypre_SStructMatrix  *P,
                          HYPRE_Int             interp_type)
{
   HYPRE_Int                ndim       = hypre_SStructMatrixNDim(P);
   hypre_SStructGraph      *graph      = hypre_SStructMatrixGraph(P);
   hypre_SStructGrid       *grid       = hypre_SStructGraphGrid(graph);
   HYPRE_Int                nparts     = hypre_SStructGridNParts(grid);

   hypre_ParCSRMatrix      *A_u        = hypre_SStructMatrixParCSRMatrix(A);
   hypre_CSRMatrix         *A_ud       = hypre_ParCSRMatrixDiag(A_u);
   hypre_CSRMatrix         *A_uo       = hypre_ParCSRMatrixOffd(A_u);
   hypre_ParCSRMatrix      *A_aug;
   hypre_ParCSRMatrix      *P_u;
   hypre_CSRMatrix         *P_ud;
   hypre_CSRMatrix         *P_uo;

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
   HYPRE_Complex           *Pp1, *Pp2;
   HYPRE_Complex            Pconst0, Pconst1, Pconst2;

   hypre_Index              Pstart, Pstride;
   hypre_Index              origin, stride, loop_size;
   hypre_Box               *shrink_box;
   hypre_Index              grow_index;
   hypre_IndexRef           shrink_start;

   HYPRE_Int               *num_ghost;
   HYPRE_Int                cdir;
   HYPRE_Int                part, nvars;
   HYPRE_Int                box_id;
   HYPRE_Int                d, i, ii, j, vi;
   HYPRE_Int                vol, offset;
   HYPRE_Int                box_start_index;

   HYPRE_Int                debug_flag = 0;
   HYPRE_Real               trunc_factor = 0.0;
   HYPRE_Int                max_elmts = 4;

   void                    *A_obj;
   hypre_IJMatrix          *A_struct_bndry_ij = NULL;
   hypre_ParCSRMatrix      *A_struct_bndry;
   hypre_ParCSRMatrix      *A_bndry;
   hypre_ParCSRMatrix      *zero;
   hypre_CSRMatrix         *zero_diag;
   hypre_CSRMatrix         *delete_zeros;
   HYPRE_Int                num_indices;
   HYPRE_Int               *indices[HYPRE_MAXDIM];
   hypre_BoxArray          *indices_boxa = NULL;
   hypre_Index              start;
   HYPRE_Real               threshold;
   hypre_BoxArray        ***convert_boxa;
   HYPRE_Int               *CF_marker;
   HYPRE_Complex            one  = 1.0;
   HYPRE_Complex            half = 0.5;

   HYPRE_MemoryLocation     memory_location_P = hypre_SStructMatrixMemoryLocation(P);

#if defined(HYPRE_USING_GPU)
   HYPRE_Int                *all_indices[HYPRE_MAXDIM];
   HYPRE_Int                *box_nnzrows;
   HYPRE_Int                *box_nnzrows_end;
   HYPRE_Int                 max_num_rownnz;
   HYPRE_Int                *nonzero_rows;
   HYPRE_Int                *nonzero_rows_end;
#endif

   /*-------------------------------------------------------
    * Create temporary boxes
    *-------------------------------------------------------*/

   tmp_box     = hypre_BoxCreate(ndim);
   shrink_box  = hypre_BoxCreate(ndim);
   compute_box = hypre_BoxCreate(ndim);

   /*-------------------------------------------------------
    * Set prolongation coefficients for each part
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
         hypre_TMemcpy(hypre_StructMatrixConstData(P_s, 0), &one, HYPRE_Complex, 1,
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
            hypre_TMemcpy(hypre_StructMatrixConstData(P_s, 1), &half, HYPRE_Complex, 1,
                          memory_location_P, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(hypre_StructMatrixConstData(P_s, 2), &half, HYPRE_Complex, 1,
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
                              HYPRE_Complex center = Pp1[Pi] + Pp2[Pi];

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

   /* Set up unstructured interpolation component */
   if (interp_type >= 0)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      /* Get nonzero rows of A_u (that is, all rows with nonzeros in diag or offd) */
      if (!hypre_CSRMatrixRownnz(A_ud))
      {
         hypre_CSRMatrixSetRownnz(A_ud);
      }
      if (!hypre_CSRMatrixRownnz(A_uo))
      {
         hypre_CSRMatrixSetRownnz(A_uo);
      }
      max_num_rownnz = hypre_CSRMatrixNumRownnz(A_ud) + hypre_CSRMatrixNumRownnz(A_uo);
      nonzero_rows   = hypre_TAlloc(HYPRE_Int, max_num_rownnz, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL(merge,
                        hypre_CSRMatrixRownnz(A_ud),
                        hypre_CSRMatrixRownnz(A_ud) + hypre_CSRMatrixNumRownnz(A_ud),
                        hypre_CSRMatrixRownnz(A_uo),
                        hypre_CSRMatrixRownnz(A_uo) + hypre_CSRMatrixNumRownnz(A_uo),
                        nonzero_rows);
      nonzero_rows_end = HYPRE_THRUST_CALL(unique,
                                           nonzero_rows,
                                           nonzero_rows + max_num_rownnz);
#endif

      /* Convert boundary of A to IJ matrix */
      offset       = 0;
      threshold    = 0.8; // WM: todo - what should this be?
      convert_boxa = hypre_TAlloc(hypre_BoxArray**, nparts, HYPRE_MEMORY_HOST);
      for (part = 0; part < nparts; part++)
      {
         A_p   = hypre_SStructMatrixPMatrix(A, part);
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);

         convert_boxa[part] = hypre_CTAlloc(hypre_BoxArray*, nvars, HYPRE_MEMORY_HOST);

         /* Loop over variables */
         for (vi = 0; vi < nvars; vi++)
         {
            A_s = hypre_SStructPMatrixSMatrix(A_p, vi, vi);
            sgrid = hypre_StructMatrixGrid(A_s);

            /* WM: todo - using the DataSpace yields a box that contains one layer
               of ghost zones on the fine grid but NOT on the coarse grid...
               need to do grid box and then add the ghosts... why? */
            /* compute_boxes = hypre_StructMatrixDataSpace(A_s); */
            compute_boxes = hypre_StructGridBoxes(sgrid);
            convert_boxa[part][vi] = hypre_BoxArrayCreate(0, ndim);

            /* Loop over boxes */
            hypre_ForBoxI(i, compute_boxes)
            {
               hypre_CopyBox(hypre_BoxArrayBox(compute_boxes, i), compute_box);
               num_ghost = hypre_StructGridNumGhost(sgrid);
               hypre_BoxGrowByArray(compute_box, num_ghost);
               hypre_BoxGetSize(compute_box, loop_size);
               vol = hypre_BoxVolume(compute_box);
               hypre_SetIndex(stride, 1);
               hypre_CopyToIndex(hypre_BoxIMin(compute_box), ndim, start);

#if defined(HYPRE_USING_GPU)
               /* Get ALL the indices */
               for (j = 0; j < ndim; j++)
               {
                  all_indices[j] = hypre_CTAlloc(HYPRE_Int, vol, HYPRE_MEMORY_DEVICE);
               }

               hypre_BoxLoop1Begin(ndim, loop_size, compute_box, start, stride, ii);
               {
                  hypre_Index index;
                  hypre_BoxLoopGetIndex(index);
                  if (ndim > 0)
                  {
                     all_indices[0][ii] = index[0] + start[0];
                  }
                  if (ndim > 1)
                  {
                     all_indices[1][ii] = index[1] + start[1];
                  }
                  if (ndim > 2)
                  {
                     all_indices[2][ii] = index[2] + start[2];
                  }
               }
               hypre_BoxLoop1End(ii);

#if defined(HYPRE_USING_SYCL)
               /* WM: todo - sycl */
#else
               /* Get the nonzero rows for this box */
               box_nnzrows     = hypre_TAlloc(HYPRE_Int, vol,
                                              HYPRE_MEMORY_DEVICE);
               box_nnzrows_end = HYPRE_THRUST_CALL(copy_if,
                                                   nonzero_rows,
                                                   nonzero_rows_end,
                                                   box_nnzrows,
                                                   in_range<HYPRE_Int>(offset, offset + vol));
               HYPRE_THRUST_CALL(transform,
                                 box_nnzrows,
                                 box_nnzrows_end,
                                 thrust::make_constant_iterator(offset),
                                 box_nnzrows,
                                 thrust::minus<HYPRE_Int>());
               num_indices = box_nnzrows_end - box_nnzrows;

               for (j = 0; j < ndim; j++)
               {
                  indices[j] = hypre_CTAlloc(HYPRE_Int, num_indices,
                                             HYPRE_MEMORY_DEVICE);
               }

               /* Gather indices at non-zero rows of A_u */
               for (j = 0; j < ndim; j++)
               {
                  HYPRE_THRUST_CALL(gather,
                                    box_nnzrows,
                                    box_nnzrows_end,
                                    all_indices[j],
                                    indices[j]);
               }

               /* Free memory */
               for (j = 0; j < ndim; j++)
               {
                  hypre_TFree(all_indices[j], HYPRE_MEMORY_DEVICE);
               }
               hypre_TFree(box_nnzrows, HYPRE_MEMORY_DEVICE);

#endif // defined(HYPRE_USING_SYCL)

#else // defined(HYPRE_USING_GPU)

               num_indices = 0;
               for (j = 0; j < ndim; j++)
               {
                  indices[j] = hypre_CTAlloc(HYPRE_Int, vol, HYPRE_MEMORY_DEVICE);
               }

               hypre_BoxLoop1ReductionBegin(ndim, loop_size, compute_box, start, stride,
                                            ii, num_indices);
               {
                  if (hypre_CSRMatrixI(A_ud)[offset + ii + 1] -
                      hypre_CSRMatrixI(A_ud)[offset + ii] +
                      hypre_CSRMatrixI(A_uo)[offset + ii + 1] -
                      hypre_CSRMatrixI(A_uo)[offset + ii] > 0)
                  {
                     hypre_Index index;
                     hypre_BoxLoopGetIndex(index);
                     for (j = 0; j < ndim; j++)
                     {
                        indices[j][num_indices] = index[j] + start[j];
                     }
                     num_indices++;
                  }
               }
               hypre_BoxLoop1ReductionEnd(ii, num_indices);

#endif // defined(HYPRE_USING_GPU)
               /* WM: todo - these offsets for the unstructured indices only
                  work with no inter-variable couplings? */
               offset += vol;

               /* Create box array from indices marking where A_u is non-trivial */
               if (num_indices)
               {
                  hypre_BoxArrayCreateFromIndices(ndim, num_indices, indices,
                                                  threshold, &indices_boxa);
                  hypre_ForBoxI(j, indices_boxa)
                  {
                     hypre_CopyBox(hypre_BoxArrayBox(indices_boxa, j), tmp_box);

                     /* WM: todo - need 2 for distance 2 interp below?
                        Make this dependent on interpolation or just hardcode to 2? */
                     hypre_BoxGrowByValue(tmp_box, 2);

                     /* WM: intersect with the struct grid box... is that right?
                        NOTE: if you change to DataSpace of the matrix above, you'll
                        need to change this line */
                     hypre_IntersectBoxes(tmp_box,
                                          hypre_BoxArrayBox(compute_boxes, i),
                                          tmp_box);
                     hypre_AppendBox(tmp_box, convert_boxa[part][vi]);
                  }
                  hypre_BoxArrayDestroy(indices_boxa);
                  indices_boxa = NULL;
               }

               /* Free memory */
               for (j = 0; j < ndim; j++)
               {
                  hypre_TFree(indices[j], HYPRE_MEMORY_DEVICE);
               }
            }
         }
      }
      hypre_SStructMatrixBoxesToUMatrix(A, grid, &A_struct_bndry_ij, convert_boxa);
      for (part = 0; part < nparts; part++)
      {
         for (vi = 0; vi < nvars; vi++)
         {
            hypre_BoxArrayDestroy(convert_boxa[part][vi]);
         }
         hypre_TFree(convert_boxa[part], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(convert_boxa, HYPRE_MEMORY_HOST);

      /* Add structured boundary portion to unstructured portion */
      HYPRE_IJMatrixGetObject(A_struct_bndry_ij, &A_obj);
      A_struct_bndry = (hypre_ParCSRMatrix *) A_obj;
      hypre_ParCSRMatrixAdd(1.0, A_struct_bndry, 1.0, A_u, &A_bndry);

      /* WM: todo - I'm adding a zero diagonal here because if BoomerAMG
         interpolation gets a totally empty matrix (no nonzeros and a NULL data array)
         you run into seg faults... this is kind of a dirty fix for now */
      zero = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A_u),
                                      hypre_ParCSRMatrixGlobalNumRows(A_u),
                                      hypre_ParCSRMatrixGlobalNumCols(A_u),
                                      hypre_ParCSRMatrixRowStarts(A_u),
                                      hypre_ParCSRMatrixRowStarts(A_u),
                                      0,
                                      hypre_ParCSRMatrixNumRows(A_u),
                                      0);
      hypre_ParCSRMatrixInitialize(zero);
      zero_diag = hypre_ParCSRMatrixDiag(zero);
      for (i = 0; i < hypre_CSRMatrixNumRows(zero_diag); i++)
      {
         hypre_CSRMatrixI(zero_diag)[i] = i;
         hypre_CSRMatrixJ(zero_diag)[i] = i;
         hypre_CSRMatrixData(zero_diag)[i] = 0.0;
      }
      hypre_CSRMatrixI(zero_diag)[ hypre_CSRMatrixNumRows(zero_diag) ] =
         hypre_CSRMatrixNumRows(zero_diag);
      hypre_ParCSRMatrixAdd(1.0, zero, 1.0, A_bndry, &A_aug);

      /* Free memory */
      hypre_ParCSRMatrixDestroy(zero);
      hypre_ParCSRMatrixDestroy(A_bndry);

      /* Get CF splitting */
      CF_marker = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A_u),
                                HYPRE_MEMORY_DEVICE);

      /* Initialize CF_marker to all C-point (F-points marked below) */
      for (i = 0; i < hypre_ParCSRMatrixNumRows(A_u); i++)
      {
         CF_marker[i] = 1;
      }

      /* Loop over parts */
      /* WM: todo - re-work CF splitting stuff below...
         I don't think what I have is general. */
      box_start_index = 0;
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);
         A_p   = hypre_SStructMatrixPMatrix(A, part);

         /* WM: todo - cdir can be -1, indicating no coarsening...
            need to account for this case */
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
               /* compute_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A_s), i); */
               hypre_CopyBox(hypre_BoxArrayBox(compute_boxes, i), compute_box);
               hypre_CopyBox(hypre_BoxArrayBox(compute_boxes, i), shrink_box);

               /* Grow the compute box to include ghosts */
               /* WM: todo - use the below instead? I guess sometimes the number of ghosts is not 1 in all directions? */
               /* HYPRE_Int *num_ghost = hypre_StructGridNumGhost(sgrid); */
               /* hypre_BoxGrowByArray(compute_box, num_ghost); */

               hypre_SetIndex(grow_index, 1);
               hypre_BoxGrowByIndex(compute_box, grow_index);

               /* Don't add ghosts to the shrink box in the coarseining direction */
               /* WM: is this right??? What if num_ghosts is not 1??? */
               /* num_ghost[2 * cdir] -= 1; */
               /* num_ghost[2 * cdir + 1] -= 1; */
               /* hypre_BoxGrowByArray(shrink_box, num_ghost); */

               hypre_IndexD(grow_index, cdir) = 0;
               hypre_BoxGrowByIndex(shrink_box, grow_index);
               shrink_start = hypre_BoxIMin(shrink_box);

               /* WM: define the start by even/odd coordinate... is this right? */
               if (hypre_IndexD(shrink_start, cdir) % 2 == 0)
               {
                  hypre_IndexD(shrink_start, cdir)++;
               }

               /* Set the stride to 2 in the coarsening direction (1 otherwise) */
               hypre_SetIndex(stride, 1);
               hypre_IndexD(stride, cdir) = 2;

               /* Get the loop size */
               /* WM: todo - what if there are multiple boxes per part???
                  Does this approach still work? */
               hypre_BoxGetStrideSize(shrink_box, stride, loop_size);

               /* Loop over dofs */
               hypre_BoxLoop1Begin(ndim, loop_size, compute_box,
                                   shrink_start, stride, ii);
               {
                  CF_marker[box_start_index + ii] = -1;
               }
               hypre_BoxLoop1End(ii);

               /* Increment box start index */
               box_start_index += hypre_BoxVolume(compute_box);
            }
         }
      }

      /* Generate unstructured interpolation */
      /* WM: todo - experiment with strenght matrix that counts only the P_s stencil
         entries and all inter-part connections as strong; this keeps the same
         sparsity pattern inside the structured part
         WM: todo - add other interpolation options (align interp_type parameter
         with BoomerAMG numbering) */
      hypre_BoomerAMGBuildInterp(A_aug,
                                 CF_marker,
                                 A_aug, /* WM: todo - do I need to do any strength measure here? */
                                 hypre_ParCSRMatrixColStarts(hypre_SStructMatrixParCSRMatrix(P)),
                                 1,
                                 NULL,
                                 debug_flag,
                                 trunc_factor,
                                 max_elmts,
                                 &P_u);

      /* Set P_u as unstructured component of P */
      hypre_IJMatrixDestroyParCSR(hypre_SStructMatrixIJMatrix(P));
      hypre_IJMatrixSetObject(hypre_SStructMatrixIJMatrix(P), P_u);
      hypre_SStructMatrixParCSRMatrix(P) = P_u;
      hypre_IJMatrixAssembleFlag(hypre_SStructMatrixIJMatrix(P)) = 1;

      /* Zero out C-point injection entries and entries in P_u outside
         of non-zero rows of A_u and delete zeros */
      P_ud = hypre_ParCSRMatrixDiag(P_u);
      P_uo = hypre_ParCSRMatrixOffd(P_u);
      for (i = 0; i < hypre_CSRMatrixNumRows(A_ud); i++)
      {
         /* If this is a C-point or a zero row in A_u, zero out P_u */
         if (CF_marker[i] == 1 ||
             (hypre_CSRMatrixI(A_ud)[i + 1] - hypre_CSRMatrixI(A_ud)[i] +
              hypre_CSRMatrixI(A_uo)[i + 1] - hypre_CSRMatrixI(A_uo)[i]) == 0)
         {
            for (j = hypre_CSRMatrixI(P_ud)[i]; j < hypre_CSRMatrixI(P_ud)[i + 1]; j++)
            {
               hypre_CSRMatrixData(P_ud)[j] = 0.0;
            }
            for (j = hypre_CSRMatrixI(P_uo)[i]; j < hypre_CSRMatrixI(P_uo)[i + 1]; j++)
            {
               hypre_CSRMatrixData(P_uo)[j] = 0.0;
            }
         }
      }

      delete_zeros = hypre_CSRMatrixDeleteZeros(P_ud, HYPRE_REAL_MIN);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(P_ud);
         P_ud = hypre_ParCSRMatrixDiag(P_u) = delete_zeros;
         hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixDiag(P_u));
      }

      delete_zeros = hypre_CSRMatrixDeleteZeros(P_uo, HYPRE_REAL_MIN);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(P_uo);
         P_uo = hypre_ParCSRMatrixOffd(P_u) = delete_zeros;
         hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixOffd(P_u));
      }

      /* Overwrite entries in P_s where appropriate with values of P_u */
      hypre_SStructMatrixCompressUToS(P, 0);

      /* Remove zeros from P_u again after the compression above */
      /* WM: todo - Currently I have to get rid of zeros twice...
         is there a better way? */
      delete_zeros = hypre_CSRMatrixDeleteZeros(P_ud, HYPRE_REAL_MIN);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(P_ud);
         P_ud = hypre_ParCSRMatrixDiag(P_u) = delete_zeros;
         hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixDiag(P_u));
      }

      delete_zeros = hypre_CSRMatrixDeleteZeros(P_uo, HYPRE_REAL_MIN);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(P_uo);
         P_uo = hypre_ParCSRMatrixOffd(P_u) = delete_zeros;
         hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixOffd(P_u));
      }
      hypre_ParCSRMatrixSetNumNonzeros(P_u);

      /* Clean up */
      HYPRE_IJMatrixDestroy(A_struct_bndry_ij);
      hypre_ParCSRMatrixDestroy(A_aug);
      hypre_TFree(CF_marker, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_GPU)
      hypre_TFree(nonzero_rows, HYPRE_MEMORY_DEVICE);
#endif
   }

   /* Free memory */
   hypre_BoxDestroy(tmp_box);
   hypre_BoxDestroy(shrink_box);
   hypre_BoxDestroy(compute_box);

   return hypre_error_flag;
}
