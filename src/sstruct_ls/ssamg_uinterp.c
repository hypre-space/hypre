/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Disable OpenMP support in this source file due to issues with box loop reduction */
#include "HYPRE_config.h"
#if defined (HYPRE_USING_OPENMP)
#define OMP0
#define OMP1
#endif

#include "_hypre_sstruct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 * Sets up unstructured interpolation coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetupUInterpOp( hypre_SStructMatrix  *A,
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

   hypre_SStructPMatrix    *A_p;
   hypre_StructMatrix      *A_s;
   hypre_SStructPGrid      *pgrid;
   hypre_StructGrid        *sgrid;
   hypre_BoxArray          *compute_boxes;
   hypre_Box               *compute_box;
   hypre_Box               *tmp_box;

   hypre_Index              stride, loop_size;
   hypre_Box               *shrink_box;
   hypre_Index              grow_index;
   hypre_IndexRef           shrink_start;

   HYPRE_Int               *num_ghost;
   HYPRE_Int                cdir;
   HYPRE_Int                part, nvars;
   HYPRE_Int                i, j, vi;
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

#if defined(HYPRE_USING_GPU)
   HYPRE_Int                *all_indices[HYPRE_MAXDIM];
   HYPRE_Int                *box_nnzrows;
   HYPRE_Int                *box_nnzrows_end;
   HYPRE_Int                 max_num_rownnz;
   HYPRE_Int                *nonzero_rows;
   HYPRE_Int                *nonzero_rows_end;

   HYPRE_MemoryLocation      memory_location = hypre_SStructMatrixMemoryLocation(A);
   HYPRE_ExecutionPolicy     exec            = hypre_GetExecPolicy1(memory_location);
#endif

   /*-------------------------------------------------------
    * Create temporary boxes
    *-------------------------------------------------------*/

   tmp_box     = hypre_BoxCreate(ndim);
   shrink_box  = hypre_BoxCreate(ndim);
   compute_box = hypre_BoxCreate(ndim);

   /* Set up unstructured interpolation component */
   if (interp_type >= 0)
   {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (exec == HYPRE_EXEC_DEVICE)
      {
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
         nonzero_rows   = hypre_TAlloc(HYPRE_Int, max_num_rownnz, memory_location);
         HYPRE_THRUST_CALL(merge,
                           hypre_CSRMatrixRownnz(A_ud),
                           hypre_CSRMatrixRownnz(A_ud) + hypre_CSRMatrixNumRownnz(A_ud),
                           hypre_CSRMatrixRownnz(A_uo),
                           hypre_CSRMatrixRownnz(A_uo) + hypre_CSRMatrixNumRownnz(A_uo),
                           nonzero_rows);
         nonzero_rows_end = HYPRE_THRUST_CALL(unique,
                                              nonzero_rows,
                                              nonzero_rows + max_num_rownnz);
      }
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
               if (exec == HYPRE_EXEC_DEVICE)
               {
                  /* Get ALL the indices */
                  for (j = 0; j < ndim; j++)
                  {
                     all_indices[j] = hypre_CTAlloc(HYPRE_Int, vol, memory_location);
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
                                                 memory_location);
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
                                                memory_location);
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
                     hypre_TFree(all_indices[j], memory_location);
                  }
                  hypre_TFree(box_nnzrows, memory_location);

#endif // defined(HYPRE_USING_SYCL)
               }
               else
#endif // defined(HYPRE_USING_GPU)
               {
                  num_indices = 0;
                  for (j = 0; j < ndim; j++)
                  {
                     indices[j] = hypre_CTAlloc(HYPRE_Int, vol, memory_location);
                  }

                  /* TODO: re-enable box loop reduction with OpenMP */
                  hypre_BoxLoop1ReductionBeginHost(ndim, loop_size, compute_box,
                                                   start, stride, ii, num_indices);
                  {
                     if (hypre_CSRMatrixI(A_ud)[offset + ii + 1] -
                         hypre_CSRMatrixI(A_ud)[offset + ii] +
                         hypre_CSRMatrixI(A_uo)[offset + ii + 1] -
                         hypre_CSRMatrixI(A_uo)[offset + ii] > 0)
                     {
                        hypre_Index index;
                        hypre_BoxLoopGetIndexHost(index);
                        for (j = 0; j < ndim; j++)
                        {
                           indices[j][num_indices] = index[j] + start[j];
                        }
                        num_indices++;
                     }
                  }
                  hypre_BoxLoop1ReductionEndHost(ii, num_indices);
               }

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

                     /* Safety check: only append valid boxes */
                     if (hypre_BoxVolume(tmp_box) > 0)
                     {
                        hypre_AppendBox(tmp_box, convert_boxa[part][vi]);
                     }
                  }
                  hypre_BoxArrayDestroy(indices_boxa);
                  indices_boxa = NULL;
               }

               /* Free memory */
               for (j = 0; j < ndim; j++)
               {
                  hypre_TFree(indices[j], memory_location);
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
                                memory_location);

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
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_CSRMatrixMoveDiagFirstDevice(hypre_ParCSRMatrixDiag(A_aug));
      }
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
      }
      hypre_CSRMatrixSetRownnz(P_ud);

      delete_zeros = hypre_CSRMatrixDeleteZeros(P_uo, HYPRE_REAL_MIN);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(P_uo);
         P_uo = hypre_ParCSRMatrixOffd(P_u) = delete_zeros;
      }
      hypre_CSRMatrixSetRownnz(P_uo);

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
         hypre_CSRMatrixSetRownnz(P_ud);
      }

      delete_zeros = hypre_CSRMatrixDeleteZeros(P_uo, HYPRE_REAL_MIN);
      if (delete_zeros)
      {
         hypre_CSRMatrixDestroy(P_uo);
         P_uo = hypre_ParCSRMatrixOffd(P_u) = delete_zeros;
         hypre_CSRMatrixSetRownnz(P_uo);
      }
      hypre_ParCSRMatrixSetNumNonzeros(P_u);

      /* Clean up */
      HYPRE_IJMatrixDestroy(A_struct_bndry_ij);
      hypre_ParCSRMatrixDestroy(A_aug);
      hypre_TFree(CF_marker, memory_location);
#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_TFree(nonzero_rows, memory_location);
      }
#endif
   }

   /* Free memory */
   hypre_BoxDestroy(tmp_box);
   hypre_BoxDestroy(shrink_box);
   hypre_BoxDestroy(compute_box);

   return hypre_error_flag;
}
