/*BHEADER**********************************************************************
 * Copyright (c) 2014,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Structured matrix-matrix multiply routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

#ifdef HYPRE_MAXTERMS
#undef HYPRE_MAXTERMS
#endif
#define HYPRE_MAXTERMS 3

/*--------------------------------------------------------------------------
 * Multiply matrices.  The matrix product has 'nterms' terms constructed from
 * the matrices in the 'matrices' array.  Each term t is given by the matrix
 * matrices[terms[t]] transposed according to the boolean transposes[t].
 *
 * This routine uses the StMatrix routines to determine if the operation is
 * allowable and to compute the stencil and stencil formulas for C.
 *
 * All of the matrices must be defined on a common fine index space, and each
 * matrix must have a unitary stride for either its domain or range (or both).
 * RDF: Need to remove the latter requirement.  Think of P*C for example, where
 * P is interpolation and C is a square matrix on the coarse grid.  Another
 * approach (maybe the most flexible) is to temporarily modify the matrices in
 * this routine so that they have a common fine index space.  This will require
 * mapping the matrix strides, the grid extents, and the stencil offsets.
 *
 * RDF TODO: Rewrite communication_info to use CommStencil idea (write routine
 * FromCommStencil and have FromStencil call it).
 *
 * RDF TODO: Compute symmetric matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmult( HYPRE_Int            nmatrices,
                     hypre_StructMatrix **matrices,
                     HYPRE_Int            nterms,
                     HYPRE_Int           *terms,
                     HYPRE_Int           *transposes,
                     hypre_StructMatrix **C_ptr )
{
   hypre_StructMatrix  *C;

   hypre_StMatrix     **st_matrices, *st_C;
   HYPRE_Int            ndim;

   HYPRE_Complex      **ap, *aconst, aprod;
   HYPRE_Int            i, m, t, e, ai, ci, fi, ii[4];
   HYPRE_Int            entry, *shift;

   /* RDF TODO: Maybe write StMatrixCreateFromStructMatrix() and
    * StructMatrixCreateFromStMatrix() routines? */

   /* Create st_matrices from matrices */
   st_matrices = hypre_TAlloc(hypre_StMatrix *, nmatrices);
   for (m = 0; m < nmatrices; m++)
   {
      matrix = matrices[m];
      stencil = hypre_StructMatrixStencil(matrix);
      size = hypre_StructStencilSize(stencil);
      hypre_StMatrixCreate(m, size, ndim, &st_matrix);
      hypre_CopyToIndex(hypre_StructMatrixRStride(matrix), ndim,
                        hypre_StMatrixRMap(st_matrix));
      hypre_CopyToIndex(hypre_StructMatrixDStride(matrix), ndim,
                        hypre_StMatrixDMap(st_matrix));
      for (e = 0; e < size; e++)
      {
         hypre_CopyToIndex(hypre_StructStencilOffset(stencil, e), ndim
                           hypre_StMatrixOffset(st_matrix, e));
         hypre_St_CoeffCreate(1, &st_coeff);
         st_coeff_term = hypre_StCoeffTerm(st_coeff, 0);
         hypre_StTermID(st_coeff_term) = m;
         hypre_StTermEntry(st_coeff_term) = e;
         hypre_StMatrixNCoeff(st_matrix, e) = 1;
         hypre_StMatrixCoeff(st_matrix, e) = st_coeff;
      }
      st_matrices[m] = st_matrix;
   }

   /* Multiply st_matrices */
   st_terms = hypre_CTAlloc(hypre_StMatrix *, nterms);
   for (t = 0; t < nterms; t++)
   {
      st_terms[t] = st_matrices[terms[t]];
   }
   hypre_StMatrixMatmult(nterms, st_terms, transposes, nmatrices, ndim, &st_C);

   /* Determine the base coarsening factor for the range and domain grids */
   rstride = hypre_StMatrixRMap(st_C);
   dstride = hypre_StMatrixDMap(st_C);
   base_stride = rstride;
   for (d = 0; d < ndim; d++)
   {
      if (rstride[d] > dstride[d])
      {
         base_stride = dstride;
         break;
      }
   }
   coarsen = 0;
   for (d = 0; d < ndim; d++)
   {
      if (base_stride[d] > 1)
      {
         coarsen = 1;
         break;
      }
   }

   /* Create the range and domain grids */
   if (transposes[0])
   {
      grid = hypre_StructMatrixDomainGrid(matrices[terms[0]]);
   }
   else
   {
      grid = hypre_StructMatrixGrid(matrices[terms[0]]);
   }
   if (transposes[nterms-1])
   {
      domain_grid = hypre_StructMatrixGrid(matrices[terms[nterms-1]]);
   }
   else
   {
      domain_grid = hypre_StructMatrixDomainGrid(matrices[terms[nterms-1]]);
   }
   if (grid == domain_grid)
   {
      domain_grid = NULL;
   }
   if (coarsen)
   {
      HYPRE_StructGridCoarsen(grid, base_stride, &grid);
      if (domain_grid != NULL)
      {
         HYPRE_StructGridCoarsen(domain_grid, base_stride, &domaingrid);
      }
   }

   /* Create the stencil and compute initial num_coeffs */
   size = hypre_StMatrixSize(st_C);
   HYPRE_StructStencilCreate(ndim, size, &stencil);
   num_coeffs = 0;
   for (e = 0; e < size; e++)
   {
      hypre_CopyToIndex(hypre_StMatrixOffset(st_C, e), offset);
      if (coarsen)
      {
         hypre_MapToCoarseIndex(offset, NULL, base_stride, ndim);
      }
      HYPRE_StructStencilSetEntry(stencil, e, offset);
      num_coeffs += hypre_StMatrixNCoeff(st_C, e);
   }

   /* Use st_C to compute information needed to build the matrix */
   /* This splits the computation into constant and variable coefficient
    * computations as indicated by num_coeffs and num_const_entries.  Variable
    * computations are further split into constant and variable components, with
    * constant contributions stored in the aconst array.  Communication stencils
    * are also computed for each matrix (not each term, so matrices that appear
    * in more than one term in the product are dealt with only once).
    * Communication stencils are then used to determine ghost sizes for a single
    * fine and coarse data space to simplify the boxloop below. */

   const_entries = hypre_TAlloc(HYPRE_Int, size);
   aconst = hypre_TAlloc(HYPRE_Complex, num_coeffs);

   num_coeffs = 0;
   num_const_entries = 0;
   if (hypre_BoxArraySize(grid_boxes) > 0)
   {
      comm_stencils = hypre_TAlloc(hypre_CommStencil *, nmatrices);
      for (m = 0; m < nmatrices; m++)
      {
         comm_stencils[m] = hypre_CommStencilCreate(ndim);
      }

      i = 0;
      ai = 0;
      /* Loop over each stencil coefficient in st_C */
      for (e = 0; e < size; e++)
      {
         const_entry = 1;
         const_values[num_const_entries] = 0.0;
         coeff = hypre_StMatrixCoeff(st_C, e);
         while (coeff != NULL)
         {
            aconst[i] = 1.0;
            /* Each coefficient in the sum is a product of nterms terms */
            for (t = 0; t < nterms; t++)
            {
               term = hypre_StCoeffTerm(coeff, t);
               aterms[ai] = *term;  /* Copy term info into aterms array */
               term = &aterms[ai];
               m = hypre_StTermID(term);
               entry = hypre_StTermEntry(term);
               shift = hypre_StTermShift(term);

               matrix = matrices[m];
               hypre_CopyToIndex(shift, ndim, csoffset);
               if (hypre_StructMatrixConstEntry(matrix, entry))
               {
                  /* Accumulate the constant contribution to the product */
                  constp = hypre_StructMatrixBoxData(matrix, 0, entry);
                  aconst[i] *= constp[0];
                  if (!transposes[t])
                  {
                     stencil = hypre_StructMatrixStencil(matrix);
                     offsetref = hypre_StructStencilOffset(stencil, entry);
                     hypre_AddIndexes(csoffset, offsetref, ndim, csoffset);
                  }
               }
               else
               {
                  const_entry = 0;
               }
               hypre_CommStencilSetEntry(comm_stencils[m], csoffset);
               ai++;
            }
            term = &aterms[ai];
            hypre_StTermEntry(term) = e;
            const_values[num_const_entries] += aconst[i];
            ai++;
            i++;

            coeff = hypre_StCoeffNext(coeff);
         }

         if (const_entry)
         {
            const_entries[num_const_entries] = e;
            num_const_entries++;
            /* Reset i and ai */
            i = num_coeffs;
            ai = i*(nterms+1);
         }
         else
         {
            num_coeffs = i;
         }
      }
      num_coeffs = i;
   }

   /* Create the matrix */
   HYPRE_StructMatrixCreate(comm, grid, stencil, &C);
   if (domain_grid != NULL)
   {
      HYPRE_StructMatrixSetDomainGrid(C, domain_grid);
   }
   HYPRE_StructMatrixSetRStride(C, rstride);
   HYPRE_StructMatrixSetDStride(C, dstride);
   HYPRE_StructMatrixSetConstantEntries(C, num_const_entries, const_entries);
   /* HYPRE_StructMatrixSetSymmetric(C, sym); */
#if 1 /* This should be set through the matmult interface somehow */
   {
      HYPRE_Int num_ghost[2*HYPRE_MAXDIM];
      for (i = 0; i < 2*HYPRE_MAXDIM; i++)
      {
         num_ghost[i] = 0;
      }
      HYPRE_StructMatrixSetNumGhost(C, num_ghost);
   }
#endif
   HYPRE_StructMatrixInitialize(C);

   /* Destroy the newly created grids (they will still exist in matrix C) */
   if (coarsen)
   {
      HYPRE_StructGridDestroy(grid);
      HYPRE_StructGridDestroy(domain_grid);
   }
   grid = hypre_StructMatrixGrid(C);
   domain_grid = hypre_StructMatrixDomainGrid(C);

   /* Set constant values in C */
   for (i = 0; i < num_const_entries; i++)
   {
      constp = hypre_StructMatrixBoxData(C, 0, const_entries[i]);
      constp[0] = const_values[i];
   }

   /* Set variable values in C */

   /* RDF START - Fix the start, stride, and base_stride stuff.  The grid_start
    * should be the imin of a projection of the grid_box onto an index space
    * with stride = rstride.  Also set ap[] to point to the appropriate mask for
    * constant coefficients. */

   if (num_coeffs > 0)
   { 
      /* RDF: Agglomerate comm pkgs and do only one communication.  Test the
       * separate case first to make debugging easier.  Note that comm_info is
       * the same for matrices and masks.  Still need to project/coarsen/map. */

      /* Add ghost layers to the matrices and update them */
      /* Create masks for matrices with constant coefficients to prevent
       * incorrect contributions in the matrix product.  Each mask is a vector
       * of all ones with updated ghost layers (this accounts for parallelism
       * and periodic boundary conditions). */
      masks = hypre_CTAlloc(hypre_StructVector, nmatrices);
      for (m = 0; m < nmatrices; m++)
      {
         HYPRE_Int              num_ghost[2*HYPRE_MAXDIM];
         hypre_CommInfo        *comm_info;
         hypre_CommPkg         *comm_pkg;
         hypre_CommHandle      *comm_handle;
         HYPRE_Complex         *vdata, *data;

         matrix = matrices[m];

         if (hypre_StructMatrixNumValues(matrix) > 0)
         {
            /* Add ghost layers to the matrices (this requires a data copy) */
            hypre_CommStencilGetNumGhost(comm_stencils[m], num_ghost);
            hypre_StructMatrixGrowByNumGhost(matrix, num_ghost);

            /* Update the ghost layers of the matrices */
            hypre_CreateCommInfoFromCStencil(hypre_StructMatrixGrid(matrix),
                                             comm_stencils[m], &comm_info);
            hypre_CommPkgCreate(comm_info,
                                hypre_StructMatrixDataSpace(matrix),
                                hypre_StructMatrixDataSpace(matrix),
                                hypre_StructMatrixNumValues(matrix), NULL, 0,
                                hypre_StructMatrixComm(matrix), &comm_pkg);
            hypre_CommInfoDestroy(comm_info);

            vdata = hypre_StructMatrixVData(matrix);
            hypre_InitializeCommunication(comm_pkg, &vdata, &vdata, 0, 0,
                                          &comm_handle);
            hypre_FinalizeCommunication(comm_handle);
         }

         if (hypre_StructMatrixNumCValues(matrix) > 0)
         {
            HYPRE_StructVectorCreate(comm, hypre_StructMatrixGrid(matrix), &mask);
            /* It's important to use num_ghost from the matrix and not the one
             * created above from comm_stencils, because they may not match. */
            hypre_StructVectorSetNumGhost(mask, hypre_StructMatrixNumGhost(matrix));
            HYPRE_StructVectorInitialize(mask);
            hypre_StructVectorSetConstantValues(mask, 1.0);

            /* Update ghosts */
            hypre_CreateCommInfoFromCStencil(hypre_StructVectorGrid(mask),
                                             comm_stencils[m], &comm_info);
            hypre_CommPkgCreate(comm_info,
                                hypre_StructVectorDataSpace(mask),
                                hypre_StructVectorDataSpace(mask), 1, NULL, 0,
                                hypre_StructVectorComm(mask), &comm_pkg);
            hypre_CommInfoDestroy(comm_info);

            data = hypre_StructVectorData(mask);
            hypre_InitializeCommunication(comm_pkg, &data, &data, 0, 0,
                                          &comm_handle);
            hypre_FinalizeCommunication(comm_handle);

            masks[m] = mask;
         }
      }

      /* Set stride array for BoxLoop */
      for (t = 0; t < nterms; t++)
      {
         /* Use first terms of aterms to get correct matrix ids */
         term = &aterms[t];
         m = hypre_StTermID(term);
         matrix = matrices[m];
         hypre_CopyIndex(base_stride, stride[t]);
         hypre_StructMatrixMapDataStride(matrix, stride[t]);
      }
      hypre_SetIndex(stride[t], 1);

      ap = hypre_TAlloc(HYPRE_Complex *, num_coeffs*(nterms+1));
      grid_boxes = hypre_StructGridBoxes(grid);
      hypre_ForBoxI(b, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, b);

         /* Set dbox and start arrays for BoxLoop */
         grid_start = hypre_BoxIMin(grid_box);
         hypre_CopyIndex(grid_start, base_start);
         hypre_MapToFineIndex(base_start, NULL, base_stride, ndim);
         for (t = 0; t < nterms; t++)
         {
            /* Use first terms of aterms to get correct matrix ids */
            term = &aterms[t];
            m = hypre_StTermID(term);
            matrix = matrices[m];
            dbox[t] = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), b);
            hypre_CopyIndex(base_start, start[t]);
            hypre_StructMatrixMapDataIndex(matrix, start[t]);
         }
         dbox[t] = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(C), b);
         hypre_CopyIndex(grid_start, start[t]);
         hypre_StructMatrixMapDataIndex(C, start[t]);

         ai = 0;
         for (i = 0; i < num_coeffs; i++)
         {
            for (t = 0; t < nterms; t++)
            {
               term = &aterms[ai];
               m = hypre_StTermID(term);
               entry = hypre_StTermEntry(term);

               matrix = matrices[m];
               if (hypre_StructMatrixConstEntry(matrix, entry))
               {
                  /* RDF: Point this at the matrix mask */
                  ap[ai] = NULL;
               }
               else
               {
                  shift = hypre_StTermShift(term);
                  ap[ai] = hypre_StructMatrixBoxData(matrix, b, entry) +
                     hypre_BoxOffsetDistance(dbox[t], shift);
               }
               ai++;
            }
            term = &aterms[ai];
            entry = hypre_StTermEntry(term);
            ap[ai] = hypre_StructMatrixBoxData(C, b, entry);
            ai++;
         }

         hypre_BoxGetSize(grid_box, loop_size);

         hypre_BoxLoopMBegin(nterms+1, ndim, loop_size, dbox, start, stride, ii);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,ii,ai,aprod,m) HYPRE_SMP_SCHEDULE
#endif
         hypre_BoxLoopMFor(ii)
         {
            ai = 0;
            for (i = 0; i < num_coeffs; i++)
            {
               aprod = aconst[i];
               for (t = 0; t < nterms; t++)
               {
                  aprod *= ap[ai][ii[t]];
                  ai++;
               }
               ap[ai][ii[t]] += aprod;
               ai++;
            }
         }
         hypre_BoxLoopMEnd(ii);
      }
   }




   /* Let the user shrink AA and BB (if desired) after this routine returns */
   /* hypre_StructMatrixSetOneGhost(matrix, 0); */
   /* hypre_StructMatrixShrink(matrix); */

   *C_ptr = C;

   return hypre_error_flag;
}
