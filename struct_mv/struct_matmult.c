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

#ifdef MAXTERMS
#undef MAXTERMS
#endif
#define MAXTERMS 3

/*--------------------------------------------------------------------------
 * Multiply matrices.  The matrix product has 'nterms' terms constructed from
 * the matrices in the 'matrices' array.  Each term t is given by the matrix
 * matrices[terms[t]] transposed according to the boolean transposes[t].
 *
 * This routine uses the StMatrix routines to determine if the operation is
 * allowable and to compute the stencil and stencil formulas for C.
 *
 * All of the matrices must be defined on a common base grid (fine index space),
 * and each matrix must have a unitary stride for either its domain or range (or
 * both).  RDF: Need to remove the latter requirement.  Think of P*C for
 * example, where P is interpolation and C is a square matrix on the coarse
 * grid.  Another approach (maybe the most flexible) is to temporarily modify
 * the matrices in this routine so that they have a common fine index space.
 * This will require mapping the matrix strides, the grid extents, and the
 * stencil offsets.
 *
 * RDF: Provide more info here about the algorithm below
 * - Each coefficient in the sum is a product of nterms terms
 * - Assumes there are at most two grid index spaces in the product
 *
 * RDF TODO: Rewrite communication_info to use CommStencil idea (write routine
 * FromCommStencil and have FromStencil call it).
 *
 * RDF TODO: Compute symmetric matrix.  Make sure to compute comm_pkg correctly
 * using add_ghost or similar idea.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmult( HYPRE_Int            nmatrices,
                     hypre_StructMatrix **matrices,
                     HYPRE_Int            nterms,
                     HYPRE_Int           *terms,
                     HYPRE_Int           *transposes,
                     hypre_StructMatrix **M_ptr )
{
#if 0
   hypre_StructMatrix  *M;            /* matrix product we are computing */

   hypre_StMatrix     **st_matrices, *st_M;
   hypre_StTerm        *term;
   HYPRE_Int            ndim;

   /* product term used to compute the non-constant stencil entries in M */
   struct a_struct
   {
      hypre_StTerm   terms[MAXTERMS]; /* stencil info for each term */
      HYPRE_Int      mentry;          /* stencil entry for M */
      HYPRE_Complex *tptr[MAXTERMS];  /* pointer to matrix data for each term */
      HYPRE_Complex *mptr;            /* pointer to matrix data for M */
      HYPRE_Complex  cprod;           /* product of the constant terms */

   } *a;

   HYPRE_Int         na;              /* number of product terms in 'a' */
   HYPRE_Int         nconst;          /* number of constant entries in M */
   HYPRE_Int         const_entry;     /* boolean for constant M entry */
   HYPRE_Complex    *const_values;    /* values for constant M entries */

   HYPRE_Complex    *constp;          /* pointer to constant data */
   HYPRE_Complex     aprod;
   hypre_Index       csoffset;        /* CommStencil offset */
   HYPRE_Int         i, m, t, e, ci, fi, ii[4];
   HYPRE_Int         entry, *shift;

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
      hypre_CopyToIndex(hypre_StructMatrixRanStride(matrix), ndim,
                        hypre_StMatrixRMap(st_matrix));
      hypre_CopyToIndex(hypre_StructMatrixDomStride(matrix), ndim,
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
   hypre_StMatrixMatmult(nterms, st_terms, transposes, nmatrices, ndim, &st_M);

   /* Determine the base coarsening factor for the grid */
   ran_stride = hypre_StMatrixRMap(st_M);
   dom_stride = hypre_StMatrixDMap(st_M);
   base_stride = ran_stride;
   for (d = 0; d < ndim; d++)
   {
      if (ran_stride[d] > dom_stride[d])
      {
         base_stride = dom_stride;
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

   /* The grid should be the same for all of the multiplied matrices */
   fgrid = hypre_StructMatrixGrid(matrices[terms[0]]);
   if (coarsen) /* RDF: Be careful here about boxnums */
   {
      HYPRE_StructGridCoarsen(fgrid, base_stride, &grid);
   }
   /* RDF: Need to compute boxnums for mapping from the M matrix boxes in grid
    * to the corresponding boxes in fgrid */

#if 0 /* RDF: Don't think this is needed */
   /* Create the range and domain boxnums */
   matrix = matrices[terms[0]];
   if (transposes[0])
   {
      ran_nboxes = hypre_StructMatrixDomNBoxes(matrix);
      ran_boxnums = hypre_StructMatrixDomBoxnums(matrix);
   }
   else
   {
      ran_nboxes = hypre_StructMatrixRanNBoxes(matrix);
      ran_boxnums = hypre_StructMatrixRanBoxnums(matrix);
   }
   matrix = matrices[terms[nterms-1]];
   if (transposes[nterms-1])
   {
      dom_nboxes = hypre_StructMatrixRanNBoxes(matrix);
      dom_boxnums = hypre_StructMatrixRanBoxnums(matrix);
   }
   else
   {
      dom_nboxes = hypre_StructMatrixDomNBoxes(matrix);
      dom_boxnums = hypre_StructMatrixDomBoxnums(matrix);
   }
#endif

   /* Create the stencil and compute initial value for 'na' */
   size = hypre_StMatrixSize(st_M);
   HYPRE_StructStencilCreate(ndim, size, &stencil);
   na = 0;
   for (e = 0; e < size; e++)
   {
      hypre_CopyToIndex(hypre_StMatrixOffset(st_M, e), offset);
      if (coarsen)
      {
         hypre_MapToCoarseIndex(offset, NULL, base_stride, ndim);
      }
      HYPRE_StructStencilSetEntry(stencil, e, offset);
      na += hypre_StMatrixNCoeff(st_M, e);
   }

   /* Use st_M to compute information needed to build the matrix */
   /* This splits the computation into constant and non-constant computations as
    * indicated by na and nconst.  Non-constant computations are stored in a and
    * further split into constant and non-constant components, with constant
    * contributions stored in a[i].cprod.  Communication stencils are also
    * computed for each matrix (not each term, so matrices that appear in more
    * than one term in the product are dealt with only once).  Communication
    * stencils are then used to determine new data spaces for resizing the
    * matrices. Since we assume there are at most two index spaces, only two
    * data spaces are computed, one fine and one coarse.  This simplifies the
    * boxloop below and allow us to use a BoxLoop3. */

   const_entries = hypre_TAlloc(HYPRE_Int, size);
   a = hypre_TAlloc(struct a_struct, na);

   na = 0;
   nconst = 0;
   if (hypre_StructGridNumBoxes(grid) > 0)
   {
      comm_stencils = hypre_TAlloc(hypre_CommStencil *, nmatrices);
      for (m = 0; m < nmatrices; m++)
      {
         comm_stencils[m] = hypre_CommStencilCreate(ndim);
      }

      i = 0;

      for (e = 0; e < size; e++)  /* Loop over each stencil coefficient in st_M */
      {
         const_entry = 1;
         const_values[nconst] = 0.0;
         coeff = hypre_StMatrixCoeff(st_M, e);
         while (coeff != NULL)
         {
            a[i].cprod = 1.0;
            for (t = 0; t < nterms; t++)
            {
               term = hypre_StCoeffTerm(coeff, t);
               a[i].terms[t] = *term;  /* Copy term info into terms */
               term = &(a[i].terms[t]);
               m = hypre_StTermID(term);
               entry = hypre_StTermEntry(term);
               shift = hypre_StTermShift(term);

               matrix = matrices[m];
               hypre_CopyToIndex(shift, ndim, csoffset);
               if (hypre_StructMatrixConstEntry(matrix, entry))
               {
                  /* Accumulate the constant contribution to the product */
                  constp = hypre_StructMatrixBoxData(matrix, 0, entry);
                  a[i].cprod *= constp[0];
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
            }
            a[i].mentry = e;
            const_values[nconst] += a[i].cprod;

            i++;

            coeff = hypre_StCoeffNext(coeff);
         }

         if (const_entry)
         {
            const_entries[nconst] = e;
            nconst++;
            /* Reset i */
            i = na;
         }

         na = i;
      }
   }

   /* Create the matrix */
   HYPRE_StructMatrixCreate(comm, grid, stencil, &M);
   HYPRE_StructMatrixSetRangeStride(M, ran_stride);
   HYPRE_StructMatrixSetDomainStride(M, dom_stride);
   HYPRE_StructMatrixSetConstantEntries(M, nconst, const_entries);
   /* HYPRE_StructMatrixSetSymmetric(M, sym); */
#if 1 /* This should be set through the matmult interface somehow */
   {
      HYPRE_Int num_ghost[2*HYPRE_MAXDIM];
      for (i = 0; i < 2*HYPRE_MAXDIM; i++)
      {
         num_ghost[i] = 0;
      }
      HYPRE_StructMatrixSetNumGhost(M, num_ghost);
   }
#endif
   HYPRE_StructMatrixInitialize(M);

   /* Destroy the newly created grid (it will still exist in matrix M) */
   if (coarsen)
   {
      HYPRE_StructGridDestroy(grid);
   }
   grid = hypre_StructMatrixGrid(M);

   /* Set constant values in M */
   for (i = 0; i < nconst; i++)
   {
      constp = hypre_StructMatrixBoxData(M, 0, const_entries[i]);
      constp[0] = const_values[i];
   }

   /* Set variable values in M */

   /*
    * RDF START - Fix the start, stride, and base_stride stuff.  The grid_start
    * should be the imin of a projection of the grid_box onto an index space
    * with stride = ran_stride.  Also set tptr[] to point to the appropriate
    * mask for constant coefficients.
    *
    * Need a domain mask for each matrix term in the product, but only if the
    * matrix has constant coefficient stencil entries.  A range mask is not
    * needed because either: 1) the matrix term to the immediate left has a
    * domain mask; 2) the matrix to the left is not constant (hence it's self
    * masking); 3) the matrix term is the first, so its range is masked by the
    * computational loop itself.
    *
    */

   if (na > 0)
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
         HYPRE_Int             *num_ghost;
         hypre_CommInfo        *comm_info;
         hypre_CommPkg         *comm_pkg;
         hypre_CommHandle      *comm_handle;
         HYPRE_Complex         *vdata, *data;

         matrix = matrices[m];

         if (hypre_StructMatrixNumValues(matrix) > 0)
         {
            /* Add ghost layers to the matrices (this requires a data copy) */
            hypre_CommStencilGetNumGhost(comm_stencils[m], &num_ghost);
            hypre_StructMatrixGrowByNumGhost(matrix, num_ghost);
            hypre_TFree(num_ghost);

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
         /* Use first product term to get correct matrix ids */
         term = &a[0].terms[t];
         m = hypre_StTermID(term);
         matrix = matrices[m];
         hypre_CopyIndex(base_stride, stride[t]);
         hypre_StructMatrixMapDataStride(matrix, stride[t]);
      }
      hypre_SetIndex(stride[t], 1);

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
            /* Use first product term to get correct matrix ids */
            term = &a[0].terms[t];
            m = hypre_StTermID(term);
            matrix = matrices[m];
            dbox[t] = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(matrix), b);
            hypre_CopyIndex(base_start, start[t]);
            hypre_StructMatrixMapDataIndex(matrix, start[t]);
         }
         dbox[t] = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(M), b);
         hypre_CopyIndex(grid_start, start[t]);
         hypre_StructMatrixMapDataIndex(M, start[t]);

         ai = 0;
         for (i = 0; i < na; i++)
         {
            for (t = 0; t < nterms; t++)
            {
               term = &(a[i].terms[t]);
               m = hypre_StTermID(term);
               entry = hypre_StTermEntry(term);

               matrix = matrices[m];
               if (hypre_StructMatrixConstEntry(matrix, entry))
               {
                  /* RDF: Point this at the matrix mask */
                  a[i].tptr[t] = NULL;
               }
               else
               {
                  shift = hypre_StTermShift(term);
                  a[i].tptr[t] = hypre_StructMatrixBoxData(matrix, b, entry) +
                     hypre_BoxOffsetDistance(dbox[t], shift);
               }
            }
            term = &(a[i].terms[t]);
            entry = hypre_StTermEntry(term);
            a[i].tptr[t] = hypre_StructMatrixBoxData(M, b, entry);
         }

         hypre_BoxGetSize(grid_box, loop_size);

         hypre_BoxLoop3Begin(ndim, loop_size,
                             dbox0, start0, stride0, i0,
                             dbox1, start1, stride1, i1,
                             dbox2, start2, stride2, i2);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,i0,i1,i2,aprod,t) HYPRE_SMP_SCHEDULE
#endif
         hypre_BoxLoop3For(i0,i1,i2)
         {
            for (i = 0; i < na; i++)
            {
               aprod = a[i].cprod;
               for (t = 0; t < nterms1; t++)
               {
                  aprod *= a[i].tptr[t][i1];
               }
               for (; t < nterms; t++)
               {
                  aprod *= a[i].tptr[t][i2];
               }
               a[i].mptr[i0] += aprod;
            }
         }
         hypre_BoxLoop3End(i0,i1,i2);


      }
   }


   /* Let the user shrink AA and BB (if desired) after this routine returns */
   /* hypre_StructMatrixSetOneGhost(matrix, 0); */
   /* hypre_StructMatrixShrink(matrix); */

   *M_ptr = M;

#endif
   return hypre_error_flag;
}

