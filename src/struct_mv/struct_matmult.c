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
 * allowable and to compute the stencil and stencil formulas for M.
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
 *
 * This routine assumes there are only two data-map strides in the product.
 * This means that at least two matrices can always be multiplied together
 * (assuming it is a valid stencil matrix multiply), hence longer products can
 * be broken up into smaller components (the latter is not yet implemented).
 * The fine and coarse data-map strides are denoted by fstride and cstride.
 * Note that both fstride and cstride are given on the same base index space and
 * may be equal.  The range and domain strides for M are denoted by ran_stride
 * and dom_stride and are also given on the base index space.  The grid for M is
 * coarsened by factor coarsen_stride, which is the smaller of ran_stride and
 * dom_stride.  The computation for each stencil coefficient of M happens on the
 * base index space with stride loop_stride, which is the larger of ran_stride
 * and dom_stride.  Since we require that either ran_stride or dom_stride is
 * larger than all other matrix strides in the product (this is how we guarantee
 * that M has only one stencil), and since the data-map stride for a matrix is
 * currently the largest of its two strides, then we have loop_stride = cstride.
 * In general, the data strides for the boxloop below are as follows:
 *
 *   Mdstride = stride 1
 *   cdstride = loop_stride / cstride (= stride 1)
 *   fdstride = loop_stride / fstride
 *   
 * Here are some examples:
 *
 *   fstride = 2, cstride = 6
 *   ran_stride = 6, dom_stride = 6, coarsen_stride = 6, loop_stride = 6
 *   Mdstride = 1, cdstride = 1, fdstride = 3
 *
 *   6     6   6               2 2               2 2     6   <-- domain/range strides
 *   |     |   |               | |               | |     |
 *   |  M  | = |       R       | |       A       | |  P  |
 *   |     |   |               | |               | |     |
 *                               |               | |     |
 *                               |               | |     |
 *                               |               | |     |
 *
 *   fstride = 2, cstride = 6
 *   ran_stride = 2, dom_stride = 6, coarsen_stride = 2, loop_stride = 6
 *   Mdstride = 1, cdstride = 1, fdstride = 3
 *
 *   2     6   2     6 6     6
 *   |     |   |     | |     |
 *   |  M  | = |  A  | |  B  |
 *   |     |   |     | |     |
 *   |     |   |     |
 *   |     |   |     |
 *   |     |   |     |
 *
 *   fstride = 4, cstride = 8
 *   ran_stride = 8, dom_stride = 2, coarsen_stride = 2, loop_stride = 8
 *   Mdstride = 1, cdstride = 1, fdstride = 2
 *
 *   8               2   8       4 4               2
 *   |       M       | = |   A   | |               |
 *                                 |       B       |
 *                                 |               |
 *
 *
 * RDF: Provide more info here about the algorithm below
 * - Each coefficient in the sum is a product of nterms terms
 * - Assumes there are at most two grid index spaces in the product
 *
 * RDF TODO: Compute symmetric matrix.  Make sure to compute comm_pkg correctly
 * using sym_ghost or similar idea.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmult( HYPRE_Int            nmatrices_input,
                     hypre_StructMatrix **matrices_input,
                     HYPRE_Int            nterms,
                     HYPRE_Int           *terms_input,
                     HYPRE_Int           *transposes,
                     hypre_StructMatrix **M_ptr )
{
   MPI_Comm             comm;

   hypre_StructMatrix **matrices;     /* matrices we are multiplying */
   HYPRE_Int            nmatrices, *terms, *matmap;

   hypre_StructMatrix  *M;            /* matrix product we are computing */
   hypre_StructStencil *Mstencil;
   hypre_StructGrid    *Mgrid;
   hypre_Index          Mran_stride, Mdom_stride;

   hypre_StMatrix     **st_matrices, *st_matrix, *st_M;
   hypre_StCoeff       *st_coeff;
   hypre_StTerm        *st_term;
   HYPRE_Int            ndim, size;

   hypre_StructMatrix  *matrix;
   hypre_StructStencil *stencil;
   hypre_StructGrid    *grid;
   HYPRE_Int            nboxes;
   HYPRE_Int           *boxnums;
   hypre_IndexRef       stride;
   hypre_Box           *box;

   hypre_IndexRef       ran_stride, dom_stride, coarsen_stride;
   HYPRE_Int            coarsen;
   HYPRE_Int           *mtypes;        /* data-map types for each matrix (fine or coarse) */

   /* product term used to compute the variable stencil entries in M */
   struct a_struct
   {
      hypre_StTerm    terms[MAXTERMS]; /* stencil info for each term */
      HYPRE_Int       mentry;          /* stencil entry for M */
      HYPRE_Complex   cprod;           /* product of the constant terms */
      HYPRE_Int       types[MAXTERMS]; /* types of computations to do for each term */
      HYPRE_Complex  *tptrs[MAXTERMS]; /* pointers to matrix data for each term */
      HYPRE_Complex  *mptr;            /* pointer to matrix data for M */

   } *a;

   HYPRE_Int            na;              /* number of product terms in 'a' */
   HYPRE_Int            nconst;          /* number of constant entries in M */
   HYPRE_Int            const_entry;     /* boolean for constant entry in M */
   HYPRE_Int           *const_entries;   /* constant entries in M */
   HYPRE_Complex       *const_values;    /* values for constant entries in M */
   hypre_CommStencil  **comm_stencils;

   hypre_StructVector  *mask;
   HYPRE_Int            need_mask;            /* boolean indicating if a bit mask is needed */
   HYPRE_Int            const_term, var_term; /* booleans used to determine 'need_mask' */

   HYPRE_Complex        prod;
   HYPRE_Complex       *constp;          /* pointer to constant data */
   HYPRE_Complex       *bitptr;          /* pointer to bit mask data */
   hypre_Index          offset;          /* CommStencil offset */
   hypre_IndexRef       shift, offsetref;
   HYPRE_Int            d, i, j, m, t, e, b, ci, fi, Mi, Mj, Mb, id, entry, Mentry;

   hypre_Index          Mstart;      /* M's stencil location on the base index space */
   hypre_Box           *loop_box;    /* boxloop extents on the base index space */
   hypre_IndexRef       loop_start;  /* boxloop start index on the base index space */
   hypre_IndexRef       loop_stride; /* boxloop stride on the base index space */
   hypre_Index          loop_size;   /* boxloop size */
   hypre_Index          Mstride;           /* data-map stride  (base index space) */
   hypre_IndexRef       fstride,  cstride; /* data-map strides (base index space) */
   hypre_Index          fdstart,  cdstart,  Mdstart;  /* boxloop data starts */
   hypre_Index          fdstride, cdstride, Mdstride; /* boxloop data strides */
   hypre_Box           *fdbox,   *cdbox,   *Mdbox;    /* boxloop data boxes */
   hypre_Index          tdstart;

   hypre_BoxArray     **data_spaces;
   hypre_BoxArray      *cdata_space, *fdata_space, *Mdata_space, *data_space;

   /* RDF TODO: Maybe write StMatrixCreateFromStructMatrix() and
    * StructMatrixCreateFromStMatrix() routines? */

   /* Create new matrices and terms arrays from the input arguments, because we
    * only want to consider those matrices actually involved in the multiply */
   matmap = hypre_CTAlloc(HYPRE_Int, nmatrices_input);
   for (t = 0; t < nterms; t++)
   {
      m = terms_input[t];
      matmap[m] = 1;
   }
   nmatrices = 0;
   for (m = 0; m < nmatrices_input; m++)
   {
      if (matmap[m])
      {
         matmap[m] = nmatrices;
         nmatrices++;
      }
   }
   matrices   = hypre_CTAlloc(hypre_StructMatrix *, nmatrices);
   terms      = hypre_CTAlloc(HYPRE_Int, nterms);
   for (t = 0; t < nterms; t++)
   {
      m = terms_input[t];
      matrices[matmap[m]] = matrices_input[m];
      terms[t] = matmap[m];
   }
   hypre_TFree(matmap);

   /* Set comm and ndim */
   matrix = matrices[0];
   comm = hypre_StructMatrixComm(matrix);
   ndim = hypre_StructMatrixNDim(matrix);

   /* Create st_matrices from terms and matrices.  This may sometimes create the
    * same StMatrix more than once, but by doing it this way, we can set the ID
    * to be the original term number so that we can tell whether a term in the
    * final product corresponds to a transposed matrix (the StMatrixMatmult
    * routine currently does not guarantee that terms in the final product will
    * be ordered the same as originally). */
   st_matrices = hypre_CTAlloc(hypre_StMatrix *, nterms);
   for (t = 0; t < nterms; t++)
   {
      m = terms[t];
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
         hypre_CopyToIndex(hypre_StructStencilOffset(stencil, e), ndim,
                           hypre_StMatrixOffset(st_matrix, e));
         hypre_StCoeffCreate(1, &st_coeff);
         st_term = hypre_StCoeffTerm(st_coeff, 0);
         hypre_StTermID(st_term) = t;
         hypre_StTermEntry(st_term) = e;
         hypre_StMatrixCoeff(st_matrix, e) = st_coeff;
      }
      st_matrices[t] = st_matrix;
   }

   /* Multiply st_matrices */
   hypre_StMatrixMatmult(nterms, st_matrices, transposes, nterms, ndim, &st_M);

   /* Free up st_matrices */
   for (t = 0; t < nterms; t++)
   {
      hypre_StMatrixDestroy(st_matrices[t]);
   }
   hypre_TFree(st_matrices);

   /* Determine the coarsening factor for M's grid (the stride for either the
    * range or the domain, whichever is smaller) */
   ran_stride = hypre_StMatrixRMap(st_M);
   dom_stride = hypre_StMatrixDMap(st_M);
   coarsen_stride = ran_stride;
   for (d = 0; d < ndim; d++)
   {
      if (ran_stride[d] > dom_stride[d])
      {
         coarsen_stride = dom_stride;
         break;
      }
   }
   coarsen = 0;
   for (d = 0; d < ndim; d++)
   {
      if (coarsen_stride[d] > 1)
      {
         coarsen = 1;
         break;
      }
   }

   /* Create Mgrid (the grid for M) */
   grid = hypre_StructMatrixGrid(matrices[0]); /* Same grid for all matrices */
   hypre_CopyToIndex(ran_stride, ndim, Mran_stride);
   hypre_CopyToIndex(dom_stride, ndim, Mdom_stride);
   if (coarsen)
   {
      /* Note: Mgrid may have fewer boxes than grid as a result of coarsening */
      HYPRE_StructGridCoarsen(grid, coarsen_stride, &Mgrid);
      hypre_MapToCoarseIndex(Mran_stride, NULL, coarsen_stride, ndim);
      hypre_MapToCoarseIndex(Mdom_stride, NULL, coarsen_stride, ndim);
   }
   else
   {
      hypre_StructGridRef(grid, &Mgrid);
   }

   /* Create Mstencil and compute an initial value for 'na' (next section below) */
   size = hypre_StMatrixSize(st_M);
   HYPRE_StructStencilCreate(ndim, size, &Mstencil);
   na = 0;
   for (e = 0; e < size; e++)
   {
      hypre_CopyToIndex(hypre_StMatrixOffset(st_M, e), ndim, offset);
      if (coarsen)
      {
         hypre_MapToCoarseIndex(offset, NULL, coarsen_stride, ndim);
      }
      HYPRE_StructStencilSetEntry(Mstencil, e, offset);
      na += hypre_StMatrixNEntryCoeffs(st_M, e);
   }

   /* Use st_M to compute information needed to build the matrix.
    *
    * This splits the computation into constant and variable computations as
    * indicated by 'na' and 'nconst'.  Variable computations are stored in 'a'
    * and further split into constant and variable subcomponents, with constant
    * contributions stored in 'a[i].cprod'.  Communication stencils are also
    * computed for each matrix (not each term, so matrices that appear in more
    * than one term in the product are dealt with only once).  Communication
    * stencils are then used to determine new data spaces for resizing the
    * matrices.  Since we assume there are at most two data-map strides, only
    * two data spaces are computed, one fine and one coarse.  This simplifies
    * the boxloop below and allow us to use a BoxLoop3.  We add an extra entry
    * to the end of 'comm_stencils' and 'data_spaces' for the bit mask, in case
    * a bit mask is needed. */

   const_entries = hypre_TAlloc(HYPRE_Int, size);
   const_values  = hypre_TAlloc(HYPRE_Complex, size);
   a = hypre_TAlloc(struct a_struct, na);

   na = 0;
   nconst = 0;
   need_mask = 0;
   if (hypre_StructGridNumBoxes(grid) > 0)
   {
      comm_stencils = hypre_TAlloc(hypre_CommStencil *, nmatrices+1);
      for (m = 0; m < nmatrices+1; m++)
      {
         comm_stencils[m] = hypre_CommStencilCreate(ndim);
      }

      i = 0;

      for (e = 0; e < size; e++)  /* Loop over each stencil coefficient in st_M */
      {
         const_entry = 1;
         const_values[nconst] = 0.0;
         st_coeff = hypre_StMatrixCoeff(st_M, e);
         while (st_coeff != NULL)
         {
            a[i].cprod = 1.0;
            const_term = 0;
            var_term = 0;
            for (t = 0; t < nterms; t++)
            {
               st_term = hypre_StCoeffTerm(st_coeff, t);
               a[i].terms[t] = *st_term;  /* Copy st_term info into terms */
               st_term = &(a[i].terms[t]);
               id = hypre_StTermID(st_term);
               entry = hypre_StTermEntry(st_term);
               shift = hypre_StTermShift(st_term);
               m = terms[id];
               matrix = matrices[m];

               hypre_CopyToIndex(shift, ndim, offset);
               if (hypre_StructMatrixConstEntry(matrix, entry))
               {
                  /* Accumulate the constant contribution to the product */
                  constp = hypre_StructMatrixConstData(matrix, entry);
                  a[i].cprod *= constp[0];
                  if (!transposes[id])
                  {
                     stencil = hypre_StructMatrixStencil(matrix);
                     offsetref = hypre_StructStencilOffset(stencil, entry);
                     hypre_AddIndexes(offset, offsetref, ndim, offset);
                  }
                  hypre_CommStencilSetEntry(comm_stencils[nmatrices], offset);
                  const_term = 1;
               }
               else
               {
                  hypre_CommStencilSetEntry(comm_stencils[m], offset);
                  const_entry = 0;
                  var_term = 1;
               }
            }
            /* Add the product terms as long as it looks like the stencil entry
             * for M will be constant */
            if (const_entry)
            {
               const_values[nconst] += a[i].cprod;
            }
            /* Need a bit mask if we have a mixed constant-and-variable product term */
            if (const_term && var_term)
            {
               need_mask = 1;
            }
            a[i].mentry = e;

            i++;

            st_coeff = hypre_StCoeffNext(st_coeff);
         }

         /* Keep track of constant stencil entries and values in M */
         if (const_entry)
         {
            const_entries[nconst] = e;
            nconst++;
            /* Reset i (the temporary counter for na) */
            i = na;
         }

         na = i;
      }
   }

   /* Create the matrix */
   HYPRE_StructMatrixCreate(comm, Mgrid, Mstencil, &M);
   HYPRE_StructMatrixSetRangeStride(M, Mran_stride);
   HYPRE_StructMatrixSetDomainStride(M, Mdom_stride);
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
   *M_ptr = M;

   /* Destroy Mstencil and Mgrid (they will still exist in matrix M) */
   HYPRE_StructStencilDestroy(Mstencil);
   HYPRE_StructGridDestroy(Mgrid);
   Mgrid = hypre_StructMatrixGrid(M);

   /* Set constant values in M */
   for (i = 0; i < nconst; i++)
   {
      constp = hypre_StructMatrixConstData(M, const_entries[i]);
      constp[0] = const_values[i];
   }

   /* Free up some stuff */
   hypre_TFree(const_entries);
   hypre_TFree(const_values);

   /* If all constant coefficients, return */
   if (na == 0)
   {
      /* Free up some stuff */
      hypre_StMatrixDestroy(st_M);
      hypre_TFree(matrices);
      hypre_TFree(terms);
      hypre_TFree(a);
      if (hypre_StructGridNumBoxes(grid) > 0)
      {
         for (m = 0; m < nmatrices+1; m++)
         {
            hypre_CommStencilDestroy(comm_stencils[m]);
         }
         hypre_TFree(comm_stencils);
      }

      HYPRE_StructMatrixAssemble(M);
      return hypre_error_flag;
   }

   /* Set variable values in M */

   /* Create a bit mask with bit data for each matrix term that has constant
    * coefficients to prevent incorrect contributions in the matrix product.
    * The bit mask is a vector with appropriately set bits and updated ghost
    * layer to account for parallelism and periodic boundary conditions. */

   loop_box = hypre_BoxCreate(ndim);

   /* Set Mstride and Mdata_space */
   hypre_StructMatrixGetDataMapStride(M, &stride);
   hypre_CopyToIndex(stride, ndim, Mstride);                    /* M's index space */
   hypre_MapToFineIndex(Mstride, NULL, coarsen_stride, ndim);   /* base index space */
   Mdata_space = hypre_StructMatrixDataSpace(M);

   /* Compute fstride and cstride (assumes only two data-map strides) */
   hypre_StructMatrixGetDataMapStride(matrices[0], &fstride);
   cstride = fstride;
   for (m = 1; m < nmatrices; m++)
   {
      hypre_StructMatrixGetDataMapStride(matrices[m], &stride);
      for (d = 0; d < ndim; d++)
      {
         if (stride[d] > fstride[d])
         {
            cstride = stride;
            break;
         }
         else if (stride[d] < cstride[d])
         {
            fstride = stride;
            break;
         }
      }
   }

   /* Compute mtypes (assumes only two data-map strides) */
   mtypes = hypre_CTAlloc(HYPRE_Int, nmatrices+1); /* initialize to fine data spaces */
   for (m = 0; m < nmatrices; m++)
   {
      hypre_StructMatrixGetDataMapStride(matrices[m], &stride);
      for (d = 0; d < ndim; d++)
      {
         if (stride[d] > fstride[d])
         {
            mtypes[m] = 1; /* coarse data space */
            break;
         }
      }
   }

   /* Compute initial data spaces for each matrix */
   data_spaces = hypre_CTAlloc(hypre_BoxArray *, nmatrices+1);
   for (m = 0; m < nmatrices; m++)
   {
      HYPRE_Int  *num_ghost;

      matrix = matrices[m];

      /* If matrix is all constant, num_ghost should be all zero */
      hypre_CommStencilCreateNumGhost(comm_stencils[m], &num_ghost);
      /* RDF TODO: Make sure num_ghost is at least as large as before, so that
       * when we call Restore() below, we don't lose any data */
      if (hypre_StructMatrixDomainIsCoarse(M))
      {
         /* Increase num_ghost (on both sides) to ensure that data spaces are
          * large enough to compute the full stencil in one boxloop.  This is
          * a result of how stencils are stored when the domain is coarse. */
         for (d = 0; d < ndim; d++)
         {
            num_ghost[2*d]   += dom_stride[d] - 1;
            num_ghost[2*d+1] += dom_stride[d] - 1;
         }
      }
      hypre_StructMatrixComputeDataSpace(matrix, num_ghost, &data_spaces[m]);
      hypre_TFree(num_ghost);
   }

   /* Compute initial bit mask data space */
   if (need_mask)
   {
      HYPRE_Int  *num_ghost;

      HYPRE_StructVectorCreate(comm, grid, &mask);
      HYPRE_StructVectorSetStride(mask, fstride); /* same stride as fine data-map stride */
      hypre_CommStencilCreateNumGhost(comm_stencils[nmatrices], &num_ghost);
      hypre_StructVectorComputeDataSpace(mask, num_ghost, &data_spaces[nmatrices]);
      hypre_TFree(num_ghost);
   }

   /* Compute fine and coarse data spaces */
   fdata_space = NULL;
   cdata_space = NULL;
   for (m = 0; m < nmatrices+1; m++)
   {
      data_space = data_spaces[m];
      if (data_space != NULL) /* This can be NULL when there is no bit mask */
      {
         switch (mtypes[m])
         {
            case 0: /* fine data space */
               if (fdata_space == NULL)
               {
                  fdata_space = data_space;
               }
               else
               {
                  hypre_ForBoxI(b, fdata_space)
                  {
                     hypre_BoxGrowByBox(hypre_BoxArrayBox(fdata_space, b),
                                        hypre_BoxArrayBox(data_space, b));
                  }
                  hypre_BoxArrayDestroy(data_space);
               }
               break;

            case 1: /* coarse data space */
               if (cdata_space == NULL)
               {
                  cdata_space = data_space;
               }
               else
               {
                  hypre_ForBoxI(b, cdata_space)
                  {
                     hypre_BoxGrowByBox(hypre_BoxArrayBox(cdata_space, b),
                                        hypre_BoxArrayBox(data_space, b));
                  }
                  hypre_BoxArrayDestroy(data_space);
               }
               break;
         }
      }
   }

   /* Resize the matrix data spaces */
   for (m = 0; m < nmatrices; m++)
   {
      switch (mtypes[m])
      {
         case 0: /* fine data space */
            data_spaces[m] = hypre_BoxArrayClone(fdata_space);
            break;

         case 1: /* coarse data space */
            data_spaces[m] = hypre_BoxArrayClone(cdata_space);
            break;
      }
      hypre_StructMatrixResize(matrices[m], data_spaces[m]);
   }

   /* Resize the bit mask data space and initialize */
   if (need_mask)
   {
      HYPRE_Int  bitval;

      data_spaces[nmatrices] = hypre_BoxArrayClone(fdata_space);
      hypre_StructVectorResize(mask, data_spaces[nmatrices]);
      hypre_StructVectorInitialize(mask);

      for (t = 0; t < nterms; t++)
      {
         /* Use a[0].terms for the list of matrices and transpose statuses */
         st_term = &(a[0].terms[t]);
         id = hypre_StTermID(st_term);
         m = terms[id];
         matrix = matrices[m];

         if (transposes[id])
         {
            nboxes  = hypre_StructMatrixRanNBoxes(matrix);
            boxnums = hypre_StructMatrixRanBoxnums(matrix);
            stride  = hypre_StructMatrixRanStride(matrix);
         }
         else
         {
            nboxes  = hypre_StructMatrixDomNBoxes(matrix);
            boxnums = hypre_StructMatrixDomBoxnums(matrix);
            stride  = hypre_StructMatrixDomStride(matrix);
         }

         bitval = (1 << t);
         loop_stride = stride;
         hypre_CopyToIndex(loop_stride, ndim, fdstride);
         hypre_StructVectorMapDataStride(mask, fdstride);
         for (j = 0; j < nboxes; j++)
         {
            b = boxnums[j];

            box = hypre_StructGridBox(grid, b);
            hypre_CopyBox(box, loop_box);
            hypre_ProjectBox(loop_box, NULL, loop_stride);
            loop_start = hypre_BoxIMin(loop_box);
            hypre_BoxGetStrideSize(loop_box, loop_stride, loop_size);

            fdbox = hypre_BoxArrayBox(fdata_space, b);
            hypre_CopyToIndex(loop_start, ndim, fdstart);
            hypre_StructVectorMapDataIndex(mask, fdstart);

            bitptr = hypre_StructVectorBoxData(mask, b);

            hypre_BoxLoop1Begin(ndim, loop_size,
                                fdbox, fdstart, fdstride, fi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,fi) HYPRE_SMP_SCHEDULE
#endif
            hypre_BoxLoop1For(fi)
            {
               bitptr[fi] = ((HYPRE_Int) bitptr[fi]) | bitval;
            }
            hypre_BoxLoop1End(fi);
         }
      }
   }

   /* Update matrix and bit mask ghost layers in just one communication stage */
   {
      hypre_CommInfo        *comm_info;
      hypre_CommPkg         *comm_pkg_a[MAXTERMS];
      HYPRE_Complex        **comm_data_a[MAXTERMS];
      hypre_CommPkg         *comm_pkg;
      HYPRE_Complex        **comm_data;
      hypre_CommHandle      *comm_handle;
      HYPRE_Int              np, nb;

      np = 0;
      nb = 0;

      /* Compute matrix communications */
      for (m = 0; m < nmatrices; m++)
      {
         matrix = matrices[m];

         if (hypre_StructMatrixNumValues(matrix) > 0)
         {
            hypre_CreateCommInfo(grid, comm_stencils[m], &comm_info);
            hypre_StructMatrixCreateCommPkg(matrix, comm_info, &comm_pkg_a[np], &comm_data_a[np]);
            nb += hypre_CommPkgNumBlocks(comm_pkg_a[np]);
            np++;
         }
      }

      /* Compute bit mask communications */
      if (need_mask)
      {
         hypre_CreateCommInfo(grid, comm_stencils[nmatrices], &comm_info);
         hypre_StructVectorMapCommInfo(mask, comm_info);
         hypre_CommPkgCreate(comm_info,
                             hypre_StructVectorDataSpace(mask),
                             hypre_StructVectorDataSpace(mask), 1, NULL, 0,
                             hypre_StructVectorComm(mask), &comm_pkg_a[np]);
         hypre_CommInfoDestroy(comm_info);
         comm_data_a[np] = hypre_TAlloc(HYPRE_Complex *, 1);
         comm_data_a[np][0] = hypre_StructVectorData(mask);
         nb++;
         np++;
      }

      /* Put everything into one CommPkg */
      hypre_CommPkgAgglomerate(np, comm_pkg_a, &comm_pkg);
      comm_data = hypre_TAlloc(HYPRE_Complex *, nb);
      nb = 0;
      for (i = 0; i < np; i++)
      {
         for (j = 0; j < hypre_CommPkgNumBlocks(comm_pkg_a[i]); j++)
         {
            comm_data[nb++] = comm_data_a[i][j];
         }
         hypre_CommPkgDestroy(comm_pkg_a[i]);
         hypre_TFree(comm_data_a[i]);
      }

      /* Communicate */
      hypre_InitializeCommunication(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
      hypre_FinalizeCommunication(comm_handle);
      hypre_CommPkgDestroy(comm_pkg);
      hypre_TFree(comm_data);
   }

   /* Set a.types[] values */
   for (i = 0; i < na; i++)
   {
      for (t = 0; t < nterms; t++)
      {
         st_term = &(a[i].terms[t]);
         id = hypre_StTermID(st_term);
         entry = hypre_StTermEntry(st_term);
         m = terms[id];
         matrix = matrices[m];
         a[i].types[t] = mtypes[m];
         if (hypre_StructMatrixConstEntry(matrix, entry))
         {
            a[i].types[t] = 2;
         }
      }
   }

   /* Set the loop_stride for the boxloop (the larger of ran_stride and dom_stride) */
   loop_stride = dom_stride;
   for (d = 0; d < ndim; d++)
   {
      if (ran_stride[d] > dom_stride[d])
      {
         loop_stride = ran_stride;
         break;
      }
   }

   /* Set the data strides for the boxloop */
   hypre_CopyToIndex(loop_stride, ndim, Mdstride);
   hypre_MapToCoarseIndex(Mdstride, NULL, Mstride, ndim); /* Should be Mdstride = 1 */
   hypre_CopyToIndex(loop_stride, ndim, fdstride);
   hypre_MapToCoarseIndex(fdstride, NULL, fstride, ndim);
   hypre_CopyToIndex(loop_stride, ndim, cdstride);
   hypre_MapToCoarseIndex(cdstride, NULL, cstride, ndim); /* Should be cdstride = 1 */
      
   b = 0;
   for (Mj = 0; Mj < hypre_StructMatrixRanNBoxes(M); Mj++)
   {
      HYPRE_Int  *grid_ids  = hypre_StructGridIDs(grid);
      HYPRE_Int  *Mgrid_ids = hypre_StructGridIDs(Mgrid);

      Mb = hypre_StructMatrixRanBoxnum(M, Mj);
      while (grid_ids[b] != Mgrid_ids[Mb])
      {
         b++;
      }

      /* This allows a full stencil computation without having to change the
       * loop start and loop_size values (DomainIsCoarse case).  It also
       * ensures that the loop_box imin and imax are in the range space
       * (RangeIsCoarse case).  The loop_box is on the base index space. */
      hypre_CopyBox(hypre_StructGridBox(Mgrid, Mb), loop_box);
      hypre_StructMatrixMapDataBox(M, loop_box);
      hypre_StructMatrixUnMapDataBox(M, loop_box);
      hypre_RefineBox(loop_box, NULL, coarsen_stride); /* Maps to the base index space */

      /* Set the loop information in terms of the base index space */
      loop_start = hypre_BoxIMin(loop_box);
      loop_stride = cstride;
      hypre_BoxGetStrideSize(loop_box, loop_stride, loop_size);

      /* Set the data boxes and data start information for the boxloop.  Note
       * that neither MatrixMapDataIndex nor VectorMapDataIndex is used here,
       * because we want to use both matrices and vectors in one boxloop.  This
       * is accounted for when setting the data pointer values a.tpr[] below. */
      Mdbox = hypre_BoxArrayBox(Mdata_space, Mb);
      fdbox = hypre_BoxArrayBox(fdata_space, b);
      cdbox = hypre_BoxArrayBox(cdata_space, b);
      hypre_CopyToIndex(loop_start, ndim, Mdstart);
      hypre_MapToCoarseIndex(Mdstart, NULL, Mstride, ndim);   /* at loop_start */
      hypre_CopyToIndex(hypre_BoxIMin(fdbox), ndim, fdstart); /* at beginning of databox */
      hypre_CopyToIndex(hypre_BoxIMin(cdbox), ndim, cdstart); /* at beginning of databox */

      /* Set data pointers a.tptrs[] and a.mptr[].  For a.tptrs[], use Mstart to
       * compute an offset from the beginning of the databox data. */
      for (i = 0; i < na; i++)
      {
         Mentry = a[i].mentry;
         a[i].mptr = hypre_StructMatrixBoxData(M, Mb, Mentry);

         hypre_StructMatrixPlaceStencil(M, Mentry, Mdstart, Mstart); /* M's index space */
         hypre_MapToFineIndex(Mstart, NULL, coarsen_stride, ndim);   /* base index space */
         for (t = 0; t < nterms; t++)
         {
            st_term = &(a[i].terms[t]);
            id = hypre_StTermID(st_term);
            entry = hypre_StTermEntry(st_term);
            shift = hypre_StTermShift(st_term);
            m = terms[id];
            matrix = matrices[m];

            hypre_AddIndexes(Mstart, shift, ndim, tdstart); /* still on base index space */
            switch (a[i].types[t])
            {
               case 0: /* variable coefficient on fine data space */
                  hypre_StructMatrixMapDataIndex(matrix, tdstart); /* now on data space */
                  a[i].tptrs[t] = hypre_StructMatrixBoxData(matrix, b, entry) +
                     hypre_BoxIndexRank(fdbox, tdstart);
                  break;

               case 1: /* variable coefficient on coarse data space */
                  hypre_StructMatrixMapDataIndex(matrix, tdstart); /* now on data space */
                  a[i].tptrs[t] = hypre_StructMatrixBoxData(matrix, b, entry) +
                     hypre_BoxIndexRank(cdbox, tdstart);
                  break;

               case 2: /* constant coefficient - point to bit mask */
                  if (!transposes[id])
                  {
                     stencil = hypre_StructMatrixStencil(matrix);
                     offsetref = hypre_StructStencilOffset(stencil, entry);
                     hypre_AddIndexes(tdstart, offsetref, ndim, tdstart);
                  }
                  hypre_StructVectorMapDataIndex(mask, tdstart); /* now on data space */
                  a[i].tptrs[t] = hypre_StructVectorBoxData(mask, b) +
                     hypre_BoxIndexRank(fdbox, tdstart);
                  break;
            }
         }
      }

      hypre_BoxLoop3Begin(ndim, loop_size,
                          Mdbox, Mdstart, Mdstride, Mi,
                          fdbox, fdstart, fdstride, fi,
                          cdbox, cdstart, cdstride, ci);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,Mi,fi,ci,prod,t) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop3For(Mi,fi,ci)
      {
         for (i = 0; i < na; i++)
         {
            prod = a[i].cprod;
            for (t = 0; t < nterms; t++)
            {
               HYPRE_Complex pprod;
               switch (a[i].types[t])
               {
                  case 0: /* variable coefficient on fine data space */
                     pprod = a[i].tptrs[t][fi];
                     break;

                  case 1: /* variable coefficient on coarse data space */
                     pprod = a[i].tptrs[t][ci];
                     break;

                  case 2: /* constant coefficient - multiply by bit mask value t */
                     pprod = (((HYPRE_Int) a[i].tptrs[t][fi]) >> t) & 1;
                     break;
               }
               prod *= pprod;
            }
            a[i].mptr[Mi] += prod;
         }
      }
      hypre_BoxLoop3End(Mi,fi,ci);

   } /* end loop over matrix M range boxes */

   /* Restore the matrices */
   for (m = 0; m < nmatrices; m++)
   {
      hypre_StructMatrixRestore(matrices[m]);
   }

   /* Free up some stuff */
   hypre_StMatrixDestroy(st_M);
   hypre_TFree(matrices);
   hypre_TFree(terms);
   hypre_TFree(a);
   for (m = 0; m < nmatrices+1; m++)
   {
      hypre_CommStencilDestroy(comm_stencils[m]);
   }
   hypre_TFree(comm_stencils);
   if (need_mask)
   {
      hypre_StructVectorDestroy(mask);
   }
   hypre_BoxDestroy(loop_box);
   hypre_TFree(mtypes);
   hypre_BoxArrayDestroy(fdata_space);
   hypre_BoxArrayDestroy(cdata_space);
   hypre_TFree(data_spaces);

   HYPRE_StructMatrixAssemble(M);
   return hypre_error_flag;
}
