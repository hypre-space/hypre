/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured matrix-matrix multiply functions
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "struct_matmult_core.h"

#ifdef HYPRE_UNROLL_MAXDEPTH
#undef HYPRE_UNROLL_MAXDEPTH
#endif
#define HYPRE_UNROLL_MAXDEPTH 1
#define UNROLL_MAXDEPTH 7
#define NEW_UNROLLED 1

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCreate
 *
 * Creates the data structure for computing struct matrix-matrix
 * multiplication.
 *
 * The matrix product has 'nterms' terms constructed from the matrices
 * in the 'matrices' array. Each term t is given by the matrix
 * matrices[terms[t]] transposed according to the boolean transposes[t].
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCreate( HYPRE_Int                  nmatrices_in,
                           hypre_StructMatrix       **matrices_in,
                           HYPRE_Int                  nterms,
                           HYPRE_Int                 *terms_in,
                           HYPRE_Int                 *transposes_in,
                           hypre_StructMatmultData  **mmdata_ptr )
{
   hypre_StructMatmultData   *mmdata;

   hypre_StructMatrix       **matrices;
   HYPRE_Int                 *terms;
   HYPRE_Int                 *transposes;
   HYPRE_Int                 *mtypes;

   hypre_CommPkg            **comm_pkg_a;
   HYPRE_Complex           ***comm_data_a;

   HYPRE_Int                  nmatrices, *matmap;
   HYPRE_Int                  m, t;

   /* Allocate data structure */
   mmdata = hypre_CTAlloc(hypre_StructMatmultData, 1, HYPRE_MEMORY_HOST);

   /* Create new matrices and terms arrays from the input arguments, because we
    * only want to consider those matrices actually involved in the multiply */
   matmap = hypre_CTAlloc(HYPRE_Int, nmatrices_in, HYPRE_MEMORY_HOST);
   for (t = 0; t < nterms; t++)
   {
      m = terms_in[t];
      matmap[m] = 1;
   }
   nmatrices = 0;
   for (m = 0; m < nmatrices_in; m++)
   {
      if (matmap[m])
      {
         matmap[m] = nmatrices;
         nmatrices++;
      }
   }

   matrices   = hypre_CTAlloc(hypre_StructMatrix *, nmatrices, HYPRE_MEMORY_HOST);
   terms      = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
   transposes = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
   for (t = 0; t < nterms; t++)
   {
      m = terms_in[t];
      matrices[matmap[m]] = matrices_in[m];
      terms[t] = matmap[m];
      transposes[t] = transposes_in[t];
   }
   hypre_TFree(matmap, HYPRE_MEMORY_HOST);

   /* Initialize */
   comm_pkg_a  = hypre_TAlloc(hypre_CommPkg *, nmatrices + 1, HYPRE_MEMORY_HOST);
   comm_data_a = hypre_TAlloc(HYPRE_Complex **, nmatrices + 1, HYPRE_MEMORY_HOST);

   /* Initialize mtypes to fine data spaces */
   mtypes = hypre_CTAlloc(HYPRE_Int, nmatrices+1, HYPRE_MEMORY_HOST);

   /* Initialize data members */
   (mmdata -> nmatrices)       = nmatrices;
   (mmdata -> matrices)        = matrices;
   (mmdata -> nterms)          = nterms;
   (mmdata -> terms)           = terms;
   (mmdata -> transposes)      = transposes;
   (mmdata -> mtypes)          = mtypes;
   (mmdata -> fstride)         = NULL;
   (mmdata -> cstride)         = NULL;
   (mmdata -> coarsen_stride)  = NULL;
   (mmdata -> cdata_space)     = NULL;
   (mmdata -> fdata_space)     = NULL;
   (mmdata -> coarsen)         = 0;
   (mmdata -> mask)            = NULL;
   (mmdata -> st_M)            = NULL;
   (mmdata -> a)               = NULL;
   (mmdata -> na)              = 0;
   (mmdata -> comm_pkg)        = NULL;
   (mmdata -> comm_pkg_a)      = comm_pkg_a;
   (mmdata -> comm_data)       = NULL;
   (mmdata -> comm_data_a)     = comm_data_a;
   (mmdata -> num_comm_pkgs)   = 0;
   (mmdata -> num_comm_blocks) = 0;

   *mmdata_ptr = mmdata;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultDestroy( hypre_StructMatmultData *mmdata )
{
   if (mmdata)
   {
      hypre_TFree(mmdata -> matrices, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> transposes, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> terms, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> mtypes, HYPRE_MEMORY_HOST);

      hypre_TFree(mmdata -> a, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> comm_pkg_a, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> comm_data_a, HYPRE_MEMORY_HOST);

      hypre_BoxArrayDestroy(mmdata -> fdata_space);
      hypre_BoxArrayDestroy(mmdata -> cdata_space);
      hypre_StMatrixDestroy(mmdata -> st_M);
      hypre_StructVectorDestroy(mmdata -> mask);

      hypre_CommPkgDestroy(mmdata -> comm_pkg);
      hypre_TFree(mmdata -> comm_data, HYPRE_MEMORY_HOST);

      hypre_TFree(mmdata, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultSetup
 *
 * Compute and assemble the StructGrid of the resulting matrix
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
hypre_StructMatmultSetup( hypre_StructMatmultData  *mmdata,
                          hypre_StructMatrix      **M_ptr )
{
   HYPRE_Int                  nterms       = (mmdata -> nterms);
   HYPRE_Int                 *terms        = (mmdata -> terms);
   HYPRE_Int                 *transposes   = (mmdata -> transposes);
   HYPRE_Int                  nmatrices    = (mmdata -> nmatrices);
   HYPRE_Int                 *mtypes       = (mmdata -> mtypes);
   hypre_StructMatrix       **matrices     = (mmdata -> matrices);
   hypre_CommPkg            **comm_pkg_a   = (mmdata -> comm_pkg_a);
   HYPRE_Complex           ***comm_data_a  = (mmdata -> comm_data_a);

   hypre_StructMatrix        *M;            /* matrix product we are computing */
   hypre_StructStencil       *Mstencil;
   hypre_StructGrid          *Mgrid;
   hypre_Index                Mran_stride, Mdom_stride;

   MPI_Comm                   comm;
   HYPRE_Int                  ndim, size;

   hypre_StructMatrix        *matrix;
   hypre_StructStencil       *stencil;
   hypre_StructGrid          *grid;
   hypre_IndexRef             stride;
   HYPRE_Int                  nboxes;
   HYPRE_Int                 *boxnums;
   hypre_Box                 *box;

   hypre_StMatrix           **st_matrices, *st_matrix, *st_M;
   hypre_StCoeff             *st_coeff;
   hypre_StTerm              *st_term;

   hypre_StructMatmultHelper *a;
   HYPRE_Int                  na;              /* number of product terms in 'a' */
   HYPRE_Int                  nconst;          /* number of constant entries in M */
   HYPRE_Int                  const_entry;     /* boolean for constant entry in M */
   HYPRE_Int                 *const_entries;   /* constant entries in M */
   HYPRE_Complex             *const_values;    /* values for constant entries in M */

   hypre_CommInfo            *comm_info;
   hypre_CommStencil        **comm_stencils;
   HYPRE_Int                  num_comm_blocks;
   HYPRE_Int                  num_comm_pkgs;

   hypre_StructVector        *mask;
   HYPRE_Int                  need_mask;          /* boolean indicating if a bit mask is needed */
   HYPRE_Int                  const_term, var_term; /* booleans used to determine 'need_mask' */

   hypre_IndexRef             ran_stride;
   hypre_IndexRef             dom_stride;
   hypre_IndexRef             coarsen_stride;
   HYPRE_Int                  coarsen;
   HYPRE_Complex             *constp;          /* pointer to constant data */
   HYPRE_Complex             *bitptr;          /* pointer to bit mask data */
   hypre_Index                offset;          /* CommStencil offset */
   hypre_IndexRef             shift;           /* stencil shift from center for st_term */
   hypre_IndexRef             offsetref;
   HYPRE_Int                  d, i, j, m, t, e, b, id, entry;

   hypre_Box                 *loop_box;    /* boxloop extents on the base index space */
   hypre_IndexRef             loop_start;  /* boxloop start index on the base index space */
   hypre_IndexRef             loop_stride; /* boxloop stride on the base index space */
   hypre_Index                loop_size;   /* boxloop size */
   hypre_IndexRef             fstride, cstride; /* data-map strides (base index space) */
   hypre_Index                fdstart;  /* boxloop data starts */
   hypre_Index                fdstride; /* boxloop data strides */
   hypre_Box                 *fdbox;    /* boxloop data boxes */

   hypre_BoxArray           **data_spaces;
   hypre_BoxArray            *cdata_space, *fdata_space, *data_space;

   HYPRE_ANNOTATE_FUNC_BEGIN;

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
   st_matrices = hypre_CTAlloc(hypre_StMatrix *, nterms, HYPRE_MEMORY_HOST);
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
   (mmdata -> st_M) = st_M;

   /* Free up st_matrices */
   for (t = 0; t < nterms; t++)
   {
      hypre_StMatrixDestroy(st_matrices[t]);
   }
   hypre_TFree(st_matrices, HYPRE_MEMORY_HOST);

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
   (mmdata -> coarsen_stride) = coarsen_stride;

   /* This flag indicates whether Mgrid will be constructed by
      coarsening the grid that belongs to matrices[0] or not */
   coarsen = 0;
   for (d = 0; d < ndim; d++)
   {
      if (coarsen_stride[d] > 1)
      {
         coarsen = 1;
         break;
      }
   }
   (mmdata -> coarsen) = coarsen;

   /* Create Mgrid (the grid for M) */
   grid = hypre_StructMatrixGrid(matrices[0]); /* Same grid for all matrices */
   hypre_CopyToIndex(ran_stride, ndim, Mran_stride);
   hypre_CopyToIndex(dom_stride, ndim, Mdom_stride);
   if (coarsen)
   {
      /* Note: Mgrid may have fewer boxes than grid as a result of coarsening */
      hypre_StructCoarsen(grid, NULL, coarsen_stride, 1, &Mgrid);
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
    * contributions stored in 'a[i].cprod'. Communication stencils are also
    * computed for each matrix (not each term, so matrices that appear in more
    * than one term in the product are dealt with only once).  Communication
    * stencils are then used to determine new data spaces for resizing the
    * matrices.  Since we assume there are at most two data-map strides, only
    * two data spaces are computed, one fine and one coarse.  This simplifies
    * the boxloop below and allow us to use a BoxLoop3.  We add an extra entry
    * to the end of 'comm_stencils' and 'data_spaces' for the bit mask, in case
    * a bit mask is needed. */

   const_entries = hypre_TAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
   const_values  = hypre_TAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);

   /* Allocate `a` and compute initial value for `na`. */
   a = hypre_TAlloc(hypre_StructMatmultHelper, na, HYPRE_MEMORY_HOST);
   (mmdata -> a) = a;

   /* Allocate memory for communication stencils */
   comm_stencils = hypre_TAlloc(hypre_CommStencil *, nmatrices+1, HYPRE_MEMORY_HOST);
   for (m = 0; m < nmatrices+1; m++)
   {
      comm_stencils[m] = hypre_CommStencilCreate(ndim);
   }

   na = 0;
   nconst = 0;
   need_mask = 0;
   if (hypre_StructGridNumBoxes(grid) > 0)
   {
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

            /* Visit next coeffcient */
            st_coeff = hypre_StCoeffNext(st_coeff);
            i++;
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

   /* Update na */
   (mmdata -> na) = na;

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
   hypre_TFree(const_entries, HYPRE_MEMORY_HOST);
   hypre_TFree(const_values, HYPRE_MEMORY_HOST);

   /* Return if all constant coefficients or no boxes */
   if (nconst == size || !(hypre_StructGridNumBoxes(grid) > 0))
   {
      /* Free up memory */
      for (m = 0; m < nmatrices+1; m++)
      {
         hypre_CommStencilDestroy(comm_stencils[m]);
      }
      hypre_TFree(comm_stencils, HYPRE_MEMORY_HOST);

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   /* Create a bit mask with bit data for each matrix term that has constant
    * coefficients to prevent incorrect contributions in the matrix product.
    * The bit mask is a vector with appropriately set bits and updated ghost
    * layer to account for parallelism and periodic boundary conditions. */

   loop_box = hypre_BoxCreate(ndim);

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
   (mmdata -> fstride) = fstride;
   (mmdata -> cstride) = cstride;

   /* Compute mtypes (assumes only two data-map strides) */
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
   data_spaces = hypre_CTAlloc(hypre_BoxArray *, nmatrices+1, HYPRE_MEMORY_HOST);
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
      hypre_TFree(num_ghost, HYPRE_MEMORY_HOST);
   }

   /* Compute initial bit mask data space */
   if (need_mask)
   {
      HYPRE_Int  *num_ghost;

      HYPRE_StructVectorCreate(comm, grid, &mask);
      HYPRE_StructVectorSetStride(mask, fstride); /* same stride as fine data-map stride */
      hypre_CommStencilCreateNumGhost(comm_stencils[nmatrices], &num_ghost);
      hypre_StructVectorComputeDataSpace(mask, num_ghost, &data_spaces[nmatrices]);
      hypre_TFree(num_ghost, HYPRE_MEMORY_HOST);
      (mmdata -> mask) = mask;
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
   (mmdata -> cdata_space) = cdata_space;
   (mmdata -> fdata_space) = fdata_space;

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

      // VPM: Should we call hypre_StructMatrixForget?
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

#define DEVICE_VAR is_device_ptr(bitptr)
            hypre_BoxLoop1Begin(ndim, loop_size,
                                fdbox, fdstart, fdstride, fi);
            {
               bitptr[fi] = ((HYPRE_Int) bitptr[fi]) | bitval;
            }
            hypre_BoxLoop1End(fi);
#undef DEVICE_VAR
         }
      }
   }

   /* Setup agglomerated communication packages for matrices and bit mask ghost layers */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "CommCreate");
   {
      /* Initialize number of packages and blocks */
      num_comm_pkgs = num_comm_blocks = 0;

      /* Compute matrix communications */
      for (m = 0; m < nmatrices; m++)
      {
         matrix = matrices[m];

         if (hypre_StructMatrixNumValues(matrix) > 0)
         {
            hypre_CreateCommInfo(grid, comm_stencils[m], &comm_info);
            hypre_StructMatrixCreateCommPkg(matrix, comm_info, &comm_pkg_a[num_comm_pkgs],
                                            &comm_data_a[num_comm_pkgs]);
            num_comm_blocks += hypre_CommPkgNumBlocks(comm_pkg_a[num_comm_pkgs]);
            num_comm_pkgs++;
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
                             hypre_StructVectorComm(mask), &comm_pkg_a[num_comm_pkgs]);
         hypre_CommInfoDestroy(comm_info);
         comm_data_a[num_comm_pkgs] = hypre_TAlloc(HYPRE_Complex *, 1, HYPRE_MEMORY_HOST);
         comm_data_a[num_comm_pkgs][0] = hypre_StructVectorData(mask);
         num_comm_blocks++;
         num_comm_pkgs++;
      }
      (mmdata -> num_comm_pkgs)   = num_comm_pkgs;
      (mmdata -> num_comm_blocks) = num_comm_blocks;
   }
   HYPRE_ANNOTATE_REGION_END("%s", "CommCreate");

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

   /* Free memory */
   hypre_BoxDestroy(loop_box);
   hypre_TFree(data_spaces, HYPRE_MEMORY_HOST);
   for (m = 0; m < nmatrices+1; m++)
   {
      hypre_CommStencilDestroy(comm_stencils[m]);
   }
   hypre_TFree(comm_stencils, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * StructMatmultCommunicate
 *
 * Communicates matrix and bit mask info with a single commpkg.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCommunicate( hypre_StructMatmultData  *mmdata,
                                hypre_StructMatrix       *M )
{
   hypre_StructGrid   *grid = hypre_StructMatrixGrid(M);

   hypre_CommPkg      *comm_pkg        = (mmdata -> comm_pkg);
   HYPRE_Complex     **comm_data       = (mmdata -> comm_data);
   hypre_CommPkg     **comm_pkg_a      = (mmdata -> comm_pkg_a);
   HYPRE_Complex    ***comm_data_a     = (mmdata -> comm_data_a);
   HYPRE_Int           num_comm_pkgs   = (mmdata -> num_comm_pkgs);
   HYPRE_Int           num_comm_blocks = (mmdata -> num_comm_blocks);

   hypre_CommHandle   *comm_handle;
   HYPRE_Int           i, j, nb;

   /* Assemble the grid. Note: StructGridGlobalSize is updated to zero so that
    * its computation is triggered in hypre_StructGridAssemble */
   hypre_StructGridGlobalSize(grid) = 0;
   hypre_StructGridAssemble(grid);

   /* If all constant coefficients, return */
   if (mmdata -> na == 0)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Agglomerate communication packages if needed */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "CommSetup");
   if (!comm_pkg || !comm_data)
   {
      hypre_CommPkgAgglomerate(num_comm_pkgs, comm_pkg_a, &comm_pkg);
      comm_data = hypre_TAlloc(HYPRE_Complex *, num_comm_blocks, HYPRE_MEMORY_HOST);
      nb = 0;
      for (i = 0; i < num_comm_pkgs; i++)
      {
         for (j = 0; j < hypre_CommPkgNumBlocks(comm_pkg_a[i]); j++)
         {
            comm_data[nb++] = comm_data_a[i][j];
         }
         hypre_CommPkgDestroy(comm_pkg_a[i]);
         hypre_TFree(comm_data_a[i], HYPRE_MEMORY_HOST);
      }

      /* Update communication info */
      mmdata -> comm_pkg  = comm_pkg;
      mmdata -> comm_data = comm_data;
   }
   HYPRE_ANNOTATE_REGION_END("%s", "CommSetup");

   hypre_InitializeCommunication(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
   hypre_FinalizeCommunication(comm_handle);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * StructMatmultCompute
 *
 * Computes coefficients of the resulting matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute( hypre_StructMatmultData  *mmdata,
                            hypre_StructMatrix       *M )
{
   hypre_StructMatmultHelper *a          = (mmdata -> a);
   HYPRE_Int              na             = (mmdata -> na);
   HYPRE_Int              nmatrices      = (mmdata -> nmatrices);
   HYPRE_Int              nterms         = (mmdata -> nterms);
   HYPRE_Int             *terms          = (mmdata -> terms);
   HYPRE_Int             *transposes     = (mmdata -> transposes);
   hypre_StructMatrix   **matrices       = (mmdata -> matrices);
   hypre_BoxArray        *fdata_space    = (mmdata -> fdata_space);
   hypre_BoxArray        *cdata_space    = (mmdata -> cdata_space);
   hypre_StructVector    *mask           = (mmdata -> mask);
   hypre_IndexRef         fstride        = (mmdata -> fstride);
   hypre_IndexRef         cstride        = (mmdata -> cstride);
   hypre_IndexRef         coarsen_stride = (mmdata -> coarsen_stride);

   /* Input matrices variables */
   HYPRE_Int              ndim;
   hypre_StructMatrix    *matrix;
   hypre_StructStencil   *stencil;
   hypre_StructGrid      *grid;
   HYPRE_Int             *grid_ids;
   HYPRE_Int              stencil_size;

   /* M matrix variables */
   hypre_StructGrid      *Mgrid       = hypre_StructMatrixGrid(M);
   hypre_StructStencil   *Mstencil    = hypre_StructMatrixStencil(M);
   HYPRE_Int              size        = hypre_StructStencilSize(Mstencil);
   HYPRE_Int             *Mgrid_ids   = hypre_StructGridIDs(Mgrid);
   hypre_BoxArray        *Mdata_space = hypre_StructMatrixDataSpace(M);
   hypre_StTerm          *st_term; /* Pointer to stencil info for each term in a */

   /* Local variables */
   hypre_Index            Mstart;      /* M's stencil location on the base index space */
   hypre_Box             *loop_box;    /* boxloop extents on the base index space */
   hypre_IndexRef         loop_start;  /* boxloop start index on the base index space */
   hypre_IndexRef         loop_stride; /* boxloop stride on the base index space */
   hypre_Index            loop_size;   /* boxloop size */
   hypre_Index            Mstride;     /* data-map stride  (base index space) */
   hypre_IndexRef         offsetref;   /* offset for constant coefficient stencil entries */
   hypre_IndexRef         shift;       /* stencil shift from center for st_term */
   hypre_IndexRef         stride;

   /* Boxloop variables */
   hypre_Index            fdstart,  cdstart,  Mdstart;  /* data starts */
   hypre_Index            fdstride, cdstride, Mdstride; /* data strides */
   hypre_Box             *fdbox,   *cdbox,   *Mdbox;    /* data boxes */
   hypre_Index            tdstart;

   /* Work pointers */
   HYPRE_Complex        **dptrs;

   /* Indices */
   HYPRE_Int              entry, Mentry;
   HYPRE_Int              Mj, Mb;
   HYPRE_Int              b, e, i, id, m, t;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* If all constant coefficients or no boxes, return */
   if (na == 0)
   {
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /* Initialize data */
   ndim     = hypre_StructMatrixNDim(matrices[0]);
   grid     = hypre_StructMatrixGrid(matrices[0]);
   grid_ids = hypre_StructGridIDs(grid);
   loop_box = hypre_BoxCreate(ndim);

   /* Allocate dptrs */
   dptrs = hypre_TAlloc(HYPRE_Complex *, nterms, HYPRE_MEMORY_HOST);
   for (t = 0; t < nterms; t++)
   {
      m = terms[t];
      dptrs[t] = hypre_StructMatrixData(matrices[m]);
   }

   /* Set Mstride */
   hypre_StructMatrixGetDataMapStride(M, &stride);
   hypre_CopyToIndex(stride, ndim, Mstride);                    /* M's index space */
   hypre_MapToFineIndex(Mstride, NULL, coarsen_stride, ndim);   /* base index space */

   /* Set the loop_stride for the boxloop (the larger of ran_stride and dom_stride) */
   loop_stride = cstride;

   /* Set the data strides for the boxloop */
   hypre_CopyToIndex(loop_stride, ndim, Mdstride);
   hypre_MapToCoarseIndex(Mdstride, NULL, Mstride, ndim); /* Should be Mdstride = 1 */
   hypre_CopyToIndex(loop_stride, ndim, fdstride);
   hypre_MapToCoarseIndex(fdstride, NULL, fstride, ndim);
   hypre_CopyToIndex(loop_stride, ndim, cdstride);
   hypre_MapToCoarseIndex(cdstride, NULL, cstride, ndim); /* Should be cdstride = 1 */

   b = 0;
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Computation");
   for (Mj = 0; Mj < hypre_StructMatrixRanNBoxes(M); Mj++)
   {
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
       * is accounted for when setting the data pointer values a.tptrs[] below. */
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
                  a[i].offsets[t] = hypre_StructMatrixDataIndices(matrix)[b][entry] +
                     hypre_BoxIndexRank(fdbox, tdstart);
                  break;

               case 1: /* variable coefficient on coarse data space */
                  hypre_StructMatrixMapDataIndex(matrix, tdstart); /* now on data space */
                  a[i].tptrs[t] = hypre_StructMatrixBoxData(matrix, b, entry) +
                     hypre_BoxIndexRank(cdbox, tdstart);
                  a[i].offsets[t] = hypre_StructMatrixDataIndices(matrix)[b][entry] +
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
                  a[i].offsets[t] = hypre_StructVectorDataIndices(mask)[b] +
                     hypre_BoxIndexRank(fdbox, tdstart);
                  break;
            }
         }
      } /* end loop over a entries */

      /* Compute M coefficients for box Mb */
      switch (nterms)
      {
         case 2:
            hypre_StructMatmultCompute_core_double(a, na, ndim, loop_size,
                                                   fdbox, fdstart, fdstride,
                                                   cdbox, cdstart, cdstride,
                                                   Mdbox, Mdstart, Mdstride);
            break;

         case 3:
            hypre_StructMatmultCompute_core_triple(a, na, ndim, loop_size, size,
                                                   fdbox, fdstart, fdstride,
                                                   cdbox, cdstart, cdstride,
                                                   Mdbox, Mdstart, Mdstride,
                                                   terms, dptrs);
            break;

         default:
            hypre_StructMatmultCompute_core_generic(a, na, nterms, ndim, loop_size,
                                                    fdbox, fdstart, fdstride,
                                                    cdbox, cdstart, cdstride,
                                                    Mdbox, Mdstart, Mdstride);
            break;
      }
   } /* end loop over matrix M range boxes */
   HYPRE_ANNOTATE_REGION_END("%s", "Computation");

   /* Restore the matrices */
   for (m = 0; m < nmatrices; m++)
   {
      hypre_StructMatrixRestore(matrices[m]);
   }

   /* Free memory */
   hypre_BoxDestroy(loop_box);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_double
 *
 * Core function for computing the double product of coefficients.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_double( hypre_StructMatmultHelper *a,
                                        HYPRE_Int    na,
                                        HYPRE_Int    ndim,
                                        hypre_Index  loop_size,
                                        hypre_Box   *fdbox,
                                        hypre_Index  fdstart,
                                        hypre_Index  fdstride,
                                        hypre_Box   *cdbox,
                                        hypre_Index  cdstart,
                                        hypre_Index  cdstride,
                                        hypre_Box   *Mdbox,
                                        hypre_Index  Mdstart,
                                        hypre_Index  Mdstride )
{
   /* TODO */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_triple
 *
 * Core function for computing the triple product of coefficients
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_triple( hypre_StructMatmultHelper *a,
                                        HYPRE_Int    na,
                                        HYPRE_Int    ndim,
                                        hypre_Index  loop_size,
                                        HYPRE_Int    stencil_size,
                                        hypre_Box   *fdbox,
                                        hypre_Index  fdstart,
                                        hypre_Index  fdstride,
                                        hypre_Box   *cdbox,
                                        hypre_Index  cdstart,
                                        hypre_Index  cdstride,
                                        hypre_Box   *Mdbox,
                                        hypre_Index  Mdstart,
                                        hypre_Index  Mdstride,
                                        HYPRE_Int   *terms,
                                        HYPRE_Complex **dptrs )
{
   HYPRE_Int     *ncomp;
   HYPRE_Int    **indices;
   HYPRE_Int   ***order;

   HYPRE_Int      max_components;
   HYPRE_Int      c, i, k, t;

   /* Allocate memory */
   max_components = 10;
   ncomp   = hypre_CTAlloc(HYPRE_Int, max_components, HYPRE_MEMORY_HOST);
   indices = hypre_TAlloc(HYPRE_Int *, max_components, HYPRE_MEMORY_HOST);
   order   = hypre_TAlloc(HYPRE_Int **, max_components, HYPRE_MEMORY_HOST);
   for (c = 0; c < max_components; c++)
   {
      indices[c] = hypre_CTAlloc(HYPRE_Int, na, HYPRE_MEMORY_HOST);
      order[c] = hypre_TAlloc(HYPRE_Int *, na, HYPRE_MEMORY_HOST);
      for (t = 0; t < na; t++)
      {
         order[c][t] = hypre_CTAlloc(HYPRE_Int, 3, HYPRE_MEMORY_HOST);
      }
   }

   /* Build components arrays */
   for (i = 0; i < na; i++)
   {
      if ( a[i].types[0] == 0 &&
           a[i].types[1] == 0 &&
           a[i].types[2] == 0 )
      {
         /* VCF * VCF * VCF */
         k = ncomp[0];
         indices[0][k] = i;
         ncomp[0]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 1 )
      {
         /* VCF * VCF * VCC */
         k = ncomp[5];
         indices[5][k]  = i;
         order[5][k][0] = 0;
         order[5][k][1] = 1;
         order[5][k][2] = 2;
         ncomp[5]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 2 )
      {
         /* VCF * VCF * CCF */
         k = ncomp[2];
         indices[2][k]  = i;
         order[2][k][0] = 0;
         order[2][k][1] = 1;
         order[2][k][2] = 2;
         ncomp[2]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 0 )
      {
         /* VCF * VCC * VCF */
         k = ncomp[5];
         indices[5][k]  = i;
         order[5][k][0] = 0;
         order[5][k][1] = 2;
         order[5][k][2] = 1;
         ncomp[5]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 1 )
      {
         /* VCF * VCC * VCC */
         k = ncomp[6];
         indices[6][k]  = i;
         order[6][k][0] = 1;
         order[6][k][1] = 2;
         order[6][k][2] = 0;
         ncomp[6]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 2 )
      {
         /* VCF * VCC * CCF */
         k = ncomp[7];
         indices[7][k]  = i;
         order[7][k][0] = 0;
         order[7][k][1] = 1;
         order[7][k][2] = 2;
         ncomp[7]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 0 )
      {
         /* VCF * CCF * VCF */
         k = ncomp[2];
         indices[2][k]  = i;
         order[2][k][0] = 0;
         order[2][k][1] = 2;
         order[2][k][2] = 1;
         ncomp[2]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 1 )
      {
         /* VCF * CCF * VCC */
         k = ncomp[7];
         indices[7][k]  = i;
         order[7][k][0] = 0;
         order[7][k][1] = 2;
         order[7][k][2] = 1;
         ncomp[7]++;
      }
      else if ( a[i].types[0] == 0 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 2 )
      {
         /* VCF * CCF * CCF */
         k = ncomp[3];
         indices[3][k]  = i;
         order[3][k][0] = 0;
         order[3][k][1] = 1;
         order[3][k][2] = 2;
         ncomp[3]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 0 )
      {
         /* VCC * VCF * VCF */
         k = ncomp[5];
         indices[5][k]  = i;
         order[5][k][0] = 1;
         order[5][k][1] = 2;
         order[5][k][2] = 0;
         ncomp[5]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 1 )
      {
         /* VCC * VCF * VCC */
         k = ncomp[6];
         indices[6][k]  = i;
         order[6][k][0] = 0;
         order[6][k][1] = 2;
         order[6][k][2] = 1;
         ncomp[6]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 2 )
      {
         /* VCC * VCF * CCF */
         k = ncomp[7];
         indices[7][k]  = i;
         order[7][k][0] = 1;
         order[7][k][1] = 0;
         order[7][k][2] = 2;
         ncomp[7]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 0 )
      {
         /* VCC * VCC * VCF */
         k = ncomp[6];
         indices[6][k]  = i;
         order[6][k][0] = 0;
         order[6][k][1] = 1;
         order[6][k][2] = 2;
         ncomp[6]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 1 )
      {
         /* VCC * VCC * VCC */
         k = ncomp[1];
         indices[1][k] = i;
         ncomp[1]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 2 )
      {
         /* VCC * VCC * CCF */
         k = ncomp[8];
         indices[8][k]  = i;
         order[8][k][0] = 0;
         order[8][k][1] = 1;
         order[8][k][2] = 2;
         ncomp[8]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 0 )
      {
         /* VCC * CCF * VCF */
         k = ncomp[7];
         indices[7][k]  = i;
         order[7][k][0] = 2;
         order[7][k][1] = 0;
         order[7][k][2] = 1;
         ncomp[7]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 1 )
      {
         /* VCC * CCF * VCC */
         k = ncomp[8];
         indices[8][k]  = i;
         order[8][k][0] = 0;
         order[8][k][1] = 2;
         order[8][k][2] = 1;
         ncomp[8]++;
      }
      else if ( a[i].types[0] == 1 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 2 )
      {
         /* VCC * CCF * CCF */
         k = ncomp[9];
         indices[9][k]  = i;
         order[9][k][0] = 0;
         order[9][k][1] = 1;
         order[9][k][2] = 2;
         ncomp[9]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 0 )
      {
         /* CCF * VCF * VCF */
         k = ncomp[2];
         indices[2][k]  = i;
         order[2][k][0] = 1;
         order[2][k][1] = 2;
         order[2][k][2] = 0;
         ncomp[2]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 1 )
      {
         /* CCF * VCF * VCC */
         k = ncomp[7];
         indices[7][k]  = i;
         order[7][k][0] = 1;
         order[7][k][1] = 2;
         order[7][k][2] = 0;
         ncomp[7]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 0 &&
                a[i].types[2] == 2 )
      {
         /* CCF * VCF * CCF */
         k = ncomp[3];
         indices[3][k]  = i;
         order[3][k][0] = 1;
         order[3][k][1] = 0;
         order[3][k][2] = 2;
         ncomp[3]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 0 )
      {
         /* CCF * VCC * VCF */
         k = ncomp[7];
         indices[7][k]  = i;
         order[7][k][0] = 2;
         order[7][k][1] = 1;
         order[7][k][2] = 0;
         ncomp[7]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 1 )
      {
         /* CCF * VCC * VCC */
         k = ncomp[8];
         indices[8][k]  = i;
         order[8][k][0] = 1;
         order[8][k][1] = 2;
         order[8][k][2] = 0;
         ncomp[8]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 1 &&
                a[i].types[2] == 2 )
      {
         /* CCF * VCC * CCF */
         k = ncomp[9];
         indices[9][k]  = i;
         order[9][k][0] = 1;
         order[9][k][1] = 0;
         order[9][k][2] = 2;
         ncomp[9]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 0 )
      {
         /* CCF * CCF * VCF */
         k = ncomp[3];
         indices[3][k]  = i;
         order[3][k][0] = 2;
         order[3][k][1] = 0;
         order[3][k][2] = 1;
         ncomp[3]++;
      }
      else if ( a[i].types[0] == 2 &&
                a[i].types[1] == 2 &&
                a[i].types[2] == 1 )
      {
         /* CCF * CCF * VCC */
         k = ncomp[9];
         indices[9][k]  = i;
         order[9][k][0] = 2;
         order[9][k][1] = 0;
         order[9][k][2] = 1;
         ncomp[9]++;
      }
      else
      {
         /* CCF * CCF * CCF */
         k = ncomp[4];
         indices[4][k] = i;
         ncomp[4]++;
      }
   }

   /* Call core functions */
   hypre_StructMatmultCompute_core_1t(a, ncomp[0], indices[0],
                                      ndim, loop_size,
                                      fdbox, fdstart, fdstride,
                                      Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_core_1t(a, ncomp[1], indices[1],
                                      ndim, loop_size,
                                      cdbox, cdstart, cdstride,
                                      Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_core_1tb(a, ncomp[2], indices[2],
                                       order[2], ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       Mdbox, Mdstart, Mdstride);

#if !NEW_UNROLLED
   hypre_StructMatmultCompute_core_1tbb(a, ncomp[3], indices[3],
                                        order[3], ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        Mdbox, Mdstart, Mdstride);
#else
   hypre_StructMatmultCompute_core_1tbb_v2(a, ncomp[3], indices[3],
                                        order[3], ndim, loop_size, stencil_size,
                                        fdbox, fdstart, fdstride,
                                        Mdbox, Mdstart, Mdstride);
#endif

   hypre_StructMatmultCompute_core_1tbbb(a, ncomp[4], indices[4],
                                         ndim, loop_size,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_core_2t(a, ncomp[5], indices[5],
                                      order[5], ndim, loop_size,
                                      fdbox, fdstart, fdstride,
                                      cdbox, cdstart, cdstride,
                                      Mdbox, Mdstart, Mdstride);

#if !NEW_UNROLLED
   hypre_StructMatmultCompute_core_2t(a, ncomp[6], indices[6],
                                      order[6], ndim, loop_size,
                                      cdbox, cdstart, cdstride,
                                      fdbox, fdstart, fdstride,
                                      Mdbox, Mdstart, Mdstride);
#elif 0
   hypre_StructMatmultCompute_core_2t_v2(a, ncomp[6], indices[6],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride, terms, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v3a(a, ncomp[6], indices[6],
                                          ndim, loop_size,
                                          cdbox, cdstart, cdstride,
                                          fdbox, fdstart, fdstride,
                                          Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v3b(a, ncomp[6], indices[6],
                                          ndim, loop_size,
                                          cdbox, cdstart, cdstride,
                                          fdbox, fdstart, fdstride,
                                          Mdbox, Mdstart, Mdstride, dptrs);

#elif 0
   hypre_StructMatmultCompute_core_2t_v4(a, ncomp[6], indices[6],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v5(a, ncomp[6], indices[6],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v5a(a, ncomp[6], indices[6],
                                          ndim, loop_size,
                                          cdbox, cdstart, cdstride,
                                          fdbox, fdstart, fdstride,
                                          Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v5b(a, ncomp[6], indices[6],
                                          ndim, loop_size,
                                          cdbox, cdstart, cdstride,
                                          fdbox, fdstart, fdstride,
                                          Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v6(a, ncomp[6], indices[6],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v7(a, ncomp[6], indices[6],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v8(a, ncomp[6], indices[6],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride, dptrs);
#elif 0
   hypre_StructMatmultCompute_core_2t_v9(a, ncomp[6], indices[6],
                                         ndim, loop_size,
                                         cdbox, cdstart, cdstride,
                                         fdbox, fdstart, fdstride,
                                         Mdbox, Mdstart, Mdstride);
#elif NEW_UNROLLED
   hypre_StructMatmultCompute_core_2t_v10(a, ncomp[6], indices[6],
                                          ndim, loop_size, stencil_size,
                                          cdbox, cdstart, cdstride,
                                          fdbox, fdstart, fdstride,
                                          Mdbox, Mdstart, Mdstride);
#endif

#if !NEW_UNROLLED
   hypre_StructMatmultCompute_core_2tb(a, ncomp[7], indices[7],
                                       order[7], ndim, loop_size,
                                       fdbox, fdstart, fdstride,
                                       cdbox, cdstart, cdstride,
                                       Mdbox, Mdstart, Mdstride);
#else
   hypre_StructMatmultCompute_core_2tb_v2(a, ncomp[7], indices[7],
                                       order[7], ndim, loop_size, stencil_size,
                                       fdbox, fdstart, fdstride,
                                       cdbox, cdstart, cdstride,
                                       Mdbox, Mdstart, Mdstride);
#endif

   hypre_StructMatmultCompute_core_2etb(a, ncomp[8], indices[8],
                                        order[8], ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        cdbox, cdstart, cdstride,
                                        Mdbox, Mdstart, Mdstride);

   hypre_StructMatmultCompute_core_2tbb(a, ncomp[9], indices[9],
                                        order[9], ndim, loop_size,
                                        fdbox, fdstart, fdstride,
                                        cdbox, cdstart, cdstride,
                                        Mdbox, Mdstart, Mdstride);

   /* Free memory */
   for (c = 0; c < max_components; c++)
   {
      for (t = 0; t < na; t++)
      {
         hypre_TFree(order[c][t], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(indices[c], HYPRE_MEMORY_HOST);
      hypre_TFree(order[c], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(indices, HYPRE_MEMORY_HOST);
   hypre_TFree(order, HYPRE_MEMORY_HOST);
   hypre_TFree(ncomp, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_generic
 *
 * Core function for computing the product of "nterms" coefficients.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_generic( hypre_StructMatmultHelper *a,
                                         HYPRE_Int    na,
                                         HYPRE_Int    nterms,
                                         HYPRE_Int    ndim,
                                         hypre_Index  loop_size,
                                         hypre_Box   *fdbox,
                                         hypre_Index  fdstart,
                                         hypre_Index  fdstride,
                                         hypre_Box   *cdbox,
                                         hypre_Index  cdstart,
                                         hypre_Index  cdstride,
                                         hypre_Box   *Mdbox,
                                         hypre_Index  Mdstart,
                                         hypre_Index  Mdstride )
{
   /* TODO: add DEVICE_VAR */
   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       fdbox, fdstart, fdstride, fi,
                       cdbox, cdstart, cdstride, ci);
   {
      HYPRE_Int      i, t;
      HYPRE_Complex  prod;
      HYPRE_Complex  pprod;

      for (i = 0; i < na; i++)
      {
         prod = a[i].cprod;
         for (t = 0; t < nterms; t++)
         {
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_1t
 *
 * Core function for computing the triple product of variable coefficients
 * living on the same data space.
 *
 * "1t" means:
 *   "1": single data space.
 *   "t": triple product.
 *
 * This can be used for the scenarios:
 *   1) VCF * VCF * CCF.
 *   2) VCC * VCC * CCF.
 *
 * where:
 *   1) VCF stands for "Variable Coefficient on Fine data space".
 *   2) VCC stands for "Variable Coefficient on Coarse data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_1t( hypre_StructMatmultHelper *a,
                                    HYPRE_Int    ncomponents,
                                    HYPRE_Int   *indices,
                                    HYPRE_Int    ndim,
                                    hypre_Index  loop_size,
                                    hypre_Box   *gdbox,
                                    hypre_Index  gdstart,
                                    hypre_Index  gdstride,
                                    hypre_Box   *Mdbox,
                                    hypre_Index  Mdstart,
                                    hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
               HYPRE_SMMCORE_1T(indices, k + 1);
               HYPRE_SMMCORE_1T(indices, k + 2);
               HYPRE_SMMCORE_1T(indices, k + 3);
               HYPRE_SMMCORE_1T(indices, k + 4);
               HYPRE_SMMCORE_1T(indices, k + 5);
               HYPRE_SMMCORE_1T(indices, k + 6);
               HYPRE_SMMCORE_1T(indices, k + 7);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
               HYPRE_SMMCORE_1T(indices, k + 1);
               HYPRE_SMMCORE_1T(indices, k + 2);
               HYPRE_SMMCORE_1T(indices, k + 3);
               HYPRE_SMMCORE_1T(indices, k + 4);
               HYPRE_SMMCORE_1T(indices, k + 5);
               HYPRE_SMMCORE_1T(indices, k + 6);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
               HYPRE_SMMCORE_1T(indices, k + 1);
               HYPRE_SMMCORE_1T(indices, k + 2);
               HYPRE_SMMCORE_1T(indices, k + 3);
               HYPRE_SMMCORE_1T(indices, k + 4);
               HYPRE_SMMCORE_1T(indices, k + 5);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
               HYPRE_SMMCORE_1T(indices, k + 1);
               HYPRE_SMMCORE_1T(indices, k + 2);
               HYPRE_SMMCORE_1T(indices, k + 3);
               HYPRE_SMMCORE_1T(indices, k + 4);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
               HYPRE_SMMCORE_1T(indices, k + 1);
               HYPRE_SMMCORE_1T(indices, k + 2);
               HYPRE_SMMCORE_1T(indices, k + 3);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
               HYPRE_SMMCORE_1T(indices, k + 1);
               HYPRE_SMMCORE_1T(indices, k + 2);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
               HYPRE_SMMCORE_1T(indices, k + 1);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1T(indices, k + 0);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_1tb
 *
 * Core function for computing the triple product of two variable coefficients
 * living on the same data space and one constant coefficient that requires
 * the usage of a bitmask.
 *
 * "1tb" means:
 *   "1": single data space.
 *   "t": triple product.
 *   "b": single bitmask.
 *
 * This can be used for the scenarios:
 *   1) VCF * VCF * CCF.
 *   2) VCF * CCF * VCF.
 *   3) CCF * VCF * VCF.
 *
 * where:
 *   1) VCF stands for "Variable Coefficient on Fine data space".
 *   2) CCF stands for "Constant Coefficient on Fine data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_1tb( hypre_StructMatmultHelper *a,
                                     HYPRE_Int    ncomponents,
                                     HYPRE_Int   *indices,
                                     HYPRE_Int  **order,
                                     HYPRE_Int    ndim,
                                     hypre_Index  loop_size,
                                     hypre_Box   *gdbox,
                                     hypre_Index  gdstart,
                                     hypre_Index  gdstride,
                                     hypre_Box   *Mdbox,
                                     hypre_Index  Mdstart,
                                     hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
               HYPRE_SMMCORE_1TB(indices, order, k + 1);
               HYPRE_SMMCORE_1TB(indices, order, k + 2);
               HYPRE_SMMCORE_1TB(indices, order, k + 3);
               HYPRE_SMMCORE_1TB(indices, order, k + 4);
               HYPRE_SMMCORE_1TB(indices, order, k + 5);
               HYPRE_SMMCORE_1TB(indices, order, k + 6);
               HYPRE_SMMCORE_1TB(indices, order, k + 7);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
               HYPRE_SMMCORE_1TB(indices, order, k + 1);
               HYPRE_SMMCORE_1TB(indices, order, k + 2);
               HYPRE_SMMCORE_1TB(indices, order, k + 3);
               HYPRE_SMMCORE_1TB(indices, order, k + 4);
               HYPRE_SMMCORE_1TB(indices, order, k + 5);
               HYPRE_SMMCORE_1TB(indices, order, k + 6);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
               HYPRE_SMMCORE_1TB(indices, order, k + 1);
               HYPRE_SMMCORE_1TB(indices, order, k + 2);
               HYPRE_SMMCORE_1TB(indices, order, k + 3);
               HYPRE_SMMCORE_1TB(indices, order, k + 4);
               HYPRE_SMMCORE_1TB(indices, order, k + 5);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
               HYPRE_SMMCORE_1TB(indices, order, k + 1);
               HYPRE_SMMCORE_1TB(indices, order, k + 2);
               HYPRE_SMMCORE_1TB(indices, order, k + 3);
               HYPRE_SMMCORE_1TB(indices, order, k + 4);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
               HYPRE_SMMCORE_1TB(indices, order, k + 1);
               HYPRE_SMMCORE_1TB(indices, order, k + 2);
               HYPRE_SMMCORE_1TB(indices, order, k + 3);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
               HYPRE_SMMCORE_1TB(indices, order, k + 1);
               HYPRE_SMMCORE_1TB(indices, order, k + 2);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
               HYPRE_SMMCORE_1TB(indices, order, k + 1);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TB(indices, order, k + 0);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_1tbb
 *
 * Core function for computing the product of three coefficients that
 * live on the same data space. One is a variable coefficient and the other
 * two are constant coefficient that require the usage of a bitmask.
 *
 * "1tbb" means:
 *   "1" : single data space.
 *   "t" : triple product.
 *   "bb": two bitmasks.
 *
 * This can be used for the scenarios:
 *   1) VCF * CCF * CCF.
 *   2) CCF * VCF * CCF.
 *   3) CCF * CCF * VCF.
 *
 * where:
 *   1) VCF stands for "Variable Coefficient on Fine data space".
 *   2) CCF stands for "Constant Coefficient on Fine data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_1tbb( hypre_StructMatmultHelper *a,
                                      HYPRE_Int    ncomponents,
                                      HYPRE_Int   *indices,
                                      HYPRE_Int  **order,
                                      HYPRE_Int    ndim,
                                      hypre_Index  loop_size,
                                      hypre_Box   *gdbox,
                                      hypre_Index  gdstart,
                                      hypre_Index  gdstride,
                                      hypre_Box   *Mdbox,
                                      hypre_Index  Mdstart,
                                      hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
               HYPRE_SMMCORE_1TBB(indices, order, k + 1);
               HYPRE_SMMCORE_1TBB(indices, order, k + 2);
               HYPRE_SMMCORE_1TBB(indices, order, k + 3);
               HYPRE_SMMCORE_1TBB(indices, order, k + 4);
               HYPRE_SMMCORE_1TBB(indices, order, k + 5);
               HYPRE_SMMCORE_1TBB(indices, order, k + 6);
               HYPRE_SMMCORE_1TBB(indices, order, k + 7);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
               HYPRE_SMMCORE_1TBB(indices, order, k + 1);
               HYPRE_SMMCORE_1TBB(indices, order, k + 2);
               HYPRE_SMMCORE_1TBB(indices, order, k + 3);
               HYPRE_SMMCORE_1TBB(indices, order, k + 4);
               HYPRE_SMMCORE_1TBB(indices, order, k + 5);
               HYPRE_SMMCORE_1TBB(indices, order, k + 6);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
               HYPRE_SMMCORE_1TBB(indices, order, k + 1);
               HYPRE_SMMCORE_1TBB(indices, order, k + 2);
               HYPRE_SMMCORE_1TBB(indices, order, k + 3);
               HYPRE_SMMCORE_1TBB(indices, order, k + 4);
               HYPRE_SMMCORE_1TBB(indices, order, k + 5);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
               HYPRE_SMMCORE_1TBB(indices, order, k + 1);
               HYPRE_SMMCORE_1TBB(indices, order, k + 2);
               HYPRE_SMMCORE_1TBB(indices, order, k + 3);
               HYPRE_SMMCORE_1TBB(indices, order, k + 4);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
               HYPRE_SMMCORE_1TBB(indices, order, k + 1);
               HYPRE_SMMCORE_1TBB(indices, order, k + 2);
               HYPRE_SMMCORE_1TBB(indices, order, k + 3);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
               HYPRE_SMMCORE_1TBB(indices, order, k + 1);
               HYPRE_SMMCORE_1TBB(indices, order, k + 2);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
               HYPRE_SMMCORE_1TBB(indices, order, k + 1);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBB(indices, order, k + 0);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

#undef HYPRE_KERNEL
#define HYPRE_KERNEL(k) cprod[k]* \
                        tptrs[0][k][gi]* \
                        ((((HYPRE_Int) tptrs[1][k][gi]) >> bitshift[1][k]) & 1)*\
                        ((((HYPRE_Int) tptrs[2][k][gi]) >> bitshift[2][k]) & 1)

HYPRE_Int
hypre_StructMatmultCompute_core_1tbb_v2( hypre_StructMatmultHelper *a,
                                     HYPRE_Int    ncomponents,
                                     HYPRE_Int   *indices,
                                     HYPRE_Int  **order,
                                     HYPRE_Int    ndim,
                                     hypre_Index  loop_size,
                                     HYPRE_Int    stencil_size,
                                     hypre_Box   *gdbox,
                                     hypre_Index  gdstart,
                                     hypre_Index  gdstride,
                                     hypre_Box   *Mdbox,
                                     hypre_Index  Mdstart,
                                     hypre_Index  Mdstride )

{
   HYPRE_ANNOTATE_FUNC_BEGIN;

   HYPRE_Int  mentry;
   HYPRE_Int  count;
   HYPRE_Int  e, k, kk;
   HYPRE_Int  depth;
   HYPRE_Int **bitshift;
   HYPRE_Complex *cprod;
   HYPRE_Complex *mptr;
   const HYPRE_Complex ***tptrs;

   /* Allocate memory */
   bitshift = hypre_CTAlloc(HYPRE_Int *, 3, HYPRE_MEMORY_HOST);
   cprod = hypre_CTAlloc(HYPRE_Complex, ncomponents, HYPRE_MEMORY_HOST);
   tptrs = hypre_CTAlloc(const HYPRE_Complex**, 3, HYPRE_MEMORY_HOST);
   for (kk = 0; kk < 3; kk++)
   {
      bitshift[kk] = hypre_CTAlloc(HYPRE_Int, ncomponents, HYPRE_MEMORY_HOST);
      tptrs[kk] = hypre_CTAlloc(const HYPRE_Complex*, ncomponents, HYPRE_MEMORY_HOST);
   }

   for (e = 0; e < stencil_size; e++)
   {
      count = 0;
      for (k = 0; k < ncomponents; k++)
      {
         mentry = a[indices[k]].mentry;
         if (mentry == e)
         {
            for (kk = 0; kk < 3; kk++)
            {
               tptrs[kk][count] = a[indices[k]].tptrs[order[k][kk]];
               bitshift[kk][count] = order[k][kk];
            }
            cprod[count] = a[indices[k]].cprod;
            mptr = a[indices[k]].mptr;
            count++;
         }
      }

      for (k = 0; k < count; k += UNROLL_MAXDEPTH)
      {
         depth = hypre_min(UNROLL_MAXDEPTH, (count - k));

         switch (depth)
         {
            case 7:
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi);
               {
                  HYPRE_Complex val = HYPRE_KERNEL(k + 0) +
                                      HYPRE_KERNEL(k + 1) +
                                      HYPRE_KERNEL(k + 2) +
                                      HYPRE_KERNEL(k + 3) +
                                      HYPRE_KERNEL(k + 4) +
                                      HYPRE_KERNEL(k + 5) +
                                      HYPRE_KERNEL(k + 6);
                  mptr[Mi] += val;
               }
               hypre_BoxLoop2End(Mi,gi);
               break;

            case 6:
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi);
               {
                  HYPRE_Complex val = HYPRE_KERNEL(k + 0) +
                                      HYPRE_KERNEL(k + 1) +
                                      HYPRE_KERNEL(k + 2) +
                                      HYPRE_KERNEL(k + 3) +
                                      HYPRE_KERNEL(k + 4) +
                                      HYPRE_KERNEL(k + 5);
                  mptr[Mi] += val;
               }
               hypre_BoxLoop2End(Mi,gi);
               break;

            case 5:
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi);
               {
                  HYPRE_Complex val = HYPRE_KERNEL(k + 0) +
                                      HYPRE_KERNEL(k + 1) +
                                      HYPRE_KERNEL(k + 2) +
                                      HYPRE_KERNEL(k + 3) +
                                      HYPRE_KERNEL(k + 4);
                  mptr[Mi] += val;
               }
               hypre_BoxLoop2End(Mi,gi);
               break;

            case 4:
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi);
               {
                  /* HYPRE_Complex val = HYPRE_KERNEL(k + 0) + */
                  /*                     HYPRE_KERNEL(k + 1) + */
                  /*                     HYPRE_KERNEL(k + 2) + */
                  /*                     HYPRE_KERNEL(k + 3); */
                  /* mptr[Mi] += val; */

                  mptr[Mi] += HYPRE_KERNEL(k + 0) + HYPRE_KERNEL(k + 1) + HYPRE_KERNEL(k + 2) + HYPRE_KERNEL(k + 3);
               }
               hypre_BoxLoop2End(Mi,gi);
               break;

            case 3:
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi);
               {
                  /* HYPRE_Complex val = HYPRE_KERNEL(k + 0) + */
                  /*                     HYPRE_KERNEL(k + 1) + */
                  /*                     HYPRE_KERNEL(k + 2); */
                  /* mptr[Mi] += val; */

                  mptr[Mi] += HYPRE_KERNEL(k + 0) + HYPRE_KERNEL(k + 1) + HYPRE_KERNEL(k + 2);
               }
               hypre_BoxLoop2End(Mi,gi);
               break;

            case 2:
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi);
               {
                  /* HYPRE_Complex val = HYPRE_KERNEL(k + 0) + */
                  /*                     HYPRE_KERNEL(k + 1); */
                  /* mptr[Mi] += val; */

                  mptr[Mi] += HYPRE_KERNEL(k + 0) + HYPRE_KERNEL(k + 1);
               }
               hypre_BoxLoop2End(Mi,gi);
               break;

            case 1:
               hypre_BoxLoop2Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi);
               {
                  /* HYPRE_Complex val = HYPRE_KERNEL(k + 0); */

                  /* mptr[Mi] += val; */

                  mptr[Mi] += HYPRE_KERNEL(k + 0);
               }
               hypre_BoxLoop2End(Mi,gi);
               break;

             default:
               break;
         }
      }
   }

   /* Free memory */
   hypre_TFree(bitshift, HYPRE_MEMORY_HOST);
   hypre_TFree(cprod, HYPRE_MEMORY_HOST);
   for (k = 0; k < 3; k++)
   {
      hypre_TFree(tptrs[k], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(tptrs, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_1tbbb
 *
 * Core function for computing the product of three constant coefficients that
 * live on the same data space and that require the usage of a bitmask.
 *
 * "1tbb" means:
 *   "1" : single data space.
 *   "t" : triple product.
 *   "bbb": three bitmasks.
 *
 * This can be used for the scenario:
 *   1) CCF * CCF * CCF.
 *
 * where:
 *   1) CCF stands for "Constant Coefficient on Fine data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_1tbbb( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
               HYPRE_SMMCORE_1TBBB(indices, k + 1);
               HYPRE_SMMCORE_1TBBB(indices, k + 2);
               HYPRE_SMMCORE_1TBBB(indices, k + 3);
               HYPRE_SMMCORE_1TBBB(indices, k + 4);
               HYPRE_SMMCORE_1TBBB(indices, k + 5);
               HYPRE_SMMCORE_1TBBB(indices, k + 6);
               HYPRE_SMMCORE_1TBBB(indices, k + 7);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
               HYPRE_SMMCORE_1TBBB(indices, k + 1);
               HYPRE_SMMCORE_1TBBB(indices, k + 2);
               HYPRE_SMMCORE_1TBBB(indices, k + 3);
               HYPRE_SMMCORE_1TBBB(indices, k + 4);
               HYPRE_SMMCORE_1TBBB(indices, k + 5);
               HYPRE_SMMCORE_1TBBB(indices, k + 6);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
               HYPRE_SMMCORE_1TBBB(indices, k + 1);
               HYPRE_SMMCORE_1TBBB(indices, k + 2);
               HYPRE_SMMCORE_1TBBB(indices, k + 3);
               HYPRE_SMMCORE_1TBBB(indices, k + 4);
               HYPRE_SMMCORE_1TBBB(indices, k + 5);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
               HYPRE_SMMCORE_1TBBB(indices, k + 1);
               HYPRE_SMMCORE_1TBBB(indices, k + 2);
               HYPRE_SMMCORE_1TBBB(indices, k + 3);
               HYPRE_SMMCORE_1TBBB(indices, k + 4);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
               HYPRE_SMMCORE_1TBBB(indices, k + 1);
               HYPRE_SMMCORE_1TBBB(indices, k + 2);
               HYPRE_SMMCORE_1TBBB(indices, k + 3);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
               HYPRE_SMMCORE_1TBBB(indices, k + 1);
               HYPRE_SMMCORE_1TBBB(indices, k + 2);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
               HYPRE_SMMCORE_1TBBB(indices, k + 1);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi);
            {
               HYPRE_SMMCORE_1TBBB(indices, k + 0);
            }
            hypre_BoxLoop2End(Mi,gi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_2t
 *
 * Core function for computing the triple product of variable coefficients
 * in which two of them live on the same data space "g" and the other lives
 * on data space "h"
 *
 * "2t" means:
 *   "2": two data spaces.
 *   "t": triple product.
 *
 * This can be used for the scenarios:
 *   1) VCF * VCF * VCC.
 *   2) VCF * VCC * VCF.
 *   3) VCC * VCF * VCF.
 *   4) VCC * VCC * VCF.
 *   5) VCC * VCF * VCC.
 *   6) VCF * VCC * VCC.
 *
 * where:
 *   1) VCF stands for "Variable Coefficient on Fine data space".
 *   2) VCC stands for "Variable Coefficient on Coarse data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t( hypre_StructMatmultHelper *a,
                                    HYPRE_Int    ncomponents,
                                    HYPRE_Int   *indices,
                                    HYPRE_Int  **order,
                                    HYPRE_Int    ndim,
                                    hypre_Index  loop_size,
                                    hypre_Box   *gdbox,
                                    hypre_Index  gdstart,
                                    hypre_Index  gdstride,
                                    hypre_Box   *hdbox,
                                    hypre_Index  hdstart,
                                    hypre_Index  hdstride,
                                    hypre_Box   *Mdbox,
                                    hypre_Index  Mdstart,
                                    hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
               HYPRE_SMMCORE_2T(indices, order, k + 1);
               HYPRE_SMMCORE_2T(indices, order, k + 2);
               HYPRE_SMMCORE_2T(indices, order, k + 3);
               HYPRE_SMMCORE_2T(indices, order, k + 4);
               HYPRE_SMMCORE_2T(indices, order, k + 5);
               HYPRE_SMMCORE_2T(indices, order, k + 6);
               HYPRE_SMMCORE_2T(indices, order, k + 7);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
               HYPRE_SMMCORE_2T(indices, order, k + 1);
               HYPRE_SMMCORE_2T(indices, order, k + 2);
               HYPRE_SMMCORE_2T(indices, order, k + 3);
               HYPRE_SMMCORE_2T(indices, order, k + 4);
               HYPRE_SMMCORE_2T(indices, order, k + 5);
               HYPRE_SMMCORE_2T(indices, order, k + 6);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
               HYPRE_SMMCORE_2T(indices, order, k + 1);
               HYPRE_SMMCORE_2T(indices, order, k + 2);
               HYPRE_SMMCORE_2T(indices, order, k + 3);
               HYPRE_SMMCORE_2T(indices, order, k + 4);
               HYPRE_SMMCORE_2T(indices, order, k + 5);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
               HYPRE_SMMCORE_2T(indices, order, k + 1);
               HYPRE_SMMCORE_2T(indices, order, k + 2);
               HYPRE_SMMCORE_2T(indices, order, k + 3);
               HYPRE_SMMCORE_2T(indices, order, k + 4);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
               HYPRE_SMMCORE_2T(indices, order, k + 1);
               HYPRE_SMMCORE_2T(indices, order, k + 2);
               HYPRE_SMMCORE_2T(indices, order, k + 3);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
               HYPRE_SMMCORE_2T(indices, order, k + 1);
               HYPRE_SMMCORE_2T(indices, order, k + 2);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
               HYPRE_SMMCORE_2T(indices, order, k + 1);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T(indices, order, k + 0);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_2t_v2
 *
 * Core function for computing the triple product of variable coefficients
 * in which two of them live on the same data space "g" and the other lives
 * on data space "h"
 *
 * "2t" means:
 *   "2": two data spaces.
 *   "t": triple product.
 *
 * This can be used for the scenarios:
 *   1) VCF * VCF * VCC.
 *   2) VCF * VCC * VCF.
 *   3) VCC * VCF * VCF.
 *   4) VCC * VCC * VCF.
 *   5) VCC * VCF * VCC.
 *   6) VCF * VCC * VCC.
 *
 * where:
 *   1) VCF stands for "Variable Coefficient on Fine data space".
 *   2) VCC stands for "Variable Coefficient on Coarse data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v2( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Int   *terms,
                                       HYPRE_Complex **dptrs )
{
   HYPRE_Int    k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_StructMatmultHelper *ak;
   HYPRE_Real cprod[512], cprodk;
   HYPRE_Int  o0[512], o0k;
   HYPRE_Int  o1[512], o1k;
   HYPRE_Int  o2[512], o2k;

   for (k = 0; k < ncomponents; k++)
   {
      cprod[k] = a[indices[k]].cprod;
      o0[k] = a[indices[k]].offsets[0];
      o1[k] = a[indices[k]].offsets[1];
      o2[k] = a[indices[k]].offsets[2];
   }

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 33:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 26);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 27);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 28);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 29);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 30);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 31);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 32);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 32:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 26);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 27);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 28);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 29);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 30);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 31);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 31:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 26);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 27);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 28);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 29);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 30);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 30:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 26);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 27);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 28);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 29);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 29:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 26);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 27);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 28);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 28:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 26);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 27);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 27:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 26);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 26:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 25);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 25:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 24);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 24:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 23);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 23:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 22);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 22:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 21);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 21:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 20);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 20:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 19);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 19:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 18);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 18:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 17);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 17:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 16);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 16:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 15);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 15:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 14);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 14:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 13);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 13:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 12);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 12:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 11);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 11:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 10);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 10:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 9);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 8);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 7);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 6);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 5);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 4);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 3);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 2);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 1);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 1:
#if 0
            ak = &a[k];
            cprodk = cprod[k];
            o0k = o0[k]; o1k = o1[k]; o2k = o2[k];
#endif
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
              HYPRE_SMMCORE_2T_V2(cprod, o0, o1, o2, k + 0);
               //HYPRE_SMMCORE_2T_V2B(ak, cprodk, o0k, o1k, o2k);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master with "+=" sign */

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v3a( hypre_StructMatmultHelper *a,
                                        HYPRE_Int    ncomponents,
                                        HYPRE_Int   *indices,
                                        HYPRE_Int    ndim,
                                        hypre_Index  loop_size,
                                        hypre_Box   *gdbox,
                                        hypre_Index  gdstart,
                                        hypre_Index  gdstride,
                                        hypre_Box   *hdbox,
                                        hypre_Index  hdstart,
                                        hypre_Index  hdstride,
                                        hypre_Box   *Mdbox,
                                        hypre_Index  Mdstart,
                                        hypre_Index  Mdstride,
                                        HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   const HYPRE_Complex *dptrs0 = dptrs[0];
   const HYPRE_Complex *dptrs1 = dptrs[1];
   const HYPRE_Complex *dptrs2 = dptrs[2];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[0].mptr[Mi]  += a[0].cprod*dptrs0[o0[0] + gi]*dptrs2[o2[0] + gi]*dptrs1[o1[0] + hi] +
                        a[1].cprod*dptrs0[o0[1] + gi]*dptrs2[o2[1] + gi]*dptrs1[o1[1] + hi] +
                        a[2].cprod*dptrs0[o0[2] + gi]*dptrs2[o2[2] + gi]*dptrs1[o1[2] + hi] +
                        a[3].cprod*dptrs0[o0[3] + gi]*dptrs2[o2[3] + gi]*dptrs1[o1[3] + hi] +
                        a[4].cprod*dptrs0[o0[4] + gi]*dptrs2[o2[4] + gi]*dptrs1[o1[4] + hi] +
                        a[5].cprod*dptrs0[o0[5] + gi]*dptrs2[o2[5] + gi]*dptrs1[o1[5] + hi] +
                        a[6].cprod*dptrs0[o0[6] + gi]*dptrs2[o2[6] + gi]*dptrs1[o1[6] + hi];

      a[7].mptr[Mi]  += a[7].cprod*dptrs0[o0[7] + gi]*dptrs2[o2[7] + gi]*dptrs1[o1[7] + hi] +
                        a[8].cprod*dptrs0[o0[8] + gi]*dptrs2[o2[8] + gi]*dptrs1[o1[8] + hi] +
                        a[9].cprod*dptrs0[o0[9] + gi]*dptrs2[o2[9] + gi]*dptrs1[o1[9] + hi];

      a[10].mptr[Mi] += a[10].cprod*dptrs0[o0[10] + gi]*dptrs2[o2[10] + gi]*dptrs1[o1[10] + hi] +
                        a[11].cprod*dptrs0[o0[11] + gi]*dptrs2[o2[11] + gi]*dptrs1[o1[11] + hi] +
                        a[12].cprod*dptrs0[o0[12] + gi]*dptrs2[o2[12] + gi]*dptrs1[o1[12] + hi];

      a[13].mptr[Mi] += a[13].cprod*dptrs0[o0[13] + gi]*dptrs2[o2[13] + gi]*dptrs1[o1[13] + hi] +
                        a[14].cprod*dptrs0[o0[14] + gi]*dptrs2[o2[14] + gi]*dptrs1[o1[14] + hi] +
                        a[15].cprod*dptrs0[o0[15] + gi]*dptrs2[o2[15] + gi]*dptrs1[o1[15] + hi];

      a[16].mptr[Mi] += a[16].cprod*dptrs0[o0[16] + gi]*dptrs2[o2[16] + gi]*dptrs1[o1[16] + hi] +
                        a[17].cprod*dptrs0[o0[17] + gi]*dptrs2[o2[17] + gi]*dptrs1[o1[17] + hi] +
                        a[18].cprod*dptrs0[o0[18] + gi]*dptrs2[o2[18] + gi]*dptrs1[o1[18] + hi];

      a[19].mptr[Mi] += a[19].cprod*dptrs0[o0[19] + gi]*dptrs2[o2[19] + gi]*dptrs1[o1[19] + hi] +
                        a[20].cprod*dptrs0[o0[20] + gi]*dptrs2[o2[20] + gi]*dptrs1[o1[20] + hi] +
                        a[21].cprod*dptrs0[o0[21] + gi]*dptrs2[o2[21] + gi]*dptrs1[o1[21] + hi];

      a[22].mptr[Mi] += a[22].cprod*dptrs0[o0[22] + gi]*dptrs2[o2[22] + gi]*dptrs1[o1[22] + hi] +
                        a[23].cprod*dptrs0[o0[23] + gi]*dptrs2[o2[23] + gi]*dptrs1[o1[23] + hi] +
                        a[24].cprod*dptrs0[o0[24] + gi]*dptrs2[o2[24] + gi]*dptrs1[o1[24] + hi];

      a[25].mptr[Mi] += a[25].cprod*dptrs0[o0[25] + gi]*dptrs2[o2[25] + gi]*dptrs1[o1[25] + hi];

      a[26].mptr[Mi] += a[26].cprod*dptrs0[o0[26] + gi]*dptrs2[o2[26] + gi]*dptrs1[o1[26] + hi];

      a[27].mptr[Mi] += a[27].cprod*dptrs0[o0[27] + gi]*dptrs2[o2[27] + gi]*dptrs1[o1[27] + hi];

      a[28].mptr[Mi] += a[28].cprod*dptrs0[o0[28] + gi]*dptrs2[o2[28] + gi]*dptrs1[o1[28] + hi];

      a[29].mptr[Mi] += a[29].cprod*dptrs0[o0[29] + gi]*dptrs2[o2[29] + gi]*dptrs1[o1[29] + hi];

      a[30].mptr[Mi] += a[30].cprod*dptrs0[o0[30] + gi]*dptrs2[o2[30] + gi]*dptrs1[o1[30] + hi];

      a[31].mptr[Mi] += a[31].cprod*dptrs0[o0[31] + gi]*dptrs2[o2[31] + gi]*dptrs1[o1[31] + hi];

      a[32].mptr[Mi] += a[32].cprod*dptrs0[o0[32] + gi]*dptrs2[o2[32] + gi]*dptrs1[o1[32] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master without "+=" sign */

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v3b( hypre_StructMatmultHelper *a,
                                        HYPRE_Int    ncomponents,
                                        HYPRE_Int   *indices,
                                        HYPRE_Int    ndim,
                                        hypre_Index  loop_size,
                                        hypre_Box   *gdbox,
                                        hypre_Index  gdstart,
                                        hypre_Index  gdstride,
                                        hypre_Box   *hdbox,
                                        hypre_Index  hdstart,
                                        hypre_Index  hdstride,
                                        hypre_Box   *Mdbox,
                                        hypre_Index  Mdstart,
                                        hypre_Index  Mdstride,
                                        HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   const HYPRE_Complex *dptrs0 = dptrs[0];
   const HYPRE_Complex *dptrs1 = dptrs[1];
   const HYPRE_Complex *dptrs2 = dptrs[2];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[0].mptr[Mi]  = a[0].cprod*dptrs0[o0[0] + gi]*dptrs2[o2[0] + gi]*dptrs1[o1[0] + hi] +
                       a[1].cprod*dptrs0[o0[1] + gi]*dptrs2[o2[1] + gi]*dptrs1[o1[1] + hi] +
                       a[2].cprod*dptrs0[o0[2] + gi]*dptrs2[o2[2] + gi]*dptrs1[o1[2] + hi] +
                       a[3].cprod*dptrs0[o0[3] + gi]*dptrs2[o2[3] + gi]*dptrs1[o1[3] + hi] +
                       a[4].cprod*dptrs0[o0[4] + gi]*dptrs2[o2[4] + gi]*dptrs1[o1[4] + hi] +
                       a[5].cprod*dptrs0[o0[5] + gi]*dptrs2[o2[5] + gi]*dptrs1[o1[5] + hi] +
                       a[6].cprod*dptrs0[o0[6] + gi]*dptrs2[o2[6] + gi]*dptrs1[o1[6] + hi];

      a[7].mptr[Mi]  = a[7].cprod*dptrs0[o0[7] + gi]*dptrs2[o2[7] + gi]*dptrs1[o1[7] + hi] +
                       a[8].cprod*dptrs0[o0[8] + gi]*dptrs2[o2[8] + gi]*dptrs1[o1[8] + hi] +
                       a[9].cprod*dptrs0[o0[9] + gi]*dptrs2[o2[9] + gi]*dptrs1[o1[9] + hi];

      a[10].mptr[Mi] = a[10].cprod*dptrs0[o0[10] + gi]*dptrs2[o2[10] + gi]*dptrs1[o1[10] + hi] +
                       a[11].cprod*dptrs0[o0[11] + gi]*dptrs2[o2[11] + gi]*dptrs1[o1[11] + hi] +
                       a[12].cprod*dptrs0[o0[12] + gi]*dptrs2[o2[12] + gi]*dptrs1[o1[12] + hi];

      a[13].mptr[Mi] = a[13].cprod*dptrs0[o0[13] + gi]*dptrs2[o2[13] + gi]*dptrs1[o1[13] + hi] +
                       a[14].cprod*dptrs0[o0[14] + gi]*dptrs2[o2[14] + gi]*dptrs1[o1[14] + hi] +
                       a[15].cprod*dptrs0[o0[15] + gi]*dptrs2[o2[15] + gi]*dptrs1[o1[15] + hi];

      a[16].mptr[Mi] = a[16].cprod*dptrs0[o0[16] + gi]*dptrs2[o2[16] + gi]*dptrs1[o1[16] + hi] +
                       a[17].cprod*dptrs0[o0[17] + gi]*dptrs2[o2[17] + gi]*dptrs1[o1[17] + hi] +
                       a[18].cprod*dptrs0[o0[18] + gi]*dptrs2[o2[18] + gi]*dptrs1[o1[18] + hi];

      a[19].mptr[Mi] = a[19].cprod*dptrs0[o0[19] + gi]*dptrs2[o2[19] + gi]*dptrs1[o1[19] + hi] +
                       a[20].cprod*dptrs0[o0[20] + gi]*dptrs2[o2[20] + gi]*dptrs1[o1[20] + hi] +
                       a[21].cprod*dptrs0[o0[21] + gi]*dptrs2[o2[21] + gi]*dptrs1[o1[21] + hi];

      a[22].mptr[Mi] = a[22].cprod*dptrs0[o0[22] + gi]*dptrs2[o2[22] + gi]*dptrs1[o1[22] + hi] +
                       a[23].cprod*dptrs0[o0[23] + gi]*dptrs2[o2[23] + gi]*dptrs1[o1[23] + hi] +
                       a[24].cprod*dptrs0[o0[24] + gi]*dptrs2[o2[24] + gi]*dptrs1[o1[24] + hi];

      a[25].mptr[Mi] = a[25].cprod*dptrs0[o0[25] + gi]*dptrs2[o2[25] + gi]*dptrs1[o1[25] + hi];

      a[26].mptr[Mi] = a[26].cprod*dptrs0[o0[26] + gi]*dptrs2[o2[26] + gi]*dptrs1[o1[26] + hi];

      a[27].mptr[Mi] = a[27].cprod*dptrs0[o0[27] + gi]*dptrs2[o2[27] + gi]*dptrs1[o1[27] + hi];

      a[28].mptr[Mi] = a[28].cprod*dptrs0[o0[28] + gi]*dptrs2[o2[28] + gi]*dptrs1[o1[28] + hi];

      a[29].mptr[Mi] = a[29].cprod*dptrs0[o0[29] + gi]*dptrs2[o2[29] + gi]*dptrs1[o1[29] + hi];

      a[30].mptr[Mi] = a[30].cprod*dptrs0[o0[30] + gi]*dptrs2[o2[30] + gi]*dptrs1[o1[30] + hi];

      a[31].mptr[Mi] = a[31].cprod*dptrs0[o0[31] + gi]*dptrs2[o2[31] + gi]*dptrs1[o1[31] + hi];

      a[32].mptr[Mi] = a[32].cprod*dptrs0[o0[32] + gi]*dptrs2[o2[32] + gi]*dptrs1[o1[32] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master with "+=" sign and simplifies
   terms that are equal to one */

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v4( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   const HYPRE_Complex *dptrs0 = dptrs[0];
   const HYPRE_Complex *dptrs1 = dptrs[1];
   const HYPRE_Complex *dptrs2 = dptrs[2];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   const HYPRE_Complex *dptrs0_0 = dptrs0 + o0[0];
   const HYPRE_Complex *dptrs0_1 = dptrs0 + o0[2];
   const HYPRE_Complex *dptrs0_2 = dptrs0 + o0[7];
   const HYPRE_Complex *dptrs0_3 = dptrs0 + o0[10];

   const HYPRE_Complex *dptrs1_0 = dptrs1 + o1[0];
   const HYPRE_Complex *dptrs1_1 = dptrs1 + o1[1];
   const HYPRE_Complex *dptrs1_2 = dptrs1 + o1[2];

   const HYPRE_Complex *dptrs2_0 = dptrs2 + o2[0];
   const HYPRE_Complex *dptrs2_1 = dptrs2 + o2[2];

   HYPRE_Complex *mptrs_1 = a[7].mptr;

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      mptrs_1[Mi]  = dptrs0_2[gi]*      dptrs2_0[gi]      *dptrs1_0[hi] +
                     dptrs0_2[gi]                         *dptrs1_1[hi] +
                                        dptrs2_0[gi]      *dptrs1[o1[9] + hi];

      a[10].mptr[Mi] = dptrs0_3[gi]*      dptrs2_1[gi]        *dptrs1_2[hi] +
                       dptrs0_3[gi]                           *dptrs1[o1[11] + hi] +
                                          dptrs2_1[gi]        *dptrs1[o1[12] + hi];

      a[13].mptr[Mi] = dptrs0[o0[13] + gi]*dptrs2_0[gi]       *dptrs1[o1[13] + hi] +
                       dptrs0[o0[14] + gi]*dptrs2_1[gi]       *dptrs1[o1[14] + hi] +
                                                               dptrs1[o1[15] + hi];

      a[16].mptr[Mi] = dptrs0[o0[16] + gi]*dptrs2_0[gi]       *dptrs1[o1[16] + hi] +
                       dptrs0[o0[17] + gi]*dptrs2_1[gi]       *dptrs1[o1[17] + hi] +
                                                               dptrs1[o1[18] + hi];

      a[19].mptr[Mi] = dptrs0[o0[19] + gi]*dptrs2_0[gi]       *dptrs1[o1[19] + hi] +
                       dptrs0[o0[20] + gi]*dptrs2_1[gi]       *dptrs1[o1[20] + hi] +
                                                               dptrs1[o1[21] + hi];

      a[22].mptr[Mi] = dptrs0[o0[22] + gi]*dptrs2_0[gi]       *dptrs1[o1[22] + hi] +
                       dptrs0[o0[23] + gi]*dptrs2_1[gi]       *dptrs1[o1[23] + hi] +
                                                               dptrs1[o1[24] + hi];

      a[25].mptr[Mi] = dptrs0[o0[25] + gi]*dptrs2_0[gi]       *dptrs1[o1[25] + hi];

      a[26].mptr[Mi] = dptrs0[o0[26] + gi]*dptrs2_0[gi]       *dptrs1[o1[26] + hi];

      a[27].mptr[Mi] = dptrs0[o0[27] + gi]*dptrs2_0[gi]       *dptrs1[o1[27] + hi];

      a[28].mptr[Mi] = dptrs0[o0[28] + gi]*dptrs2_0[gi]       *dptrs1[o1[28] + hi];

      a[29].mptr[Mi] = dptrs0[o0[29] + gi]*dptrs2_1[gi]       *dptrs1[o1[29] + hi];

      a[30].mptr[Mi] = dptrs0[o0[30] + gi]*dptrs2_1[gi]       *dptrs1[o1[30] + hi];

      a[31].mptr[Mi] = dptrs0[o0[31] + gi]*dptrs2_1[gi]       *dptrs1[o1[31] + hi];

      a[32].mptr[Mi] = dptrs0[o0[32] + gi]*dptrs2_1[gi]       *dptrs1[o1[32] + hi];

      a[0].mptr[Mi]  = dptrs0_0[gi]*      dptrs2_0[gi]      *dptrs1_0[hi] +
                       dptrs0_0[gi]                         *dptrs1_1[hi] +
                       dptrs0_1[gi]*      dptrs2_1[gi]      *dptrs1_2[hi] +
                       dptrs0_1[gi]                         *dptrs1[o1[3] + hi] +
                                                             dptrs1[o1[4] + hi] +
                                          dptrs2_1[gi]      *dptrs1[o1[5] + hi] +
                                          dptrs2_0[gi]      *dptrs1[o1[6] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master without the "+=" sign but using as
   many boxloops as the number of stencil entries in M */

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v5( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   const HYPRE_Complex *dptrs0 = dptrs[0];
   const HYPRE_Complex *dptrs1 = dptrs[1];
   const HYPRE_Complex *dptrs2 = dptrs[2];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[0].mptr[Mi]  = a[0].cprod*dptrs0[o0[0] + gi]*dptrs2[o2[0] + gi]*dptrs1[o1[0] + hi] +
                       a[1].cprod*dptrs0[o0[1] + gi]*dptrs2[o2[1] + gi]*dptrs1[o1[1] + hi] +
                       a[2].cprod*dptrs0[o0[2] + gi]*dptrs2[o2[2] + gi]*dptrs1[o1[2] + hi] +
                       a[3].cprod*dptrs0[o0[3] + gi]*dptrs2[o2[3] + gi]*dptrs1[o1[3] + hi] +
                       a[4].cprod*dptrs0[o0[4] + gi]*dptrs2[o2[4] + gi]*dptrs1[o1[4] + hi] +
                       a[5].cprod*dptrs0[o0[5] + gi]*dptrs2[o2[5] + gi]*dptrs1[o1[5] + hi] +
                       a[6].cprod*dptrs0[o0[6] + gi]*dptrs2[o2[6] + gi]*dptrs1[o1[6] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      a[7].mptr[Mi]  = a[7].cprod*dptrs0[o0[7] + gi]*dptrs2[o2[7] + gi]*dptrs1[o1[7] + hi] +
                       a[8].cprod*dptrs0[o0[8] + gi]*dptrs2[o2[8] + gi]*dptrs1[o1[8] + hi] +
                       a[9].cprod*dptrs0[o0[9] + gi]*dptrs2[o2[9] + gi]*dptrs1[o1[9] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      a[10].mptr[Mi] = a[10].cprod*dptrs0[o0[10] + gi]*dptrs2[o2[10] + gi]*dptrs1[o1[10] + hi] +
                       a[11].cprod*dptrs0[o0[11] + gi]*dptrs2[o2[11] + gi]*dptrs1[o1[11] + hi] +
                       a[12].cprod*dptrs0[o0[12] + gi]*dptrs2[o2[12] + gi]*dptrs1[o1[12] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      a[13].mptr[Mi] = a[13].cprod*dptrs0[o0[13] + gi]*dptrs2[o2[13] + gi]*dptrs1[o1[13] + hi] +
                       a[14].cprod*dptrs0[o0[14] + gi]*dptrs2[o2[14] + gi]*dptrs1[o1[14] + hi] +
                       a[15].cprod*dptrs0[o0[15] + gi]*dptrs2[o2[15] + gi]*dptrs1[o1[15] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      a[16].mptr[Mi] = a[16].cprod*dptrs0[o0[16] + gi]*dptrs2[o2[16] + gi]*dptrs1[o1[16] + hi] +
                       a[17].cprod*dptrs0[o0[17] + gi]*dptrs2[o2[17] + gi]*dptrs1[o1[17] + hi] +
                       a[18].cprod*dptrs0[o0[18] + gi]*dptrs2[o2[18] + gi]*dptrs1[o1[18] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      a[19].mptr[Mi] = a[19].cprod*dptrs0[o0[19] + gi]*dptrs2[o2[19] + gi]*dptrs1[o1[19] + hi] +
                       a[20].cprod*dptrs0[o0[20] + gi]*dptrs2[o2[20] + gi]*dptrs1[o1[20] + hi] +
                       a[21].cprod*dptrs0[o0[21] + gi]*dptrs2[o2[21] + gi]*dptrs1[o1[21] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      a[22].mptr[Mi] = a[22].cprod*dptrs0[o0[22] + gi]*dptrs2[o2[22] + gi]*dptrs1[o1[22] + hi] +
                       a[23].cprod*dptrs0[o0[23] + gi]*dptrs2[o2[23] + gi]*dptrs1[o1[23] + hi] +
                       a[24].cprod*dptrs0[o0[24] + gi]*dptrs2[o2[24] + gi]*dptrs1[o1[24] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      a[25].mptr[Mi] = a[25].cprod*dptrs0[o0[25] + gi]*dptrs2[o2[25] + gi]*dptrs1[o1[25] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[26].mptr[Mi] = a[26].cprod*dptrs0[o0[26] + gi]*dptrs2[o2[26] + gi]*dptrs1[o1[26] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[27].mptr[Mi] = a[27].cprod*dptrs0[o0[27] + gi]*dptrs2[o2[27] + gi]*dptrs1[o1[27] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[28].mptr[Mi] = a[28].cprod*dptrs0[o0[28] + gi]*dptrs2[o2[28] + gi]*dptrs1[o1[28] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[29].mptr[Mi] = a[29].cprod*dptrs0[o0[29] + gi]*dptrs2[o2[29] + gi]*dptrs1[o1[29] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[30].mptr[Mi] = a[30].cprod*dptrs0[o0[30] + gi]*dptrs2[o2[30] + gi]*dptrs1[o1[30] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[31].mptr[Mi] = a[31].cprod*dptrs0[o0[31] + gi]*dptrs2[o2[31] + gi]*dptrs1[o1[31] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      a[32].mptr[Mi] = a[32].cprod*dptrs0[o0[32] + gi]*dptrs2[o2[32] + gi]*dptrs1[o1[32] + hi];
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master
     1) without the "+=" sign
     2) as many boxloops as the number of stencil entries in M.
     3) Uses a temporary value to set the resulting stencil coefficient
*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v5a( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   HYPRE_Complex val;
   const HYPRE_Complex *dptrs0 = dptrs[0];
   const HYPRE_Complex *dptrs1 = dptrs[1];
   const HYPRE_Complex *dptrs2 = dptrs[2];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val  = a[0].cprod*dptrs0[o0[0] + gi]*dptrs2[o2[0] + gi]*dptrs1[o1[0] + hi] +
             a[1].cprod*dptrs0[o0[1] + gi]*dptrs2[o2[1] + gi]*dptrs1[o1[1] + hi] +
             a[2].cprod*dptrs0[o0[2] + gi]*dptrs2[o2[2] + gi]*dptrs1[o1[2] + hi] +
             a[3].cprod*dptrs0[o0[3] + gi]*dptrs2[o2[3] + gi]*dptrs1[o1[3] + hi] +
             a[4].cprod*dptrs0[o0[4] + gi]*dptrs2[o2[4] + gi]*dptrs1[o1[4] + hi] +
             a[5].cprod*dptrs0[o0[5] + gi]*dptrs2[o2[5] + gi]*dptrs1[o1[5] + hi] +
             a[6].cprod*dptrs0[o0[6] + gi]*dptrs2[o2[6] + gi]*dptrs1[o1[6] + hi];

      a[0].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[7].cprod*dptrs0[o0[7] + gi]*dptrs2[o2[7] + gi]*dptrs1[o1[7] + hi] +
            a[8].cprod*dptrs0[o0[8] + gi]*dptrs2[o2[8] + gi]*dptrs1[o1[8] + hi] +
            a[9].cprod*dptrs0[o0[9] + gi]*dptrs2[o2[9] + gi]*dptrs1[o1[9] + hi];

      a[7].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[10].cprod*dptrs0[o0[10] + gi]*dptrs2[o2[10] + gi]*dptrs1[o1[10] + hi] +
            a[11].cprod*dptrs0[o0[11] + gi]*dptrs2[o2[11] + gi]*dptrs1[o1[11] + hi] +
            a[12].cprod*dptrs0[o0[12] + gi]*dptrs2[o2[12] + gi]*dptrs1[o1[12] + hi];

      a[10].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[13].cprod*dptrs0[o0[13] + gi]*dptrs2[o2[13] + gi]*dptrs1[o1[13] + hi] +
            a[14].cprod*dptrs0[o0[14] + gi]*dptrs2[o2[14] + gi]*dptrs1[o1[14] + hi] +
            a[15].cprod*dptrs0[o0[15] + gi]*dptrs2[o2[15] + gi]*dptrs1[o1[15] + hi];

      a[13].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[16].cprod*dptrs0[o0[16] + gi]*dptrs2[o2[16] + gi]*dptrs1[o1[16] + hi] +
            a[17].cprod*dptrs0[o0[17] + gi]*dptrs2[o2[17] + gi]*dptrs1[o1[17] + hi] +
            a[18].cprod*dptrs0[o0[18] + gi]*dptrs2[o2[18] + gi]*dptrs1[o1[18] + hi];

      a[16].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[19].cprod*dptrs0[o0[19] + gi]*dptrs2[o2[19] + gi]*dptrs1[o1[19] + hi] +
            a[20].cprod*dptrs0[o0[20] + gi]*dptrs2[o2[20] + gi]*dptrs1[o1[20] + hi] +
            a[21].cprod*dptrs0[o0[21] + gi]*dptrs2[o2[21] + gi]*dptrs1[o1[21] + hi];

      a[19].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[22].cprod*dptrs0[o0[22] + gi]*dptrs2[o2[22] + gi]*dptrs1[o1[22] + hi] +
            a[23].cprod*dptrs0[o0[23] + gi]*dptrs2[o2[23] + gi]*dptrs1[o1[23] + hi] +
            a[24].cprod*dptrs0[o0[24] + gi]*dptrs2[o2[24] + gi]*dptrs1[o1[24] + hi];

      a[22].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[25].cprod*dptrs0[o0[25] + gi]*dptrs2[o2[25] + gi]*dptrs1[o1[25] + hi];
      a[25].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[26].cprod*dptrs0[o0[26] + gi]*dptrs2[o2[26] + gi]*dptrs1[o1[26] + hi];
      a[26].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[27].cprod*dptrs0[o0[27] + gi]*dptrs2[o2[27] + gi]*dptrs1[o1[27] + hi];
      a[27].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[28].cprod*dptrs0[o0[28] + gi]*dptrs2[o2[28] + gi]*dptrs1[o1[28] + hi];
      a[28].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[29].cprod*dptrs0[o0[29] + gi]*dptrs2[o2[29] + gi]*dptrs1[o1[29] + hi];
      a[29].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[30].cprod*dptrs0[o0[30] + gi]*dptrs2[o2[30] + gi]*dptrs1[o1[30] + hi];
      a[30].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[31].cprod*dptrs0[o0[31] + gi]*dptrs2[o2[31] + gi]*dptrs1[o1[31] + hi];
      a[31].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[32].cprod*dptrs0[o0[32] + gi]*dptrs2[o2[32] + gi]*dptrs1[o1[32] + hi];
      a[32].mptr[Mi] = val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master
     1) Uses the "+=" sign
     2) as many boxloops as the number of stencil entries in M.
     3) Uses a temporary value to set the resulting stencil coefficient
*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v5b( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   HYPRE_Complex val;
   const HYPRE_Complex *dptrs0 = dptrs[0];
   const HYPRE_Complex *dptrs1 = dptrs[1];
   const HYPRE_Complex *dptrs2 = dptrs[2];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val  = a[0].cprod*dptrs0[o0[0] + gi]*dptrs2[o2[0] + gi]*dptrs1[o1[0] + hi] +
             a[1].cprod*dptrs0[o0[1] + gi]*dptrs2[o2[1] + gi]*dptrs1[o1[1] + hi] +
             a[2].cprod*dptrs0[o0[2] + gi]*dptrs2[o2[2] + gi]*dptrs1[o1[2] + hi] +
             a[3].cprod*dptrs0[o0[3] + gi]*dptrs2[o2[3] + gi]*dptrs1[o1[3] + hi] +
             a[4].cprod*dptrs0[o0[4] + gi]*dptrs2[o2[4] + gi]*dptrs1[o1[4] + hi] +
             a[5].cprod*dptrs0[o0[5] + gi]*dptrs2[o2[5] + gi]*dptrs1[o1[5] + hi] +
             a[6].cprod*dptrs0[o0[6] + gi]*dptrs2[o2[6] + gi]*dptrs1[o1[6] + hi];

      a[0].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[7].cprod*dptrs0[o0[7] + gi]*dptrs2[o2[7] + gi]*dptrs1[o1[7] + hi] +
            a[8].cprod*dptrs0[o0[8] + gi]*dptrs2[o2[8] + gi]*dptrs1[o1[8] + hi] +
            a[9].cprod*dptrs0[o0[9] + gi]*dptrs2[o2[9] + gi]*dptrs1[o1[9] + hi];

      a[7].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[10].cprod*dptrs0[o0[10] + gi]*dptrs2[o2[10] + gi]*dptrs1[o1[10] + hi] +
            a[11].cprod*dptrs0[o0[11] + gi]*dptrs2[o2[11] + gi]*dptrs1[o1[11] + hi] +
            a[12].cprod*dptrs0[o0[12] + gi]*dptrs2[o2[12] + gi]*dptrs1[o1[12] + hi];

      a[10].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[13].cprod*dptrs0[o0[13] + gi]*dptrs2[o2[13] + gi]*dptrs1[o1[13] + hi] +
            a[14].cprod*dptrs0[o0[14] + gi]*dptrs2[o2[14] + gi]*dptrs1[o1[14] + hi] +
            a[15].cprod*dptrs0[o0[15] + gi]*dptrs2[o2[15] + gi]*dptrs1[o1[15] + hi];

      a[13].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[16].cprod*dptrs0[o0[16] + gi]*dptrs2[o2[16] + gi]*dptrs1[o1[16] + hi] +
            a[17].cprod*dptrs0[o0[17] + gi]*dptrs2[o2[17] + gi]*dptrs1[o1[17] + hi] +
            a[18].cprod*dptrs0[o0[18] + gi]*dptrs2[o2[18] + gi]*dptrs1[o1[18] + hi];

      a[16].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[19].cprod*dptrs0[o0[19] + gi]*dptrs2[o2[19] + gi]*dptrs1[o1[19] + hi] +
            a[20].cprod*dptrs0[o0[20] + gi]*dptrs2[o2[20] + gi]*dptrs1[o1[20] + hi] +
            a[21].cprod*dptrs0[o0[21] + gi]*dptrs2[o2[21] + gi]*dptrs1[o1[21] + hi];

      a[19].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[22].cprod*dptrs0[o0[22] + gi]*dptrs2[o2[22] + gi]*dptrs1[o1[22] + hi] +
            a[23].cprod*dptrs0[o0[23] + gi]*dptrs2[o2[23] + gi]*dptrs1[o1[23] + hi] +
            a[24].cprod*dptrs0[o0[24] + gi]*dptrs2[o2[24] + gi]*dptrs1[o1[24] + hi];

      a[22].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {

      val = a[25].cprod*dptrs0[o0[25] + gi]*dptrs2[o2[25] + gi]*dptrs1[o1[25] + hi];
      a[25].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[26].cprod*dptrs0[o0[26] + gi]*dptrs2[o2[26] + gi]*dptrs1[o1[26] + hi];
      a[26].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[27].cprod*dptrs0[o0[27] + gi]*dptrs2[o2[27] + gi]*dptrs1[o1[27] + hi];
      a[27].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[28].cprod*dptrs0[o0[28] + gi]*dptrs2[o2[28] + gi]*dptrs1[o1[28] + hi];
      a[28].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[29].cprod*dptrs0[o0[29] + gi]*dptrs2[o2[29] + gi]*dptrs1[o1[29] + hi];
      a[29].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[30].cprod*dptrs0[o0[30] + gi]*dptrs2[o2[30] + gi]*dptrs1[o1[30] + hi];
      a[30].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[31].cprod*dptrs0[o0[31] + gi]*dptrs2[o2[31] + gi]*dptrs1[o1[31] + hi];
      a[31].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = a[32].cprod*dptrs0[o0[32] + gi]*dptrs2[o2[32] + gi]*dptrs1[o1[32] + hi];
      a[32].mptr[Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master
     1) Uses the "+=" sign
     2) as many boxloops as the number of stencil entries in M.
     3) Uses a temporary value to set the resulting stencil coefficient
*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v6( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   HYPRE_Complex val;
   HYPRE_Complex c[512];
   const HYPRE_Complex *dptrs0 = dptrs[0];
   const HYPRE_Complex *dptrs1 = dptrs[1];
   const HYPRE_Complex *dptrs2 = dptrs[2];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      c[k]  = a[k].cprod;
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   HYPRE_Complex *mptr[15];
   mptr[0 ] = a[0].mptr;
   mptr[1 ] = a[7].mptr;
   mptr[2 ] = a[10].mptr;
   mptr[3 ] = a[13].mptr;
   mptr[4 ] = a[16].mptr;
   mptr[5 ] = a[19].mptr;
   mptr[6 ] = a[22].mptr;
   mptr[7 ] = a[25].mptr;
   mptr[8 ] = a[26].mptr;
   mptr[9 ] = a[27].mptr;
   mptr[10] = a[28].mptr;
   mptr[11] = a[29].mptr;
   mptr[12] = a[30].mptr;
   mptr[13] = a[31].mptr;
   mptr[14] = a[32].mptr;

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val  = c[0]*dptrs0[o0[0] + gi]*dptrs2[o2[0] + gi]*dptrs1[o1[0] + hi] +
             c[1]*dptrs0[o0[1] + gi]*dptrs2[o2[1] + gi]*dptrs1[o1[1] + hi] +
             c[2]*dptrs0[o0[2] + gi]*dptrs2[o2[2] + gi]*dptrs1[o1[2] + hi] +
             c[3]*dptrs0[o0[3] + gi]*dptrs2[o2[3] + gi]*dptrs1[o1[3] + hi] +
             c[4]*dptrs0[o0[4] + gi]*dptrs2[o2[4] + gi]*dptrs1[o1[4] + hi] +
             c[5]*dptrs0[o0[5] + gi]*dptrs2[o2[5] + gi]*dptrs1[o1[5] + hi] +
             c[6]*dptrs0[o0[6] + gi]*dptrs2[o2[6] + gi]*dptrs1[o1[6] + hi];

      mptr[0][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[7]*dptrs0[o0[7] + gi]*dptrs2[o2[7] + gi]*dptrs1[o1[7] + hi] +
            c[8]*dptrs0[o0[8] + gi]*dptrs2[o2[8] + gi]*dptrs1[o1[8] + hi] +
            c[9]*dptrs0[o0[9] + gi]*dptrs2[o2[9] + gi]*dptrs1[o1[9] + hi];

      mptr[1][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[10]*dptrs0[o0[10] + gi]*dptrs2[o2[10] + gi]*dptrs1[o1[10] + hi] +
            c[11]*dptrs0[o0[11] + gi]*dptrs2[o2[11] + gi]*dptrs1[o1[11] + hi] +
            c[12]*dptrs0[o0[12] + gi]*dptrs2[o2[12] + gi]*dptrs1[o1[12] + hi];

      mptr[2][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[13]*dptrs0[o0[13] + gi]*dptrs2[o2[13] + gi]*dptrs1[o1[13] + hi] +
            c[14]*dptrs0[o0[14] + gi]*dptrs2[o2[14] + gi]*dptrs1[o1[14] + hi] +
            c[15]*dptrs0[o0[15] + gi]*dptrs2[o2[15] + gi]*dptrs1[o1[15] + hi];

      mptr[3][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[16]*dptrs0[o0[16] + gi]*dptrs2[o2[16] + gi]*dptrs1[o1[16] + hi] +
            c[17]*dptrs0[o0[17] + gi]*dptrs2[o2[17] + gi]*dptrs1[o1[17] + hi] +
            c[18]*dptrs0[o0[18] + gi]*dptrs2[o2[18] + gi]*dptrs1[o1[18] + hi];

      mptr[4][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[19]*dptrs0[o0[19] + gi]*dptrs2[o2[19] + gi]*dptrs1[o1[19] + hi] +
            c[20]*dptrs0[o0[20] + gi]*dptrs2[o2[20] + gi]*dptrs1[o1[20] + hi] +
            c[21]*dptrs0[o0[21] + gi]*dptrs2[o2[21] + gi]*dptrs1[o1[21] + hi];

      mptr[5][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[22]*dptrs0[o0[22] + gi]*dptrs2[o2[22] + gi]*dptrs1[o1[22] + hi] +
            c[23]*dptrs0[o0[23] + gi]*dptrs2[o2[23] + gi]*dptrs1[o1[23] + hi] +
            c[24]*dptrs0[o0[24] + gi]*dptrs2[o2[24] + gi]*dptrs1[o1[24] + hi];

      mptr[6][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[25]*dptrs0[o0[25] + gi]*dptrs2[o2[25] + gi]*dptrs1[o1[25] + hi];
      mptr[7][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[26]*dptrs0[o0[26] + gi]*dptrs2[o2[26] + gi]*dptrs1[o1[26] + hi];
      mptr[8][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[27]*dptrs0[o0[27] + gi]*dptrs2[o2[27] + gi]*dptrs1[o1[27] + hi];
      mptr[9][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[28]*dptrs0[o0[28] + gi]*dptrs2[o2[28] + gi]*dptrs1[o1[28] + hi];
      mptr[10][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[29]*dptrs0[o0[29] + gi]*dptrs2[o2[29] + gi]*dptrs1[o1[29] + hi];
      mptr[11][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[30]*dptrs0[o0[30] + gi]*dptrs2[o2[30] + gi]*dptrs1[o1[30] + hi];
      mptr[12][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[31]*dptrs0[o0[31] + gi]*dptrs2[o2[31] + gi]*dptrs1[o1[31] + hi];
      mptr[13][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[32]*dptrs0[o0[32] + gi]*dptrs2[o2[32] + gi]*dptrs1[o1[32] + hi];
      mptr[14][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master
     1) Uses the "+=" sign
     2) as many boxloops as the number of stencil entries in M.
     3) Uses a temporary value to set the resulting stencil coefficient
     4) dptrs entries are named explicitly.
*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v7( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Complex **dptrs )

{
   HYPRE_Int  o0[512], o1[512], o2[512];
   HYPRE_Int  k;
   HYPRE_Complex val;
   HYPRE_Complex c[512];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      c[k]  = a[k].cprod;
      o0[k] = a[k].offsets[0];
      o1[k] = a[k].offsets[1];
      o2[k] = a[k].offsets[2];
   }

   const HYPRE_Complex *dptrs00  = dptrs[0] + o0[0];
   const HYPRE_Complex *dptrs01  = dptrs[0] + o0[1];
   const HYPRE_Complex *dptrs02  = dptrs[0] + o0[2];
   const HYPRE_Complex *dptrs03  = dptrs[0] + o0[3];
   const HYPRE_Complex *dptrs04  = dptrs[0] + o0[4];
   const HYPRE_Complex *dptrs05  = dptrs[0] + o0[5];
   const HYPRE_Complex *dptrs06  = dptrs[0] + o0[6];
   const HYPRE_Complex *dptrs07  = dptrs[0] + o0[7];
   const HYPRE_Complex *dptrs08  = dptrs[0] + o0[8];
   const HYPRE_Complex *dptrs09  = dptrs[0] + o0[9];
   const HYPRE_Complex *dptrs010 = dptrs[0] + o0[10];
   const HYPRE_Complex *dptrs011 = dptrs[0] + o0[11];
   const HYPRE_Complex *dptrs012 = dptrs[0] + o0[12];
   const HYPRE_Complex *dptrs013 = dptrs[0] + o0[13];
   const HYPRE_Complex *dptrs014 = dptrs[0] + o0[14];
   const HYPRE_Complex *dptrs015 = dptrs[0] + o0[15];
   const HYPRE_Complex *dptrs016 = dptrs[0] + o0[16];
   const HYPRE_Complex *dptrs017 = dptrs[0] + o0[17];
   const HYPRE_Complex *dptrs018 = dptrs[0] + o0[18];
   const HYPRE_Complex *dptrs019 = dptrs[0] + o0[19];
   const HYPRE_Complex *dptrs020 = dptrs[0] + o0[20];
   const HYPRE_Complex *dptrs021 = dptrs[0] + o0[21];
   const HYPRE_Complex *dptrs022 = dptrs[0] + o0[22];
   const HYPRE_Complex *dptrs023 = dptrs[0] + o0[23];
   const HYPRE_Complex *dptrs024 = dptrs[0] + o0[24];
   const HYPRE_Complex *dptrs025 = dptrs[0] + o0[25];
   const HYPRE_Complex *dptrs026 = dptrs[0] + o0[26];
   const HYPRE_Complex *dptrs027 = dptrs[0] + o0[27];
   const HYPRE_Complex *dptrs028 = dptrs[0] + o0[28];
   const HYPRE_Complex *dptrs029 = dptrs[0] + o0[29];
   const HYPRE_Complex *dptrs030 = dptrs[0] + o0[30];
   const HYPRE_Complex *dptrs031 = dptrs[0] + o0[31];
   const HYPRE_Complex *dptrs032 = dptrs[0] + o0[32];

   const HYPRE_Complex *dptrs10  = dptrs[1] + o0[0];
   const HYPRE_Complex *dptrs11  = dptrs[1] + o0[1];
   const HYPRE_Complex *dptrs12  = dptrs[1] + o0[2];
   const HYPRE_Complex *dptrs13  = dptrs[1] + o0[3];
   const HYPRE_Complex *dptrs14  = dptrs[1] + o0[4];
   const HYPRE_Complex *dptrs15  = dptrs[1] + o0[5];
   const HYPRE_Complex *dptrs16  = dptrs[1] + o0[6];
   const HYPRE_Complex *dptrs17  = dptrs[1] + o0[7];
   const HYPRE_Complex *dptrs18  = dptrs[1] + o0[8];
   const HYPRE_Complex *dptrs19  = dptrs[1] + o0[9];
   const HYPRE_Complex *dptrs110 = dptrs[1] + o0[10];
   const HYPRE_Complex *dptrs111 = dptrs[1] + o0[11];
   const HYPRE_Complex *dptrs112 = dptrs[1] + o0[12];
   const HYPRE_Complex *dptrs113 = dptrs[1] + o0[13];
   const HYPRE_Complex *dptrs114 = dptrs[1] + o0[14];
   const HYPRE_Complex *dptrs115 = dptrs[1] + o0[15];
   const HYPRE_Complex *dptrs116 = dptrs[1] + o0[16];
   const HYPRE_Complex *dptrs117 = dptrs[1] + o0[17];
   const HYPRE_Complex *dptrs118 = dptrs[1] + o0[18];
   const HYPRE_Complex *dptrs119 = dptrs[1] + o0[19];
   const HYPRE_Complex *dptrs120 = dptrs[1] + o0[20];
   const HYPRE_Complex *dptrs121 = dptrs[1] + o0[21];
   const HYPRE_Complex *dptrs122 = dptrs[1] + o0[22];
   const HYPRE_Complex *dptrs123 = dptrs[1] + o0[23];
   const HYPRE_Complex *dptrs124 = dptrs[1] + o0[24];
   const HYPRE_Complex *dptrs125 = dptrs[1] + o0[25];
   const HYPRE_Complex *dptrs126 = dptrs[1] + o0[26];
   const HYPRE_Complex *dptrs127 = dptrs[1] + o0[27];
   const HYPRE_Complex *dptrs128 = dptrs[1] + o0[28];
   const HYPRE_Complex *dptrs129 = dptrs[1] + o0[29];
   const HYPRE_Complex *dptrs130 = dptrs[1] + o0[30];
   const HYPRE_Complex *dptrs131 = dptrs[1] + o0[31];
   const HYPRE_Complex *dptrs132 = dptrs[1] + o0[32];

   const HYPRE_Complex *dptrs20  = dptrs[2] + o0[0];
   const HYPRE_Complex *dptrs21  = dptrs[2] + o0[1];
   const HYPRE_Complex *dptrs22  = dptrs[2] + o0[2];
   const HYPRE_Complex *dptrs23  = dptrs[2] + o0[3];
   const HYPRE_Complex *dptrs24  = dptrs[2] + o0[4];
   const HYPRE_Complex *dptrs25  = dptrs[2] + o0[5];
   const HYPRE_Complex *dptrs26  = dptrs[2] + o0[6];
   const HYPRE_Complex *dptrs27  = dptrs[2] + o0[7];
   const HYPRE_Complex *dptrs28  = dptrs[2] + o0[8];
   const HYPRE_Complex *dptrs29  = dptrs[2] + o0[9];
   const HYPRE_Complex *dptrs210 = dptrs[2] + o0[10];
   const HYPRE_Complex *dptrs211 = dptrs[2] + o0[11];
   const HYPRE_Complex *dptrs212 = dptrs[2] + o0[12];
   const HYPRE_Complex *dptrs213 = dptrs[2] + o0[13];
   const HYPRE_Complex *dptrs214 = dptrs[2] + o0[14];
   const HYPRE_Complex *dptrs215 = dptrs[2] + o0[15];
   const HYPRE_Complex *dptrs216 = dptrs[2] + o0[16];
   const HYPRE_Complex *dptrs217 = dptrs[2] + o0[17];
   const HYPRE_Complex *dptrs218 = dptrs[2] + o0[18];
   const HYPRE_Complex *dptrs219 = dptrs[2] + o0[19];
   const HYPRE_Complex *dptrs220 = dptrs[2] + o0[20];
   const HYPRE_Complex *dptrs221 = dptrs[2] + o0[21];
   const HYPRE_Complex *dptrs222 = dptrs[2] + o0[22];
   const HYPRE_Complex *dptrs223 = dptrs[2] + o0[23];
   const HYPRE_Complex *dptrs224 = dptrs[2] + o0[24];
   const HYPRE_Complex *dptrs225 = dptrs[2] + o0[25];
   const HYPRE_Complex *dptrs226 = dptrs[2] + o0[26];
   const HYPRE_Complex *dptrs227 = dptrs[2] + o0[27];
   const HYPRE_Complex *dptrs228 = dptrs[2] + o0[28];
   const HYPRE_Complex *dptrs229 = dptrs[2] + o0[29];
   const HYPRE_Complex *dptrs230 = dptrs[2] + o0[30];
   const HYPRE_Complex *dptrs231 = dptrs[2] + o0[31];
   const HYPRE_Complex *dptrs232 = dptrs[2] + o0[32];

   HYPRE_Complex *mptr[15];
   mptr[0 ] = a[0].mptr;
   mptr[1 ] = a[7].mptr;
   mptr[2 ] = a[10].mptr;
   mptr[3 ] = a[13].mptr;
   mptr[4 ] = a[16].mptr;
   mptr[5 ] = a[19].mptr;
   mptr[6 ] = a[22].mptr;
   mptr[7 ] = a[25].mptr;
   mptr[8 ] = a[26].mptr;
   mptr[9 ] = a[27].mptr;
   mptr[10] = a[28].mptr;
   mptr[11] = a[29].mptr;
   mptr[12] = a[30].mptr;
   mptr[13] = a[31].mptr;
   mptr[14] = a[32].mptr;

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val  = c[0]*dptrs00[gi]*dptrs20[gi]*dptrs10[hi] +
             c[1]*dptrs01[gi]*dptrs21[gi]*dptrs11[hi] +
             c[2]*dptrs02[gi]*dptrs22[gi]*dptrs12[hi] +
             c[3]*dptrs03[gi]*dptrs23[gi]*dptrs13[hi] +
             c[4]*dptrs04[gi]*dptrs24[gi]*dptrs14[hi] +
             c[5]*dptrs05[gi]*dptrs25[gi]*dptrs15[hi] +
             c[6]*dptrs06[gi]*dptrs26[gi]*dptrs16[hi];

      mptr[0][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[7]*dptrs07[gi]*dptrs27[gi]*dptrs17[hi] +
            c[8]*dptrs08[gi]*dptrs28[gi]*dptrs18[hi] +
            c[9]*dptrs09[gi]*dptrs29[gi]*dptrs19[hi];

      mptr[1][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[10]*dptrs010[gi]*dptrs210[gi]*dptrs110[hi] +
            c[11]*dptrs011[gi]*dptrs211[gi]*dptrs111[hi] +
            c[12]*dptrs012[gi]*dptrs212[gi]*dptrs112[hi];

      mptr[2][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[13]*dptrs013[gi]*dptrs213[gi]*dptrs113[hi] +
            c[14]*dptrs014[gi]*dptrs214[gi]*dptrs114[hi] +
            c[15]*dptrs015[gi]*dptrs215[gi]*dptrs115[hi];

      mptr[3][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[16]*dptrs016[gi]*dptrs216[gi]*dptrs116[hi] +
            c[17]*dptrs017[gi]*dptrs217[gi]*dptrs117[hi] +
            c[18]*dptrs018[gi]*dptrs218[gi]*dptrs118[hi];

      mptr[4][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[19]*dptrs019[gi]*dptrs219[gi]*dptrs119[hi] +
            c[20]*dptrs020[gi]*dptrs220[gi]*dptrs120[hi] +
            c[21]*dptrs021[gi]*dptrs221[gi]*dptrs121[hi];

      mptr[5][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[22]*dptrs022[gi]*dptrs222[gi]*dptrs122[hi] +
            c[23]*dptrs023[gi]*dptrs223[gi]*dptrs123[hi] +
            c[24]*dptrs024[gi]*dptrs224[gi]*dptrs124[hi];

      mptr[6][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[25]*dptrs025[gi]*dptrs225[gi]*dptrs125[hi];
      mptr[7][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[26]*dptrs026[gi]*dptrs226[gi]*dptrs126[hi];
      mptr[8][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[27]*dptrs027[gi]*dptrs227[gi]*dptrs127[hi];
      mptr[9][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[28]*dptrs028[gi]*dptrs228[gi]*dptrs128[hi];
      mptr[10][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[29]*dptrs029[gi]*dptrs229[gi]*dptrs129[hi];
      mptr[11][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[30]*dptrs030[gi]*dptrs230[gi]*dptrs130[hi];
      mptr[12][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[31]*dptrs031[gi]*dptrs231[gi]*dptrs131[hi];
      mptr[13][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[32]*dptrs032[gi]*dptrs232[gi]*dptrs132[hi];
      mptr[14][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master
     1) Uses the "+=" sign
     2) as many boxloops as the number of stencil entries in M.
     3) Uses a temporary value to set the resulting stencil coefficient
     4) dptrs entries are named explicitly through a static array.
*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v8( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride,
                                       HYPRE_Complex **dptrs )

{
   HYPRE_Int  k, kk;
   HYPRE_Complex val;
   HYPRE_Complex c[512];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      c[k] = a[k].cprod;
   }

   const HYPRE_Complex *dptrspo[3][33];
   for (k = 0; k < 3; k++)
   {
      for (kk = 0; kk < 33; kk++)
      {
         dptrspo[k][kk] = dptrs[k] + a[kk].offsets[k];
      }
   }

   HYPRE_Complex *mptr[15];
   mptr[0 ] = a[0].mptr;
   mptr[1 ] = a[7].mptr;
   mptr[2 ] = a[10].mptr;
   mptr[3 ] = a[13].mptr;
   mptr[4 ] = a[16].mptr;
   mptr[5 ] = a[19].mptr;
   mptr[6 ] = a[22].mptr;
   mptr[7 ] = a[25].mptr;
   mptr[8 ] = a[26].mptr;
   mptr[9 ] = a[27].mptr;
   mptr[10] = a[28].mptr;
   mptr[11] = a[29].mptr;
   mptr[12] = a[30].mptr;
   mptr[13] = a[31].mptr;
   mptr[14] = a[32].mptr;

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val  = c[0]*dptrspo[0][0][gi]*dptrspo[2][0][gi]*dptrspo[1][0][hi] +
             c[1]*dptrspo[0][1][gi]*dptrspo[2][1][gi]*dptrspo[1][1][hi] +
             c[2]*dptrspo[0][2][gi]*dptrspo[2][2][gi]*dptrspo[1][2][hi] +
             c[3]*dptrspo[0][3][gi]*dptrspo[2][3][gi]*dptrspo[1][3][hi] +
             c[4]*dptrspo[0][4][gi]*dptrspo[2][4][gi]*dptrspo[1][4][hi] +
             c[5]*dptrspo[0][5][gi]*dptrspo[2][5][gi]*dptrspo[1][5][hi] +
             c[6]*dptrspo[0][6][gi]*dptrspo[2][6][gi]*dptrspo[1][6][hi];

      mptr[0][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[7]*dptrspo[0][7][gi]*dptrspo[2][7][gi]*dptrspo[1][7][hi] +
            c[8]*dptrspo[0][8][gi]*dptrspo[2][8][gi]*dptrspo[1][8][hi] +
            c[9]*dptrspo[0][9][gi]*dptrspo[2][9][gi]*dptrspo[1][9][hi];

      mptr[1][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[10]*dptrspo[0][10][gi]*dptrspo[2][10][gi]*dptrspo[1][10][hi] +
            c[11]*dptrspo[0][11][gi]*dptrspo[2][11][gi]*dptrspo[1][11][hi] +
            c[12]*dptrspo[0][12][gi]*dptrspo[2][12][gi]*dptrspo[1][12][hi];

      mptr[2][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[13]*dptrspo[0][13][gi]*dptrspo[2][13][gi]*dptrspo[1][13][hi] +
            c[14]*dptrspo[0][14][gi]*dptrspo[2][14][gi]*dptrspo[1][14][hi] +
            c[15]*dptrspo[0][15][gi]*dptrspo[2][15][gi]*dptrspo[1][15][hi];

      mptr[3][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[16]*dptrspo[0][16][gi]*dptrspo[2][16][gi]*dptrspo[1][16][hi] +
            c[17]*dptrspo[0][17][gi]*dptrspo[2][17][gi]*dptrspo[1][17][hi] +
            c[18]*dptrspo[0][18][gi]*dptrspo[2][18][gi]*dptrspo[1][18][hi];

      mptr[4][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[19]*dptrspo[0][19][gi]*dptrspo[2][19][gi]*dptrspo[1][19][hi] +
            c[20]*dptrspo[0][20][gi]*dptrspo[2][20][gi]*dptrspo[1][20][hi] +
            c[21]*dptrspo[0][21][gi]*dptrspo[2][21][gi]*dptrspo[1][21][hi];

      mptr[5][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[22]*dptrspo[0][22][gi]*dptrspo[2][22][gi]*dptrspo[1][22][hi] +
            c[23]*dptrspo[0][23][gi]*dptrspo[2][23][gi]*dptrspo[1][23][hi] +
            c[24]*dptrspo[0][24][gi]*dptrspo[2][24][gi]*dptrspo[1][24][hi];

      mptr[6][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[25]*dptrspo[0][25][gi]*dptrspo[2][25][gi]*dptrspo[1][25][hi];
      mptr[7][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[26]*dptrspo[0][26][gi]*dptrspo[2][26][gi]*dptrspo[1][26][hi];
      mptr[8][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[27]*dptrspo[0][27][gi]*dptrspo[2][27][gi]*dptrspo[1][27][hi];
      mptr[9][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[28]*dptrspo[0][28][gi]*dptrspo[2][28][gi]*dptrspo[1][28][hi];
      mptr[10][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[29]*dptrspo[0][29][gi]*dptrspo[2][29][gi]*dptrspo[1][29][hi];
      mptr[11][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[30]*dptrspo[0][30][gi]*dptrspo[2][30][gi]*dptrspo[1][30][hi];
      mptr[12][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[31]*dptrspo[0][31][gi]*dptrspo[2][31][gi]*dptrspo[1][31][hi];
      mptr[13][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[32]*dptrspo[0][32][gi]*dptrspo[2][32][gi]*dptrspo[1][32][hi];
      mptr[14][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master
     1) Uses the "+=" sign
     2) as many boxloops as the number of stencil entries in M.
     3) Uses a temporary value to set the resulting stencil coefficient
     4) tptrs are used
*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v9( hypre_StructMatmultHelper *a,
                                       HYPRE_Int    ncomponents,
                                       HYPRE_Int   *indices,
                                       HYPRE_Int    ndim,
                                       hypre_Index  loop_size,
                                       hypre_Box   *gdbox,
                                       hypre_Index  gdstart,
                                       hypre_Index  gdstride,
                                       hypre_Box   *hdbox,
                                       hypre_Index  hdstart,
                                       hypre_Index  hdstride,
                                       hypre_Box   *Mdbox,
                                       hypre_Index  Mdstart,
                                       hypre_Index  Mdstride )
{
   HYPRE_Int  k, kk;
   HYPRE_Complex val;
   HYPRE_Complex c[512];

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k++)
   {
      c[k] = a[k].cprod;
   }

   const HYPRE_Complex *dptrspo[3][33];
   for (k = 0; k < 3; k++)
   {
      for (kk = 0; kk < 33; kk++)
      {
         dptrspo[k][kk] = a[kk].tptrs[k];
      }
   }

   HYPRE_Complex *mptr[15];
   mptr[0 ] = a[0].mptr;
   mptr[1 ] = a[7].mptr;
   mptr[2 ] = a[10].mptr;
   mptr[3 ] = a[13].mptr;
   mptr[4 ] = a[16].mptr;
   mptr[5 ] = a[19].mptr;
   mptr[6 ] = a[22].mptr;
   mptr[7 ] = a[25].mptr;
   mptr[8 ] = a[26].mptr;
   mptr[9 ] = a[27].mptr;
   mptr[10] = a[28].mptr;
   mptr[11] = a[29].mptr;
   mptr[12] = a[30].mptr;
   mptr[13] = a[31].mptr;
   mptr[14] = a[32].mptr;

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val  = c[0]*dptrspo[0][0][gi]*dptrspo[2][0][gi]*dptrspo[1][0][hi] +
             c[1]*dptrspo[0][1][gi]*dptrspo[2][1][gi]*dptrspo[1][1][hi] +
             c[2]*dptrspo[0][2][gi]*dptrspo[2][2][gi]*dptrspo[1][2][hi] +
             c[3]*dptrspo[0][3][gi]*dptrspo[2][3][gi]*dptrspo[1][3][hi] +
             c[4]*dptrspo[0][4][gi]*dptrspo[2][4][gi]*dptrspo[1][4][hi] +
             c[5]*dptrspo[0][5][gi]*dptrspo[2][5][gi]*dptrspo[1][5][hi] +
             c[6]*dptrspo[0][6][gi]*dptrspo[2][6][gi]*dptrspo[1][6][hi];

      mptr[0][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[7]*dptrspo[0][7][gi]*dptrspo[2][7][gi]*dptrspo[1][7][hi] +
            c[8]*dptrspo[0][8][gi]*dptrspo[2][8][gi]*dptrspo[1][8][hi] +
            c[9]*dptrspo[0][9][gi]*dptrspo[2][9][gi]*dptrspo[1][9][hi];

      mptr[1][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[10]*dptrspo[0][10][gi]*dptrspo[2][10][gi]*dptrspo[1][10][hi] +
            c[11]*dptrspo[0][11][gi]*dptrspo[2][11][gi]*dptrspo[1][11][hi] +
            c[12]*dptrspo[0][12][gi]*dptrspo[2][12][gi]*dptrspo[1][12][hi];

      mptr[2][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[13]*dptrspo[0][13][gi]*dptrspo[2][13][gi]*dptrspo[1][13][hi] +
            c[14]*dptrspo[0][14][gi]*dptrspo[2][14][gi]*dptrspo[1][14][hi] +
            c[15]*dptrspo[0][15][gi]*dptrspo[2][15][gi]*dptrspo[1][15][hi];

      mptr[3][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[16]*dptrspo[0][16][gi]*dptrspo[2][16][gi]*dptrspo[1][16][hi] +
            c[17]*dptrspo[0][17][gi]*dptrspo[2][17][gi]*dptrspo[1][17][hi] +
            c[18]*dptrspo[0][18][gi]*dptrspo[2][18][gi]*dptrspo[1][18][hi];

      mptr[4][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[19]*dptrspo[0][19][gi]*dptrspo[2][19][gi]*dptrspo[1][19][hi] +
            c[20]*dptrspo[0][20][gi]*dptrspo[2][20][gi]*dptrspo[1][20][hi] +
            c[21]*dptrspo[0][21][gi]*dptrspo[2][21][gi]*dptrspo[1][21][hi];

      mptr[5][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[22]*dptrspo[0][22][gi]*dptrspo[2][22][gi]*dptrspo[1][22][hi] +
            c[23]*dptrspo[0][23][gi]*dptrspo[2][23][gi]*dptrspo[1][23][hi] +
            c[24]*dptrspo[0][24][gi]*dptrspo[2][24][gi]*dptrspo[1][24][hi];

      mptr[6][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[25]*dptrspo[0][25][gi]*dptrspo[2][25][gi]*dptrspo[1][25][hi];
      mptr[7][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[26]*dptrspo[0][26][gi]*dptrspo[2][26][gi]*dptrspo[1][26][hi];
      mptr[8][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[27]*dptrspo[0][27][gi]*dptrspo[2][27][gi]*dptrspo[1][27][hi];
      mptr[9][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[28]*dptrspo[0][28][gi]*dptrspo[2][28][gi]*dptrspo[1][28][hi];
      mptr[10][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[29]*dptrspo[0][29][gi]*dptrspo[2][29][gi]*dptrspo[1][29][hi];
      mptr[11][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[30]*dptrspo[0][30][gi]*dptrspo[2][30][gi]*dptrspo[1][30][hi];
      mptr[12][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[31]*dptrspo[0][31][gi]*dptrspo[2][31][gi]*dptrspo[1][31][hi];
      mptr[13][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   hypre_BoxLoop3Begin(ndim, loop_size,
                       Mdbox, Mdstart, Mdstride, Mi,
                       gdbox, gdstart, gdstride, gi,
                       hdbox, hdstart, hdstride, hi);
   {
      val = c[32]*dptrspo[0][32][gi]*dptrspo[2][32][gi]*dptrspo[1][32][hi];
      mptr[14][Mi] += val;
   }
   hypre_BoxLoop3End(Mi,gi,hi);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/* Test with hacked version that approximates master
     1) Uses the "+=" sign
     2) as many boxloops as the number of stencil entries in M.
     3) Uses a temporary value to set the resulting stencil coefficient
     4) tptrs are used
     5) Unrolled version
*/

HYPRE_Int
hypre_StructMatmultCompute_core_2t_v10( hypre_StructMatmultHelper *a,
                                        HYPRE_Int    ncomponents,
                                        HYPRE_Int   *indices,
                                        HYPRE_Int    ndim,
                                        hypre_Index  loop_size,
                                        HYPRE_Int    stencil_size,
                                        hypre_Box   *gdbox,
                                        hypre_Index  gdstart,
                                        hypre_Index  gdstride,
                                        hypre_Box   *hdbox,
                                        hypre_Index  hdstart,
                                        hypre_Index  hdstride,
                                        hypre_Box   *Mdbox,
                                        hypre_Index  Mdstart,
                                        hypre_Index  Mdstride )
{
   HYPRE_ANNOTATE_FUNC_BEGIN;

   HYPRE_Int  mentry;
   HYPRE_Int  count;
   HYPRE_Int  e, k, kk;
   HYPRE_Int  depth;
   HYPRE_Complex val;
   HYPRE_Complex *c;
   HYPRE_Complex *mptr;
   const HYPRE_Complex ***dptrs;

   /* Allocate memory */
   c = hypre_CTAlloc(HYPRE_Complex, ncomponents, HYPRE_MEMORY_HOST);
   dptrs = hypre_CTAlloc(const HYPRE_Complex**, 3, HYPRE_MEMORY_HOST);
   for (kk = 0; kk < 3; kk++)
   {
      dptrs[kk] = hypre_CTAlloc(const HYPRE_Complex*, ncomponents, HYPRE_MEMORY_HOST);
   }

   for (e = 0; e < stencil_size; e++)
   {
      count = 0;
      for (k = 0; k < ncomponents; k++)
      {
         mentry = a[indices[k]].mentry;
         if (mentry == e)
         {
            for (kk = 0; kk < 3; kk++)
            {
               dptrs[kk][count] = a[indices[k]].tptrs[kk];
            }
            c[count] = a[indices[k]].cprod;
            mptr = a[indices[k]].mptr;
            count++;
         }
      }

      for (k = 0; k < count; k += UNROLL_MAXDEPTH)
      {
         depth = hypre_min(UNROLL_MAXDEPTH, (count - k));

         switch (depth)
         {
            case 7:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = c[k+0]*dptrs[0][k+0][gi]*dptrs[2][k+0][gi]*dptrs[1][k+0][hi] +
                        c[k+1]*dptrs[0][k+1][gi]*dptrs[2][k+1][gi]*dptrs[1][k+1][hi] +
                        c[k+2]*dptrs[0][k+2][gi]*dptrs[2][k+2][gi]*dptrs[1][k+2][hi] +
                        c[k+3]*dptrs[0][k+3][gi]*dptrs[2][k+3][gi]*dptrs[1][k+3][hi] +
                        c[k+4]*dptrs[0][k+4][gi]*dptrs[2][k+4][gi]*dptrs[1][k+4][hi] +
                        c[k+5]*dptrs[0][k+5][gi]*dptrs[2][k+5][gi]*dptrs[1][k+5][hi] +
                        c[k+6]*dptrs[0][k+6][gi]*dptrs[2][k+6][gi]*dptrs[1][k+6][hi];

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 6:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = c[k+0]*dptrs[0][k+0][gi]*dptrs[2][k+0][gi]*dptrs[1][k+0][hi] +
                        c[k+1]*dptrs[0][k+1][gi]*dptrs[2][k+1][gi]*dptrs[1][k+1][hi] +
                        c[k+2]*dptrs[0][k+2][gi]*dptrs[2][k+2][gi]*dptrs[1][k+2][hi] +
                        c[k+3]*dptrs[0][k+3][gi]*dptrs[2][k+3][gi]*dptrs[1][k+3][hi] +
                        c[k+4]*dptrs[0][k+4][gi]*dptrs[2][k+4][gi]*dptrs[1][k+4][hi] +
                        c[k+5]*dptrs[0][k+5][gi]*dptrs[2][k+5][gi]*dptrs[1][k+5][hi];

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 5:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = c[k+0]*dptrs[0][k+0][gi]*dptrs[2][k+0][gi]*dptrs[1][k+0][hi] +
                        c[k+1]*dptrs[0][k+1][gi]*dptrs[2][k+1][gi]*dptrs[1][k+1][hi] +
                        c[k+2]*dptrs[0][k+2][gi]*dptrs[2][k+2][gi]*dptrs[1][k+2][hi] +
                        c[k+3]*dptrs[0][k+3][gi]*dptrs[2][k+3][gi]*dptrs[1][k+3][hi] +
                        c[k+4]*dptrs[0][k+4][gi]*dptrs[2][k+4][gi]*dptrs[1][k+4][hi];

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 4:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = c[k+0]*dptrs[0][k+0][gi]*dptrs[2][k+0][gi]*dptrs[1][k+0][hi] +
                        c[k+1]*dptrs[0][k+1][gi]*dptrs[2][k+1][gi]*dptrs[1][k+1][hi] +
                        c[k+2]*dptrs[0][k+2][gi]*dptrs[2][k+2][gi]*dptrs[1][k+2][hi] +
                        c[k+3]*dptrs[0][k+3][gi]*dptrs[2][k+3][gi]*dptrs[1][k+3][hi];

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 3:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = c[k+0]*dptrs[0][k+0][gi]*dptrs[2][k+0][gi]*dptrs[1][k+0][hi] +
                        c[k+1]*dptrs[0][k+1][gi]*dptrs[2][k+1][gi]*dptrs[1][k+1][hi] +
                        c[k+2]*dptrs[0][k+2][gi]*dptrs[2][k+2][gi]*dptrs[1][k+2][hi];

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 2:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = c[k+0]*dptrs[0][k+0][gi]*dptrs[2][k+0][gi]*dptrs[1][k+0][hi] +
                        c[k+1]*dptrs[0][k+1][gi]*dptrs[2][k+1][gi]*dptrs[1][k+1][hi];

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 1:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = c[k+0]*dptrs[0][k+0][gi]*dptrs[2][k+0][gi]*dptrs[1][k+0][hi];

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

             default:
               break;
         }
      }
   }

   /* Free memory */
   hypre_TFree(c, HYPRE_MEMORY_HOST);
   for (k = 0; k < 3; k++)
   {
      hypre_TFree(dptrs[k], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(dptrs, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_2tb
 *
 * Core function for computing the product of three coefficients. Two
 * coefficients are variable and live on data spaces "g" and "h". The third
 * coefficient is constant, it lives on data space "g", and it requires the
 * usage of a bitmask
 *
 * "2tb" means:
 *   "2": two data spaces.
 *   "t": triple product.
 *   "b": single bitmask.
 *
 * This can be used for the scenarios:
 *   1) VCF * VCC * CCF.
 *   2) VCF * CCF * VCC.
 *   3) VCC * VCF * CCF.
 *   4) VCC * CCF * VCF.
 *   5) CCF * VCF * VCC.
 *   6) CCF * VCC * VCF
 *
 * where:
 *   1) VCF stands for "Variable Coefficient on Fine data space".
 *   2) VCC stands for "Variable Coefficient on Coarse data space".
 *   3) CCF stands for "Constant Coefficient on Fine data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2tb( hypre_StructMatmultHelper *a,
                                     HYPRE_Int    ncomponents,
                                     HYPRE_Int   *indices,
                                     HYPRE_Int  **order,
                                     HYPRE_Int    ndim,
                                     hypre_Index  loop_size,
                                     hypre_Box   *gdbox,
                                     hypre_Index  gdstart,
                                     hypre_Index  gdstride,
                                     hypre_Box   *hdbox,
                                     hypre_Index  hdstart,
                                     hypre_Index  hdstride,
                                     hypre_Box   *Mdbox,
                                     hypre_Index  Mdstart,
                                     hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
               HYPRE_SMMCORE_2TB(indices, order, k + 1);
               HYPRE_SMMCORE_2TB(indices, order, k + 2);
               HYPRE_SMMCORE_2TB(indices, order, k + 3);
               HYPRE_SMMCORE_2TB(indices, order, k + 4);
               HYPRE_SMMCORE_2TB(indices, order, k + 5);
               HYPRE_SMMCORE_2TB(indices, order, k + 6);
               HYPRE_SMMCORE_2TB(indices, order, k + 7);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
               HYPRE_SMMCORE_2TB(indices, order, k + 1);
               HYPRE_SMMCORE_2TB(indices, order, k + 2);
               HYPRE_SMMCORE_2TB(indices, order, k + 3);
               HYPRE_SMMCORE_2TB(indices, order, k + 4);
               HYPRE_SMMCORE_2TB(indices, order, k + 5);
               HYPRE_SMMCORE_2TB(indices, order, k + 6);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
               HYPRE_SMMCORE_2TB(indices, order, k + 1);
               HYPRE_SMMCORE_2TB(indices, order, k + 2);
               HYPRE_SMMCORE_2TB(indices, order, k + 3);
               HYPRE_SMMCORE_2TB(indices, order, k + 4);
               HYPRE_SMMCORE_2TB(indices, order, k + 5);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
               HYPRE_SMMCORE_2TB(indices, order, k + 1);
               HYPRE_SMMCORE_2TB(indices, order, k + 2);
               HYPRE_SMMCORE_2TB(indices, order, k + 3);
               HYPRE_SMMCORE_2TB(indices, order, k + 4);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
               HYPRE_SMMCORE_2TB(indices, order, k + 1);
               HYPRE_SMMCORE_2TB(indices, order, k + 2);
               HYPRE_SMMCORE_2TB(indices, order, k + 3);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
               HYPRE_SMMCORE_2TB(indices, order, k + 1);
               HYPRE_SMMCORE_2TB(indices, order, k + 2);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
               HYPRE_SMMCORE_2TB(indices, order, k + 1);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TB(indices, order, k + 0);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

#undef HYPRE_KERNEL
#define HYPRE_KERNEL(k) cprod[k]* \
                        tptrs[0][k][gi]* \
                        tptrs[1][k][hi]* \
                        ((((HYPRE_Int) tptrs[2][k][gi]) >> bitshift[k]) & 1)

HYPRE_Int
hypre_StructMatmultCompute_core_2tb_v2( hypre_StructMatmultHelper *a,
                                     HYPRE_Int    ncomponents,
                                     HYPRE_Int   *indices,
                                     HYPRE_Int  **order,
                                     HYPRE_Int    ndim,
                                     hypre_Index  loop_size,
                                     HYPRE_Int    stencil_size,
                                     hypre_Box   *gdbox,
                                     hypre_Index  gdstart,
                                     hypre_Index  gdstride,
                                     hypre_Box   *hdbox,
                                     hypre_Index  hdstart,
                                     hypre_Index  hdstride,
                                     hypre_Box   *Mdbox,
                                     hypre_Index  Mdstart,
                                     hypre_Index  Mdstride )

{
   HYPRE_ANNOTATE_FUNC_BEGIN;

   HYPRE_Int  mentry;
   HYPRE_Int  count;
   HYPRE_Int  e, k, kk;
   HYPRE_Int  depth;
   HYPRE_Complex val;
   HYPRE_Int *bitshift;
   HYPRE_Complex *cprod;
   HYPRE_Complex *mptr;
   const HYPRE_Complex ***tptrs;

   /* Allocate memory */
   bitshift = hypre_CTAlloc(HYPRE_Int, ncomponents, HYPRE_MEMORY_HOST);
   cprod = hypre_CTAlloc(HYPRE_Complex, ncomponents, HYPRE_MEMORY_HOST);
   tptrs = hypre_CTAlloc(const HYPRE_Complex**, 3, HYPRE_MEMORY_HOST);
   for (kk = 0; kk < 3; kk++)
   {
      tptrs[kk] = hypre_CTAlloc(const HYPRE_Complex*, ncomponents, HYPRE_MEMORY_HOST);
   }

   for (e = 0; e < stencil_size; e++)
   {
      count = 0;
      for (k = 0; k < ncomponents; k++)
      {
         mentry = a[indices[k]].mentry;
         if (mentry == e)
         {
            for (kk = 0; kk < 3; kk++)
            {
               tptrs[kk][count] = a[indices[k]].tptrs[order[k][kk]];
            }
            cprod[count] = a[indices[k]].cprod;
            bitshift[count] = order[k][2];
            mptr = a[indices[k]].mptr;
            count++;
         }
      }

      for (k = 0; k < count; k += UNROLL_MAXDEPTH)
      {
         depth = hypre_min(UNROLL_MAXDEPTH, (count - k));

         switch (depth)
         {
            case 7:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = HYPRE_KERNEL(k + 0) +
                        HYPRE_KERNEL(k + 1) +
                        HYPRE_KERNEL(k + 2) +
                        HYPRE_KERNEL(k + 3) +
                        HYPRE_KERNEL(k + 4) +
                        HYPRE_KERNEL(k + 5) +
                        HYPRE_KERNEL(k + 6);

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 6:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = HYPRE_KERNEL(k + 0) +
                        HYPRE_KERNEL(k + 1) +
                        HYPRE_KERNEL(k + 2) +
                        HYPRE_KERNEL(k + 3) +
                        HYPRE_KERNEL(k + 4) +
                        HYPRE_KERNEL(k + 5);

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 5:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = HYPRE_KERNEL(k + 0) +
                        HYPRE_KERNEL(k + 1) +
                        HYPRE_KERNEL(k + 2) +
                        HYPRE_KERNEL(k + 3) +
                        HYPRE_KERNEL(k + 4);

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 4:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = HYPRE_KERNEL(k + 0) +
                        HYPRE_KERNEL(k + 1) +
                        HYPRE_KERNEL(k + 2) +
                        HYPRE_KERNEL(k + 3);

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 3:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = HYPRE_KERNEL(k + 0) +
                        HYPRE_KERNEL(k + 1) +
                        HYPRE_KERNEL(k + 2);

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 2:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = HYPRE_KERNEL(k + 0) +
                        HYPRE_KERNEL(k + 1);

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

            case 1:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   Mdbox, Mdstart, Mdstride, Mi,
                                   gdbox, gdstart, gdstride, gi,
                                   hdbox, hdstart, hdstride, hi);
               {
                  val = HYPRE_KERNEL(k + 0);

                  mptr[Mi] += val;
               }
               hypre_BoxLoop3End(Mi,gi,hi);
               break;

             default:
               break;
         }
      }
   }

   /* Free memory */
   hypre_TFree(bitshift, HYPRE_MEMORY_HOST);
   hypre_TFree(cprod, HYPRE_MEMORY_HOST);
   for (k = 0; k < 3; k++)
   {
      hypre_TFree(tptrs[k], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(tptrs, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_2etb
 *
 * Core function for computing the product of three coefficients.
 * Two coefficients are variable and live on data space "h".
 * The third coefficient is constant, it lives on data space "g", and it
 * requires the usage of a bitmask
 *
 * "2etb" means:
 *   "2": two data spaces.
 *   "e": data spaces for variable coefficients are the same.
 *   "t": triple product.
 *   "b": single bitmask.
 *
 * This can be used for the scenarios:
 *   1) VCC * VCC * CCF.
 *   2) VCC * CCF * VCC.
 *   3) CCF * VCC * VCC.
 *
 * where:
 *   1) VCC stands for "Variable Coefficient on Coarse data space".
 *   2) CCF stands for "Constant Coefficient on Fine data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2etb( hypre_StructMatmultHelper *a,
                                      HYPRE_Int    ncomponents,
                                      HYPRE_Int   *indices,
                                      HYPRE_Int  **order,
                                      HYPRE_Int    ndim,
                                      hypre_Index  loop_size,
                                      hypre_Box   *gdbox,
                                      hypre_Index  gdstart,
                                      hypre_Index  gdstride,
                                      hypre_Box   *hdbox,
                                      hypre_Index  hdstart,
                                      hypre_Index  hdstride,
                                      hypre_Box   *Mdbox,
                                      hypre_Index  Mdstart,
                                      hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
               HYPRE_SMMCORE_2ETB(indices, order, k + 1);
               HYPRE_SMMCORE_2ETB(indices, order, k + 2);
               HYPRE_SMMCORE_2ETB(indices, order, k + 3);
               HYPRE_SMMCORE_2ETB(indices, order, k + 4);
               HYPRE_SMMCORE_2ETB(indices, order, k + 5);
               HYPRE_SMMCORE_2ETB(indices, order, k + 6);
               HYPRE_SMMCORE_2ETB(indices, order, k + 7);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
               HYPRE_SMMCORE_2ETB(indices, order, k + 1);
               HYPRE_SMMCORE_2ETB(indices, order, k + 2);
               HYPRE_SMMCORE_2ETB(indices, order, k + 3);
               HYPRE_SMMCORE_2ETB(indices, order, k + 4);
               HYPRE_SMMCORE_2ETB(indices, order, k + 5);
               HYPRE_SMMCORE_2ETB(indices, order, k + 6);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
               HYPRE_SMMCORE_2ETB(indices, order, k + 1);
               HYPRE_SMMCORE_2ETB(indices, order, k + 2);
               HYPRE_SMMCORE_2ETB(indices, order, k + 3);
               HYPRE_SMMCORE_2ETB(indices, order, k + 4);
               HYPRE_SMMCORE_2ETB(indices, order, k + 5);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
               HYPRE_SMMCORE_2ETB(indices, order, k + 1);
               HYPRE_SMMCORE_2ETB(indices, order, k + 2);
               HYPRE_SMMCORE_2ETB(indices, order, k + 3);
               HYPRE_SMMCORE_2ETB(indices, order, k + 4);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
               HYPRE_SMMCORE_2ETB(indices, order, k + 1);
               HYPRE_SMMCORE_2ETB(indices, order, k + 2);
               HYPRE_SMMCORE_2ETB(indices, order, k + 3);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
               HYPRE_SMMCORE_2ETB(indices, order, k + 1);
               HYPRE_SMMCORE_2ETB(indices, order, k + 2);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
               HYPRE_SMMCORE_2ETB(indices, order, k + 1);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2ETB(indices, order, k + 0);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmultCompute_core_2tbb
 *
 * Core function for computing the product of three coefficients.
 * One coefficient is variable and live on data space "g".
 * Two coefficients are constant, live on data space "h", and requires
 * the usage of a bitmask.
 *
 * "2etb" means:
 *   "2" : two data spaces.
 *   "t" : triple product.
 *   "bb": two bitmasks.
 *
 * This can be used for the scenarios:
 *   1) VCC * CCF * CCF.
 *   2) CCF * VCC * CCF.
 *   3) CCF * CCF * VCC.
 *
 * where:
 *   1) VCC stands for "Variable Coefficient on Coarse data space".
 *   2) CCF stands for "Constant Coefficient on Fine data space".
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmultCompute_core_2tbb( hypre_StructMatmultHelper *a,
                                      HYPRE_Int    ncomponents,
                                      HYPRE_Int   *indices,
                                      HYPRE_Int  **order,
                                      HYPRE_Int    ndim,
                                      hypre_Index  loop_size,
                                      hypre_Box   *gdbox,
                                      hypre_Index  gdstart,
                                      hypre_Index  gdstride,
                                      hypre_Box   *hdbox,
                                      hypre_Index  hdstart,
                                      hypre_Index  hdstride,
                                      hypre_Box   *Mdbox,
                                      hypre_Index  Mdstart,
                                      hypre_Index  Mdstride )

{
   HYPRE_Int k, depth;

   if (ncomponents < 1)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (k = 0; k < ncomponents; k += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (ncomponents - k));

      switch (depth)
      {
         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
               HYPRE_SMMCORE_2TBB(indices, order, k + 1);
               HYPRE_SMMCORE_2TBB(indices, order, k + 2);
               HYPRE_SMMCORE_2TBB(indices, order, k + 3);
               HYPRE_SMMCORE_2TBB(indices, order, k + 4);
               HYPRE_SMMCORE_2TBB(indices, order, k + 5);
               HYPRE_SMMCORE_2TBB(indices, order, k + 6);
               HYPRE_SMMCORE_2TBB(indices, order, k + 7);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
               HYPRE_SMMCORE_2TBB(indices, order, k + 1);
               HYPRE_SMMCORE_2TBB(indices, order, k + 2);
               HYPRE_SMMCORE_2TBB(indices, order, k + 3);
               HYPRE_SMMCORE_2TBB(indices, order, k + 4);
               HYPRE_SMMCORE_2TBB(indices, order, k + 5);
               HYPRE_SMMCORE_2TBB(indices, order, k + 6);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
               HYPRE_SMMCORE_2TBB(indices, order, k + 1);
               HYPRE_SMMCORE_2TBB(indices, order, k + 2);
               HYPRE_SMMCORE_2TBB(indices, order, k + 3);
               HYPRE_SMMCORE_2TBB(indices, order, k + 4);
               HYPRE_SMMCORE_2TBB(indices, order, k + 5);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
               HYPRE_SMMCORE_2TBB(indices, order, k + 1);
               HYPRE_SMMCORE_2TBB(indices, order, k + 2);
               HYPRE_SMMCORE_2TBB(indices, order, k + 3);
               HYPRE_SMMCORE_2TBB(indices, order, k + 4);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
               HYPRE_SMMCORE_2TBB(indices, order, k + 1);
               HYPRE_SMMCORE_2TBB(indices, order, k + 2);
               HYPRE_SMMCORE_2TBB(indices, order, k + 3);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
               HYPRE_SMMCORE_2TBB(indices, order, k + 1);
               HYPRE_SMMCORE_2TBB(indices, order, k + 2);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
               HYPRE_SMMCORE_2TBB(indices, order, k + 1);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                Mdbox, Mdstart, Mdstride, Mi,
                                gdbox, gdstart, gdstride, gi,
                                hdbox, hdstart, hdstride, hi);
            {
               HYPRE_SMMCORE_2TBB(indices, order, k + 0);
            }
            hypre_BoxLoop3End(Mi,gi,hi);
            break;
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmult
 *
 * Computes the product of "nmatrices" of type hypre_StructMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmult( HYPRE_Int            nmatrices,
                     hypre_StructMatrix **matrices,
                     HYPRE_Int            nterms,
                     HYPRE_Int           *terms,
                     HYPRE_Int           *trans,
                     hypre_StructMatrix **M_ptr )
{
   hypre_StructMatmultData *mmdata;

   hypre_StructMatmultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatmultSetup(mmdata, M_ptr);
   hypre_StructMatmultCommunicate(mmdata, *M_ptr);
   hypre_StructMatmultCompute(mmdata, *M_ptr);
   HYPRE_StructMatrixAssemble(*M_ptr);
   hypre_StructMatmultDestroy(mmdata);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatmat
 *
 * Computes the product of two hypre_StructMatrix objects: M = A*B
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmat( hypre_StructMatrix  *A,
                    hypre_StructMatrix  *B,
                    hypre_StructMatrix **M_ptr )
{
   hypre_StructMatmultData *mmdata;

   HYPRE_Int           nmatrices   = 2;
   HYPRE_StructMatrix  matrices[2] = {A, B};
   HYPRE_Int           nterms      = 2;
   HYPRE_Int           terms[3]    = {0, 1};
   HYPRE_Int           trans[2]    = {0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatmultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatmultSetup(mmdata, M_ptr);
   hypre_StructMatmultCommunicate(mmdata, *M_ptr);
   hypre_StructMatmultCompute(mmdata, *M_ptr);
   hypre_StructMatmultDestroy(mmdata);

   /* Assemble matrix M */
   HYPRE_StructMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixPtAP
 *
 * Computes M = P^T*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixPtAP( hypre_StructMatrix  *A,
                        hypre_StructMatrix  *P,
                        hypre_StructMatrix **M_ptr)
{
   hypre_StructMatmultData *mmdata;

   HYPRE_Int           nmatrices   = 2;
   HYPRE_StructMatrix  matrices[2] = {A, P};
   HYPRE_Int           nterms      = 3;
   HYPRE_Int           terms[3]    = {1, 0, 1};
   HYPRE_Int           trans[3]    = {1, 0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatmultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatmultSetup(mmdata, M_ptr);
   hypre_StructMatmultCommunicate(mmdata, *M_ptr);
   hypre_StructMatmultCompute(mmdata, *M_ptr);
   hypre_StructMatmultDestroy(mmdata);

   /* Assemble matrix M */
   HYPRE_StructMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixRAP
 *
 * Computes M = R*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixRAP( hypre_StructMatrix  *R,
                       hypre_StructMatrix  *A,
                       hypre_StructMatrix  *P,
                       hypre_StructMatrix **M_ptr)
{
   hypre_StructMatmultData *mmdata;

   HYPRE_Int           nmatrices   = 3;
   HYPRE_StructMatrix  matrices[3] = {A, P, R};
   HYPRE_Int           nterms      = 3;
   HYPRE_Int           terms[3]    = {2, 0, 1};
   HYPRE_Int           trans[3]    = {0, 0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatmultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatmultSetup(mmdata, M_ptr);
   hypre_StructMatmultCommunicate(mmdata, *M_ptr);
   hypre_StructMatmultCompute(mmdata, *M_ptr);
   hypre_StructMatmultDestroy(mmdata);

   /* Assemble matrix M */
   HYPRE_StructMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixRTtAP
 *
 * Computes M = RT^T*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixRTtAP( hypre_StructMatrix  *RT,
                         hypre_StructMatrix  *A,
                         hypre_StructMatrix  *P,
                         hypre_StructMatrix **M_ptr)
{
   hypre_StructMatmultData *mmdata;

   HYPRE_Int           nmatrices   = 3;
   HYPRE_StructMatrix  matrices[3] = {A, P, RT};
   HYPRE_Int           nterms      = 3;
   HYPRE_Int           terms[3]    = {2, 0, 1};
   HYPRE_Int           trans[3]    = {1, 0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatmultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatmultSetup(mmdata, M_ptr);
   hypre_StructMatmultCommunicate(mmdata, *M_ptr);
   hypre_StructMatmultCompute(mmdata, *M_ptr);
   hypre_StructMatmultDestroy(mmdata);

   /* Assemble matrix M */
   HYPRE_StructMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}
