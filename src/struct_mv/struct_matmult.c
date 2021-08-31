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

/*--------------------------------------------------------------------------
 * hypre_StructMatrixMultCreate
 *
 * Creates the data structure for computing struct matrix-matrix
 * multiplication.
 *
 * The matrix product has 'nterms' terms constructed from the matrices
 * in the 'matrices' array. Each term t is given by the matrix
 * matrices[terms[t]] transposed according to the boolean transposes[t].
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMultCreate( HYPRE_Int             nmatrices_in,
                              hypre_StructMatrix  **matrices_in,
                              HYPRE_Int             nterms,
                              HYPRE_Int            *terms_in,
                              HYPRE_Int            *transposes_in,
                              hypre_StructMMData  **mmdata_ptr )
{
   hypre_StructMMData   *mmdata;

   hypre_StructMatrix  **matrices;
   HYPRE_Int            *terms;
   HYPRE_Int            *transposes;
   HYPRE_Int            *mtypes;

   hypre_CommPkg       **comm_pkg_a;
   HYPRE_Complex      ***comm_data_a;

   HYPRE_Int             nmatrices, *matmap;
   HYPRE_Int             m, t;

   /* Allocate data structure */
   mmdata = hypre_CTAlloc(hypre_StructMMData, 1, HYPRE_MEMORY_HOST);

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
 * hypre_StructMatrixMultDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMultDestroy( hypre_StructMMData *mmdata )
{
   if (mmdata)
   {
      hypre_TFree(mmdata -> matrices, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> transposes, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> terms, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> mtypes, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> a, HYPRE_MEMORY_HOST);

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
 * hypre_StructMatrixMultSetup
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
hypre_StructMatrixMultSetup( hypre_StructMMData  *mmdata,
                             hypre_StructMatrix **M_ptr )
{
   HYPRE_Int                nterms       = (mmdata -> nterms);
   HYPRE_Int               *terms        = (mmdata -> terms);
   HYPRE_Int               *transposes   = (mmdata -> transposes);
   HYPRE_Int                nmatrices    = (mmdata -> nmatrices);
   HYPRE_Int               *mtypes       = (mmdata -> mtypes);
   hypre_StructMatrix     **matrices     = (mmdata -> matrices);
   hypre_CommPkg          **comm_pkg_a   = (mmdata -> comm_pkg_a);
   HYPRE_Complex         ***comm_data_a  = (mmdata -> comm_data_a);

   hypre_StructMatrix      *M;            /* matrix product we are computing */
   hypre_StructStencil     *Mstencil;
   hypre_StructGrid        *Mgrid;
   hypre_Index              Mran_stride, Mdom_stride;

   MPI_Comm                 comm;
   HYPRE_Int                ndim, size;

   hypre_StructMatrix      *matrix;
   hypre_StructStencil     *stencil;
   hypre_StructGrid        *grid;
   hypre_IndexRef           stride;
   HYPRE_Int                nboxes;
   HYPRE_Int               *boxnums;
   hypre_Box               *box;

   hypre_StMatrix         **st_matrices, *st_matrix, *st_M;
   hypre_StCoeff           *st_coeff;
   hypre_StTerm            *st_term;

   hypre_StructMMHelper    *a;
   HYPRE_Int                na;              /* number of product terms in 'a' */
   HYPRE_Int                nconst;          /* number of constant entries in M */
   HYPRE_Int                const_entry;     /* boolean for constant entry in M */
   HYPRE_Int               *const_entries;   /* constant entries in M */
   HYPRE_Complex           *const_values;    /* values for constant entries in M */

   hypre_CommInfo          *comm_info;
   hypre_CommStencil      **comm_stencils;
   HYPRE_Int                num_comm_blocks;
   HYPRE_Int                num_comm_pkgs;

   hypre_StructVector      *mask;
   HYPRE_Int                need_mask;            /* boolean indicating if a bit mask is needed */
   HYPRE_Int                const_term, var_term; /* booleans used to determine 'need_mask' */

   hypre_IndexRef           ran_stride;
   hypre_IndexRef           dom_stride;
   hypre_IndexRef           coarsen_stride;
   HYPRE_Int                coarsen;
   HYPRE_Complex           *constp;          /* pointer to constant data */
   HYPRE_Complex           *bitptr;          /* pointer to bit mask data */
   hypre_Index              offset;          /* CommStencil offset */
   hypre_IndexRef           shift;           /* stencil shift from center for st_term */
   hypre_IndexRef           offsetref;
   HYPRE_Int                d, i, j, m, t, e, b, id, entry;

   hypre_Box               *loop_box;    /* boxloop extents on the base index space */
   hypre_IndexRef           loop_start;  /* boxloop start index on the base index space */
   hypre_IndexRef           loop_stride; /* boxloop stride on the base index space */
   hypre_Index              loop_size;   /* boxloop size */
   hypre_IndexRef           fstride, cstride; /* data-map strides (base index space) */
   hypre_Index              fdstart;  /* boxloop data starts */
   hypre_Index              fdstride; /* boxloop data strides */
   hypre_Box               *fdbox;    /* boxloop data boxes */

   hypre_BoxArray         **data_spaces;
   hypre_BoxArray          *cdata_space, *fdata_space, *data_space;

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

   const_entries = hypre_TAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
   const_values  = hypre_TAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
   a = hypre_TAlloc(hypre_StructMMHelper, na, HYPRE_MEMORY_HOST);
   (mmdata -> a) = a;

   na = 0;
   nconst = 0;
   need_mask = 0;
   if (hypre_StructGridNumBoxes(grid) > 0)
   {
      comm_stencils = hypre_TAlloc(hypre_CommStencil *, nmatrices+1, HYPRE_MEMORY_HOST);
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

   /* If all constant coefficients, return */
   if (na == 0)
   {
      /* Free up some stuff */
      if (hypre_StructGridNumBoxes(grid) > 0)
      {
         for (m = 0; m < nmatrices+1; m++)
         {
            hypre_CommStencilDestroy(comm_stencils[m]);
         }
         hypre_TFree(comm_stencils, HYPRE_MEMORY_HOST);
      }

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
 * StructMatrixMultCommunicate
 *
 * Communicates matrix and bit mask info with a single commpkg.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMultCommunicate( hypre_StructMMData  *mmdata )
{
   hypre_CommPkg      *comm_pkg        = (mmdata -> comm_pkg);
   HYPRE_Complex     **comm_data       = (mmdata -> comm_data);
   hypre_CommPkg     **comm_pkg_a      = (mmdata -> comm_pkg_a);
   HYPRE_Complex    ***comm_data_a     = (mmdata -> comm_data_a);
   HYPRE_Int           num_comm_pkgs   = (mmdata -> num_comm_pkgs);
   HYPRE_Int           num_comm_blocks = (mmdata -> num_comm_blocks);

   hypre_CommHandle   *comm_handle;
   HYPRE_Int           i, j, nb;

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

      /* Free memory */
      hypre_TFree(comm_pkg_a, HYPRE_MEMORY_HOST);
      hypre_TFree(comm_data_a, HYPRE_MEMORY_HOST);

      mmdata -> comm_pkg_a  = NULL;
      mmdata -> comm_data_a = NULL;
      mmdata -> comm_pkg    = comm_pkg;
      mmdata -> comm_data   = comm_data;
   }
   HYPRE_ANNOTATE_REGION_END("%s", "CommSetup");

   hypre_InitializeCommunication(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
   hypre_FinalizeCommunication(comm_handle);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * StructMatrixMultCompute
 *
 * Computes coefficients of the resulting matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMultCompute( hypre_StructMMData  *mmdata,
                               hypre_StructMatrix  *M )
{
   hypre_StructMMHelper  *a              = (mmdata -> a);
   HYPRE_Int              na             = (mmdata -> na);
   HYPRE_Int              nterms         = (mmdata -> nterms);
   HYPRE_Int             *terms          = (mmdata -> terms);
   HYPRE_Int             *transposes     = (mmdata -> transposes);
   HYPRE_Int              nmatrices      = (mmdata -> nmatrices);
   hypre_StructMatrix   **matrices       = (mmdata -> matrices);
   hypre_BoxArray        *fdata_space    = (mmdata -> fdata_space);
   hypre_BoxArray        *cdata_space    = (mmdata -> cdata_space);
   hypre_StructVector    *mask           = (mmdata -> mask);
   hypre_IndexRef         fstride        = (mmdata -> fstride);
   hypre_IndexRef         cstride        = (mmdata -> cstride);
   hypre_IndexRef         coarsen_stride = (mmdata -> coarsen_stride);

   /* Input matrices variables */
   HYPRE_Int              ndim;
   hypre_StructGrid      *grid;
   HYPRE_Int             *grid_ids;

   /* M matrix variables */
   hypre_StructGrid      *Mgrid;
   hypre_StructMatrix    *matrix;
   hypre_StructStencil   *stencil;
   hypre_StTerm          *st_term;     /* Pointer to stencil info for each term in a */
   hypre_BoxArray        *Mdata_space;
   HYPRE_Int             *Mgrid_ids;

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

   /* Indices */
   HYPRE_Int              entry;
   HYPRE_Int              Mj, Mb, Mentry;
   HYPRE_Int              b, i, id, m, t;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* If all constant coefficients, return */
   if (na == 0)
   {
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /* Initialize data */
   ndim        = hypre_StructMatrixNDim(matrices[0]);
   grid        = hypre_StructMatrixGrid(matrices[0]);
   grid_ids    = hypre_StructGridIDs(grid);

   loop_box    = hypre_BoxCreate(ndim);
   Mgrid       = hypre_StructMatrixGrid(M);
   Mdata_space = hypre_StructMatrixDataSpace(M);
   Mgrid_ids   = hypre_StructGridIDs(Mgrid);

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

      /* TODO: Add DEVICE_VAR */
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
   } /* end loop over matrix M range boxes */

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
 * hypre_StructMatrixMultGroup
 *
 * Computes the product of a group of StructMatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMultGroup(HYPRE_Int            nmatrices,
                            hypre_StructMatrix **matrices,
                            HYPRE_Int            nterms,
                            HYPRE_Int           *terms,
                            HYPRE_Int           *trans,
                            hypre_StructMatrix **M_ptr )
{
   hypre_StructMMData *mmdata;

   hypre_StructMatrixMultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatrixMultSetup(mmdata, M_ptr);
   hypre_StructMatrixMultCommunicate(mmdata);
   hypre_StructMatrixMultCompute(mmdata, *M_ptr);
   HYPRE_StructMatrixAssemble(*M_ptr);
   hypre_StructMatrixMultDestroy(mmdata);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixMult
 *
 * Computes M = A*B
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMult( hypre_StructMatrix  *A,
                        hypre_StructMatrix  *B,
                        hypre_StructMatrix **M_ptr )
{
   hypre_StructMMData *mmdata;

   HYPRE_Int           nmatrices   = 2;
   HYPRE_StructMatrix  matrices[2] = {A, B};
   HYPRE_Int           nterms      = 2;
   HYPRE_Int           terms[3]    = {0, 1};
   HYPRE_Int           trans[2]    = {0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatrixMultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatrixMultSetup(mmdata, M_ptr);
   hypre_StructMatrixMultCommunicate(mmdata);
   hypre_StructMatrixMultCompute(mmdata, *M_ptr);
   hypre_StructMatrixMultDestroy(mmdata);

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
   hypre_StructMMData *mmdata;

   HYPRE_Int           nmatrices   = 2;
   HYPRE_StructMatrix  matrices[2] = {A, P};
   HYPRE_Int           nterms      = 3;
   HYPRE_Int           terms[3]    = {1, 0, 1};
   HYPRE_Int           trans[3]    = {1, 0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatrixMultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatrixMultSetup(mmdata, M_ptr);
   hypre_StructMatrixMultCommunicate(mmdata);
   hypre_StructMatrixMultCompute(mmdata, *M_ptr);
   hypre_StructMatrixMultDestroy(mmdata);

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
   hypre_StructMMData *mmdata;

   HYPRE_Int           nmatrices   = 3;
   HYPRE_StructMatrix  matrices[3] = {A, P, R};
   HYPRE_Int           nterms      = 3;
   HYPRE_Int           terms[3]    = {2, 0, 1};
   HYPRE_Int           trans[3]    = {0, 0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatrixMultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatrixMultSetup(mmdata, M_ptr);
   hypre_StructMatrixMultCommunicate(mmdata);
   hypre_StructMatrixMultCompute(mmdata, *M_ptr);
   hypre_StructMatrixMultDestroy(mmdata);

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
   hypre_StructMMData *mmdata;

   HYPRE_Int           nmatrices   = 3;
   HYPRE_StructMatrix  matrices[3] = {A, P, RT};
   HYPRE_Int           nterms      = 3;
   HYPRE_Int           terms[3]    = {2, 0, 1};
   HYPRE_Int           trans[3]    = {1, 0, 0};

   /* Compute resulting matrix M */
   hypre_StructMatrixMultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_StructMatrixMultSetup(mmdata, M_ptr);
   hypre_StructMatrixMultCommunicate(mmdata);
   hypre_StructMatrixMultCompute(mmdata, *M_ptr);
   hypre_StructMatrixMultDestroy(mmdata);

   /* Assemble matrix M */
   HYPRE_StructMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}
