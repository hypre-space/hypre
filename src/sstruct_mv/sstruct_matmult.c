/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "sstruct_matmult.h"

//#define DEBUG_MATMULT

/*==========================================================================
 * SStructPMatrix matrix-multiply routines
 *
 * The pmatrix (SStructPMatrix) contains an nvars x nvars array of struct
 * (StructMatrix) matrices.  The multiply is then a block-matrix multiply
 * involving these struct matrices.
 *
 * NOTE: This only works for cell-centered variable types (see below comment).
 * This is also restricted to cases where there is only one struct matrix term
 * to compute M_ij of the pmatrix M.
 *
 * RDF: The struct matmult requires a common base grid, but the base grid in a
 * pmatrix will differ depending on the variable types involved (see the sgrids
 * construction in SStructPGridAssemble).  Need to figure out how to handle this
 * (note that the "Engwer trick" would be a good solution and also minimizes the
 * box manager requirements).  Another note: Stencil entries are currently split
 * such that inter-variable-type couplings are put in the unstructured matrix.
 * Hence, with the exception of the above term size restriction, this could be
 * made to work in general.  We ultimately want to have all of the structured
 * stencil entries to go in the pmatrix, of course.
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultCreate(HYPRE_Int                   nmatrices_input,
                            hypre_SStructPMatrix      **pmatrices_input,
                            HYPRE_Int                   nterms,
                            HYPRE_Int                  *terms_input,
                            HYPRE_Int                  *trans_input,
                            hypre_SStructPMatmultData **pmmdata_ptr)
{
   hypre_SStructPMatmultData  *pmmdata;
   hypre_StructMatmultData    *smmdata;
   HYPRE_Int                ***smmid;
   HYPRE_Int                 **smmsz;

   hypre_SStructPMatrix      **pmatrices;
   hypre_StructMatrix        **smatrices;
   HYPRE_Int                  *sterms;

   HYPRE_Int                  *terms;
   HYPRE_Int                  *trans;
   HYPRE_Int                  *matmap;
   HYPRE_Int                   nmatrices;
   HYPRE_Int                   nvars;
   HYPRE_Int                   max_matmults, max_matrices;
   HYPRE_Int                   m, t, vi, vj;

   HYPRE_Int                  *i, *n, k, nn, ii;   /* Nested for-loop variables */
   HYPRE_Int                   zero_product;

   pmmdata = hypre_CTAlloc(hypre_SStructPMatmultData, 1, HYPRE_MEMORY_HOST);

   /* Create new matrices and terms arrays from the input arguments, because we
    * only want to consider those matrices actually involved in the multiply */
   matmap = hypre_CTAlloc(HYPRE_Int, nmatrices_input, HYPRE_MEMORY_HOST);
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
   pmatrices = hypre_CTAlloc(hypre_SStructPMatrix *, nmatrices, HYPRE_MEMORY_HOST);
   terms     = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
   trans     = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
   for (t = 0; t < nterms; t++)
   {
      m = terms_input[t];
      pmatrices[matmap[m]] = pmatrices_input[m];
      terms[t] = matmap[m];
      trans[t] = trans_input[t];
   }
   hypre_TFree(matmap, HYPRE_MEMORY_HOST);

   /* Set nvars */
   nvars = hypre_SStructPMatrixNVars(pmatrices[0]);
   (pmmdata -> nvars) = nvars;

   /* This mimics the following nested for-loop (similar to BoxLoop) to compute
    * the all-at-once PMatrix product M = A1 * A2 * ... * AN, where N = nterms:
    *
    *    for i ...
    *    {
    *       for j ...
    *       {
    *          M_ij = 0
    *          for k1 ...
    *             for k2 ...
    *                ...
    *                   for km ...  // where m = N-1
    *                   {
    *                      M_ij += A1_{i,k1} * A2_{k1,k2} * ... * AN_{km,j}
    *                   }
    *       }
    *    }
    *
    */

   i = hypre_CTAlloc(HYPRE_Int, (nterms + 1), HYPRE_MEMORY_HOST);
   n = hypre_CTAlloc(HYPRE_Int, (nterms + 1), HYPRE_MEMORY_HOST);

   /* In general, we need to have the same number of matrices as terms */
   smatrices = hypre_CTAlloc(hypre_StructMatrix *, nterms, HYPRE_MEMORY_HOST);
   sterms = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
   for (k = 0; k < nterms; k++)
   {
      sterms[k] = k;
   }

   max_matmults = (HYPRE_Int) hypre_pow(nvars, (nterms + 1));
   max_matrices = nterms * nvars * nvars;
   hypre_StructMatmultCreate(max_matmults, max_matrices, &smmdata);
   smmid = hypre_TAlloc(HYPRE_Int **, nvars, HYPRE_MEMORY_HOST);
   smmsz = hypre_TAlloc(HYPRE_Int *, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smmid[vi] = hypre_TAlloc(HYPRE_Int *, nvars, HYPRE_MEMORY_HOST);
      smmsz[vi] = hypre_TAlloc(HYPRE_Int, nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         /* Initialize loop variables */
         nn = 1;
         for (k = 1; k < nterms; k++)
         {
            nn *= nvars;
            i[k] = 0;
            n[k] = nvars - 2;  /* offsetting by 2 produces a simpler comparison below */
         }
         i[0]      = vi;
         i[nterms] = vj;
         n[nterms] = nvars;  /* This ensures that the below loop-index update terminates */

         /* Initialize the array entries to NULL */
         smmid[vi][vj] = hypre_CTAlloc(HYPRE_Int, nn, HYPRE_MEMORY_HOST);
         smmsz[vi][vj] = 0;
         /* Run through the nested for-loop */
         for (ii = 0; ii < nn; ii++)
         {
            /* M_ij += A1_{i,k1} * A2_{k1,k2} * ... * AN_{km,j} */

            zero_product = 0;
            for (k = 0; k < nterms; k++)
            {
               /* If (trans[k] > 0), use the transpose matrix (reverse the indices) */
               smatrices[k] = hypre_SStructPMatrixSMatrix(pmatrices[terms[k]],
                                                          trans[k] ? i[k + 1] : i[k],
                                                          trans[k] ? i[k] : i[k + 1]);
               if (smatrices[k] == NULL)
               {
                  zero_product = 1;
                  break;
               }
            }
            if (!zero_product)
            {
               hypre_StructMatmultSetProduct(smmdata, nterms, smatrices, nterms, sterms, trans,
                                             &smmid[vi][vj][smmsz[vi][vj]]);
               smmsz[vi][vj]++;
            }

            /* Update loop indices */
            for (k = 1; i[k] > n[k]; k++)
            {
               i[k] = 0;
            }
            i[k]++;
         }
      }
   }
   hypre_TFree(i, HYPRE_MEMORY_HOST);
   hypre_TFree(n, HYPRE_MEMORY_HOST);
   hypre_TFree(smatrices, HYPRE_MEMORY_HOST);
   hypre_TFree(sterms, HYPRE_MEMORY_HOST);

   /* Set SStructPMatmultData object */
   (pmmdata -> smmdata)    = smmdata;
   (pmmdata -> smmid)      = smmid;
   (pmmdata -> smmsz)      = smmsz;
   (pmmdata -> nmatrices)  = nmatrices;
   (pmmdata -> pmatrices)  = pmatrices;
   (pmmdata -> nterms)     = nterms;
   (pmmdata -> terms)      = terms;
   (pmmdata -> transposes) = trans;
   (pmmdata -> comm_pkg)   = NULL;
   (pmmdata -> comm_data)  = NULL;

   *pmmdata_ptr = pmmdata;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Destroys an object of type hypre_SStructPMatmultData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultDestroy( hypre_SStructPMatmultData *pmmdata )
{
   HYPRE_Int vi, vj, nvars;

   if (pmmdata)
   {
      hypre_StructMatmultDestroy(pmmdata -> smmdata);
      nvars = (pmmdata -> nvars);
      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            hypre_TFree(pmmdata -> smmid[vi][vj], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(pmmdata -> smmid[vi], HYPRE_MEMORY_HOST);
         hypre_TFree(pmmdata -> smmsz[vi], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(pmmdata -> smmid, HYPRE_MEMORY_HOST);
      hypre_TFree(pmmdata -> smmsz, HYPRE_MEMORY_HOST);

      hypre_TFree(pmmdata -> pmatrices, HYPRE_MEMORY_HOST);
      hypre_TFree(pmmdata -> transposes, HYPRE_MEMORY_HOST);
      hypre_TFree(pmmdata -> terms, HYPRE_MEMORY_HOST);

      hypre_TFree(pmmdata, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultInitialize( hypre_SStructPMatmultData  *pmmdata,
                                 HYPRE_Int                   assemble_grid,
                                 hypre_SStructPMatrix      **pM_ptr )
{
   hypre_StructMatmultData     *smmdata = (pmmdata -> smmdata);
   HYPRE_Int                 ***smmid   = (pmmdata -> smmid);
   HYPRE_Int                  **smmsz   = (pmmdata -> smmsz);
   HYPRE_Int                    nvars   = (pmmdata -> nvars);
   hypre_SStructPMatrix        *pmatrix = pmmdata -> pmatrices[0];

   MPI_Comm                     comm;
   HYPRE_Int                    ndim;
   hypre_StructStencil         *stencil;
   hypre_Index                 *offset;

   hypre_BoxArrayArray         *fpbnd_boxaa;
   hypre_BoxArrayArray         *cpbnd_boxaa;
   hypre_Index                  origin;
   hypre_IndexRef               coarsen_stride;
   HYPRE_Int                    coarsen;
   HYPRE_Int                    num_boxes;
   hypre_BoxArray              *grid_boxes;

   HYPRE_SStructVariable       *vartypes;
   hypre_SStructStencil       **pstencils;
   hypre_SStructPGrid          *pgrid;
   hypre_SStructPGrid          *pfgrid;
   hypre_SStructPMatrix        *pM;
   hypre_StructMatrix          *sM;
   hypre_StructGrid            *sgrid;
   HYPRE_Int                  **smaps;
   HYPRE_Int                   *sentries;

   HYPRE_Int                    vi, vj, e, cnt;
   HYPRE_Int                    pstencil_size;
   HYPRE_Int                    max_stencil_size;

   /* Initialize variables */
   ndim      = hypre_SStructPMatrixNDim(pmatrix);
   comm      = hypre_SStructPMatrixComm(pmatrix);
   pfgrid    = hypre_SStructPMatrixPGrid(pmatrix); /* Same grid for all input matrices */
   vartypes  = hypre_SStructPGridVarTypes(pfgrid);
   hypre_SetIndex(origin, 0);

   /* Check: This currently only works for cell-centered variable types */
   for (vi = 0; vi < nvars; vi++)
   {
      if (vartypes[vi] != HYPRE_SSTRUCT_VARIABLE_CELL)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "SStructPMatmult currently supports only cell variable types\n");
         return hypre_error_flag;
      }
   }

   /* Create temporary semi-struct stencil data structure */
   pstencils = hypre_TAlloc(hypre_SStructStencil *, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      HYPRE_SStructStencilCreate(ndim, 0, &pstencils[vi]);
   }

   /* Create part grid data structure */
   hypre_SStructPGridCreate(comm, ndim, &pgrid);
   hypre_SStructPGridSetVariables(pgrid, nvars, vartypes);
   /* RDF: Need to figure out how to handle the cell grid (see 'RDF' below also) */
   hypre_StructGridDestroy(hypre_SStructPGridCellSGrid(pgrid));
   hypre_SStructPGridCellSGrid(pgrid) = NULL;

   /* Create part matrix data structure */
   hypre_SStructPMatrixCreate(comm, pgrid, pstencils, &pM);
   smaps = hypre_SStructPMatrixSMaps(pM);

   /* Initialize the struct matmults for this part */
   /* RDF NOTE: This does not assemble the struct grids.  They are assembled
    * below or in HYPRE_SStructGridAssemble() to reduce box manager overhead.
    * Check: The struct 'assemble_grid' feature may not be needed anymore. */
   hypre_StructMatmultInitialize(smmdata, 0);

   /* Setup part matrix data structure */
   max_stencil_size = 0;
   for (vi = 0; vi < nvars; vi++)
   {
      pstencil_size = 0;
      coarsen = 0;
      coarsen_stride = NULL;
      for (vj = 0; vj < nvars; vj++)
      {
         /* This currently only works if smmsz[vi][vj] <= 1.  That is, either
          * M_ij = 0 or M_ij = A1_{vi,k1} * A2_{k1,k2} * ... * AN_{km,vj} (only
          * one product in the sum).  TODO: Need to write a matrix sum routine
          * and extend this to work in general. */
         if (smmsz[vi][vj] > 1)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "SStructPMatmult currently supports only one StructMatmult term\n");
            return hypre_error_flag;
         }

         if (smmsz[vi][vj] > 0)
         {
            /* Destroy the default struct matrix for the (vi,vj)-block of the PMatrix */
            sM = hypre_SStructPMatrixSMatrix(pM, vi, vj);
            hypre_StructMatrixDestroy(sM);

            /* There is exactly one product in this block. Transfer its caller
             * reference to pM instead of allocating and adding into a copy. */
            hypre_StructMatmultGetMatrix(smmdata, smmid[vi][vj][0], &sM);
            hypre_SStructPMatrixSMatrix(pM, vi, vj) = sM;

            /* Replace the struct stencil for the (vi,vj)-block with actual stencils */
            stencil = hypre_StructMatrixStencil(sM);
            hypre_SStructPMatrixSStencil(pM, vi, vj) = hypre_StructStencilRef(stencil);

            /* Update the part stencil size */
            pstencil_size += hypre_StructStencilSize(stencil);

            /* Set up the struct grids for the part */
            /* RDF: This may only work when there is a cell variable type.  We need to
             * either construct a cell grid from the struct matmult grids somehow, or
             * do something entirely different. */
            if (hypre_SStructPGridSGrid(pgrid, vi) == NULL)
            {
               /* Set a reference to the grid in sM */
               sgrid = hypre_StructMatrixGrid(sM);
               hypre_StructGridRef(sgrid, &hypre_SStructPGridSGrid(pgrid, vi));

               /* Set periodic based on struct grid (which could be coarsened) */
               hypre_CopyIndex(hypre_StructGridPeriodic(sgrid), hypre_SStructPGridPeriodic(pgrid));

               /* Build part boundaries array */
               num_boxes   = hypre_StructGridNumBoxes(sgrid);
               grid_boxes  = hypre_StructGridBoxes(sgrid);
               fpbnd_boxaa = hypre_SStructPGridPBndBoxArrayArray(pfgrid, vi);
               if (num_boxes)
               {
                  coarsen = (smmdata -> coarsen);
                  if (coarsen)
                  {
                     coarsen_stride = (smmdata -> coarsen_stride);
                     hypre_CoarsenBoxArrayArrayOutward(fpbnd_boxaa, grid_boxes, origin,
                                                       coarsen_stride, &cpbnd_boxaa);
                  }
                  else
                  {
                     cpbnd_boxaa = hypre_BoxArrayArrayClone(fpbnd_boxaa);
                  }
                  hypre_SStructPGridPBndBoxArrayArray(pgrid, vi) = cpbnd_boxaa;
               }
            }
         }
      }
      max_stencil_size = hypre_max(pstencil_size, max_stencil_size);

      /* Update smaps array */
      smaps[vi] = hypre_TReAlloc(smaps[vi], HYPRE_Int, pstencil_size, HYPRE_MEMORY_HOST);

      /* Destroy placeholder semi-struct stencil and update with actual one */
      HYPRE_SStructStencilDestroy(pstencils[vi]);
      HYPRE_SStructStencilCreate(ndim, pstencil_size, &pstencils[vi]);
      cnt = 0;
      for (vj = 0; vj < nvars; vj++)
      {
         sM = hypre_SStructPMatrixSMatrix(pM, vi, vj);
         if (sM)
         {
            stencil = hypre_StructMatrixStencil(sM);
            offset  = hypre_StructStencilShape(stencil);
            for (e = 0; e < hypre_StructStencilSize(stencil); e++)
            {
               HYPRE_SStructStencilSetEntry(pstencils[vi], cnt++, offset[e], vj);
               smaps[vi][e] = e;
            }
         }
      }
   }

   /* Update sentries array */
   hypre_TFree(hypre_SStructPMatrixSEntries(pM), HYPRE_MEMORY_HOST);
   sentries = hypre_TAlloc(HYPRE_Int, max_stencil_size, HYPRE_MEMORY_HOST);
   hypre_SStructPMatrixSEntries(pM) = sentries;
   hypre_SStructPMatrixSEntriesSize(pM) = max_stencil_size;

   if (assemble_grid)
   {
      hypre_SStructPGridAssemble(pgrid);
   }
   hypre_SStructPGridDestroy(pgrid);  /* The grid will remain in the pM matrix */

   /* Point to resulting matrix */
   *pM_ptr = pM;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultCommSetup( hypre_SStructPMatmultData *pmmdata )
{
   hypre_StructMatmultData *smmdata = (pmmdata -> smmdata);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_StructMatmultCommSetup(smmdata);
   (pmmdata -> comm_pkg)  = (smmdata -> comm_pkg);
   (pmmdata -> comm_data) = (smmdata -> comm_data);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

HYPRE_Int
hypre_SStructPMatmultCommunicate( hypre_SStructPMatmultData *pmmdata )
{
   hypre_StructMatmultData *smmdata = (pmmdata -> smmdata);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SStructPMatmultCommSetup(pmmdata);
   hypre_StructMatmultCommunicate(smmdata);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultCompute( hypre_SStructPMatmultData *pmmdata,
                              hypre_SStructPMatrix      *pM )
{
   hypre_StructMatmultData    *smmdata = (pmmdata -> smmdata);
   HYPRE_Int                ***smmid   = (pmmdata -> smmid);
   HYPRE_Int                 **smmsz   = (pmmdata -> smmsz);
   HYPRE_Int                   nvars   = (pmmdata -> nvars);

   HYPRE_Int                   vi, vj;

   HYPRE_UNUSED_VAR(pM);

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         /* The (vi,vj)-block of the PMatrix is the matrix of its only product */
         if (smmsz[vi][vj] > 0)
         {
            hypre_StructMatmultCompute(smmdata, smmid[vi][vj][0]);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the product of several SStructPMatrix matrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmult(HYPRE_Int               nmatrices,
                      hypre_SStructPMatrix  **matrices,
                      HYPRE_Int               nterms,
                      HYPRE_Int              *terms,
                      HYPRE_Int              *trans,
                      hypre_SStructPMatrix  **M_ptr )
{
   hypre_SStructPMatmultData *mmdata;

   hypre_SStructPMatmultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_SStructPMatmultInitialize(mmdata, 1, M_ptr); /* Make sure to assemble the grid */
   hypre_SStructPMatmultCommunicate(mmdata);
   hypre_SStructPMatmultCompute(mmdata, *M_ptr);
   hypre_SStructPMatmultDestroy(mmdata);

   hypre_SStructPMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the product of two SStructPMatrix matrices: M = A*B
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmat( hypre_SStructPMatrix  *A,
                      hypre_SStructPMatrix  *B,
                      hypre_SStructPMatrix **M_ptr )
{
   HYPRE_Int              nmatrices   = 2;
   hypre_SStructPMatrix  *matrices[2] = {A, B};
   HYPRE_Int              nterms      = 2;
   HYPRE_Int              terms[3]    = {0, 1};
   HYPRE_Int              trans[2]    = {0, 0};

   hypre_SStructPMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes M = P^T*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixPtAP( hypre_SStructPMatrix  *A,
                          hypre_SStructPMatrix  *P,
                          hypre_SStructPMatrix **M_ptr )
{
   HYPRE_Int              nmatrices   = 2;
   hypre_SStructPMatrix  *matrices[2] = {A, P};
   HYPRE_Int              nterms      = 3;
   HYPRE_Int              terms[3]    = {1, 0, 1};
   HYPRE_Int              trans[3]    = {1, 0, 0};

   hypre_SStructPMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes M = R*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixRAP( hypre_SStructPMatrix  *R,
                         hypre_SStructPMatrix  *A,
                         hypre_SStructPMatrix  *P,
                         hypre_SStructPMatrix **M_ptr )
{
   HYPRE_Int              nmatrices   = 3;
   hypre_SStructPMatrix  *matrices[3] = {A, P, R};
   HYPRE_Int              nterms      = 3;
   HYPRE_Int              terms[3]    = {2, 0, 1};
   HYPRE_Int              trans[3]    = {0, 0, 0};

   hypre_SStructPMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes M = RT^T*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixRTtAP( hypre_SStructPMatrix  *RT,
                           hypre_SStructPMatrix  *A,
                           hypre_SStructPMatrix  *P,
                           hypre_SStructPMatrix **M_ptr )
{
   HYPRE_Int              nmatrices   = 3;
   hypre_SStructPMatrix  *matrices[3] = {A, P, RT};
   HYPRE_Int              nterms      = 3;
   HYPRE_Int              terms[3]    = {2, 0, 1};
   HYPRE_Int              trans[3]    = {1, 0, 0};

   hypre_SStructPMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}

/*==========================================================================
 * SStructMatrix matrix-multiply routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultCreate(HYPRE_Int                  nmatrices_input,
                           hypre_SStructMatrix      **matrices_input,
                           HYPRE_Int                  nterms,
                           HYPRE_Int                 *terms_input,
                           HYPRE_Int                 *trans_input,
                           hypre_SStructMatmultData **mmdata_ptr)
{
   hypre_SStructMatmultData   *mmdata;
   hypre_SStructPMatmultData  *pmmdata;

   hypre_SStructPMatrix      **pmatrices;
   hypre_SStructMatrix       **matrices;

   HYPRE_Int                  *terms;
   HYPRE_Int                  *trans;
   HYPRE_Int                  *matmap;
   HYPRE_Int                   nmatrices;
   HYPRE_Int                   part, nparts;
   HYPRE_Int                   m, t;

   mmdata = hypre_CTAlloc(hypre_SStructMatmultData, 1, HYPRE_MEMORY_HOST);

   /* Create new matrices and terms arrays from the input arguments, because we
    * only want to consider those matrices actually involved in the multiply */
   matmap = hypre_CTAlloc(HYPRE_Int, nmatrices_input, HYPRE_MEMORY_HOST);
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
   matrices  = hypre_CTAlloc(hypre_SStructMatrix *, nmatrices, HYPRE_MEMORY_HOST);
   pmatrices = hypre_CTAlloc(hypre_SStructPMatrix *, nmatrices, HYPRE_MEMORY_HOST);
   terms     = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
   trans     = hypre_CTAlloc(HYPRE_Int, nterms, HYPRE_MEMORY_HOST);
   for (t = 0; t < nterms; t++)
   {
      m = terms_input[t];
      matrices[matmap[m]] = matrices_input[m];
      terms[t] = matmap[m];
      trans[t] = trans_input[t];
   }
   hypre_TFree(matmap, HYPRE_MEMORY_HOST);

   /* Set number of parts */
   nparts = hypre_SStructMatrixNParts(matrices[0]);
   (mmdata -> nparts) = nparts;

   /* Create SStructPMatmultData object */
   (mmdata -> pmmdata) = hypre_TAlloc(hypre_SStructPMatmultData *, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      for (m = 0; m < nmatrices; m++)
      {
         pmatrices[m] = hypre_SStructMatrixPMatrix(matrices[m], part);
      }
      hypre_SStructPMatmultCreate(nmatrices, pmatrices, nterms, terms, trans, &pmmdata);
      (mmdata -> pmmdata)[part] = pmmdata;
   }
   hypre_TFree(pmatrices, HYPRE_MEMORY_HOST);

   /* Set SStructMatmultData object */
   (mmdata -> nterms)      = nterms;
   (mmdata -> nmatrices)   = nmatrices;
   (mmdata -> matrices)    = matrices;
   (mmdata -> terms)       = terms;
   (mmdata -> transposes)  = trans;
   (mmdata -> comm_pkg)    = NULL;
   (mmdata -> comm_data)   = NULL;

   *mmdata_ptr = mmdata;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Destroys an object of type hypre_SStructMatmultData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultDestroy( hypre_SStructMatmultData *mmdata )
{
   HYPRE_Int part, nparts;

   if (mmdata)
   {
      nparts = (mmdata -> nparts);
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPMatmultDestroy((mmdata -> pmmdata)[part]);
      }
      hypre_TFree(mmdata -> pmmdata, HYPRE_MEMORY_HOST);

      hypre_TFree(mmdata -> matrices, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> transposes, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> terms, HYPRE_MEMORY_HOST);

      hypre_CommPkgDestroy(mmdata -> comm_pkg);
      hypre_TFree(mmdata -> comm_data, HYPRE_MEMORY_HOST);

      hypre_TFree(mmdata, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatmultInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultInitialize( hypre_SStructMatmultData   *mmdata,
                                hypre_SStructMatrix       **M_ptr )
{
   HYPRE_Int                   nparts   = (mmdata -> nparts);
   HYPRE_Int                   nterms   = (mmdata -> nterms);
   HYPRE_Int                  *terms    = (mmdata -> terms);
   HYPRE_Int                  *trans    = (mmdata -> transposes);
   hypre_SStructPMatmultData **pmmdata  = (mmdata -> pmmdata);
   hypre_SStructMatrix       **matrices = (mmdata -> matrices);

   /* M matrix variables */
   hypre_SStructMatrix        *M;
   hypre_SStructGrid          *Mgrid;
   hypre_SStructGraph         *Mgraph;
   hypre_SStructPMatrix       *pM;
   hypre_SStructStencil       *stencil;
   HYPRE_Int                   stencil_size;
   HYPRE_Int                   pstencil_size;
   HYPRE_Int                   max_stencil_size;
   HYPRE_Int                ***splits;
   HYPRE_Int                  *sentries;
   HYPRE_Int                  *uentries;

   /* Unstructured component variables */
   hypre_IJMatrix             *ij_M;
   hypre_IJMatrix             *ijmatrix;
   HYPRE_BigInt                ilower, iupper;
   HYPRE_BigInt                jlower, jupper;

   /* Input matrices variables */
   hypre_SStructPMatrix       *pmatrix;
   HYPRE_SStructVariable      *vartypes;
   hypre_SStructPGrid         *pgrid;

   /* Local variables */
   MPI_Comm                    comm;
   HYPRE_Int                   ndim;
   HYPRE_Int                   part;
   HYPRE_Int                   i, vi, vj, nvars;

   /* TODO: sanity check for input matrices */

   /* Initialize variables */
   comm   = hypre_SStructMatrixComm(matrices[0]);
   ndim   = hypre_SStructMatrixNDim(matrices[0]);

   /* Create the grid for M */
   HYPRE_SStructGridCreate(comm, ndim, nparts, &Mgrid);
   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrices[0], part);
      pgrid = hypre_SStructPMatrixPGrid(pmatrix);
      nvars = hypre_SStructPGridNVars(pgrid);
      vartypes = hypre_SStructPGridVarTypes(pgrid);

      HYPRE_SStructGridSetVariables(Mgrid, part, nvars, vartypes);
   }

   /* Create M's graph */
   HYPRE_SStructGraphCreate(comm, Mgrid, &Mgraph);

   /* Create temporary stencil data structure */
   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(hypre_SStructGridPGrid(Mgrid, part));
      for (vi = 0; vi < nvars; vi++)
      {
         HYPRE_SStructStencilCreate(ndim, 0, &stencil);
         HYPRE_SStructGraphSetStencil(Mgraph, part, vi, stencil);
         HYPRE_SStructStencilDestroy(stencil);
      }
   }

   /* Assemble M's graph */
   HYPRE_SStructGraphAssemble(Mgraph);

   /* Create the matrix M */
   HYPRE_SStructMatrixCreate(comm, Mgraph, &M);
   splits = hypre_SStructMatrixSplits(M);

   /* Decrease reference counter for Mgraph and Mgrid */
   HYPRE_SStructGraphDestroy(Mgraph);
   HYPRE_SStructGridDestroy(Mgrid);

   /* Setup Pmatrix */
   max_stencil_size = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(Mgrid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      /* Create resulting part matrix */
      hypre_SStructPMatmultInitialize(pmmdata[part], 0, &pM); /* Don't assemble the grid */

      /* Update part matrix of M */
      hypre_SStructMatrixPMatrix(M, part) = pM;

      /* Update part grid of M */
      hypre_SStructPGridDestroy(pgrid);
      pgrid = hypre_SStructPMatrixPGrid(pM);
      hypre_SStructPGridRef(pgrid, &hypre_SStructGridPGrid(Mgrid, part));

      /* Update graph stencils */
      for (vi = 0; vi < nvars; vi++)
      {
         stencil = hypre_SStructGraphStencil(Mgraph, part, vi);
         HYPRE_SStructStencilDestroy(stencil);
         stencil = hypre_SStructPMatrixStencil(pM, vi);
         hypre_SStructStencilRef(stencil, &hypre_SStructGraphStencil(Mgraph, part, vi));
         stencil_size = hypre_SStructStencilSize(stencil);
         max_stencil_size = hypre_max(max_stencil_size, stencil_size);

         /* Update split array */
         splits[part][vi] = hypre_TReAlloc(splits[part][vi], HYPRE_Int,
                                           stencil_size, HYPRE_MEMORY_HOST);
         pstencil_size = 0;
         for (i = 0; i < stencil_size; i++)
         {
            vj = hypre_SStructStencilVar(stencil, i);
            if (hypre_SStructPGridVarType(pgrid, vi) ==
                hypre_SStructPGridVarType(pgrid, vj))
            {
               splits[part][vi][i] = pstencil_size;
               pstencil_size++;
            }
            else
            {
               splits[part][vi][i] = -1;
            }
         }
      }
   }

   /* Update sentries and uentries arrays */
   hypre_TFree(hypre_SStructMatrixSEntries(M), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_SStructMatrixUEntries(M), HYPRE_MEMORY_HOST);
   sentries = hypre_TAlloc(HYPRE_Int, max_stencil_size, HYPRE_MEMORY_HOST);
   uentries = hypre_TAlloc(HYPRE_Int, max_stencil_size, HYPRE_MEMORY_HOST);
   hypre_SStructMatrixSEntries(M) = sentries;
   hypre_SStructMatrixUEntries(M) = uentries;

   /* Assemble semi-struct grid */
   HYPRE_SStructGridAssemble(Mgrid);

   /* Set row bounds of the unstructured matrix component */
   ijmatrix = hypre_SStructMatrixIJMatrix(matrices[terms[0]]);
   if (trans[0])
   {
      ilower = hypre_IJMatrixColPartitioning(ijmatrix)[0];
      iupper = hypre_IJMatrixColPartitioning(ijmatrix)[1] - 1;
   }
   else
   {
      ilower = hypre_IJMatrixRowPartitioning(ijmatrix)[0];
      iupper = hypre_IJMatrixRowPartitioning(ijmatrix)[1] - 1;
   }

   /* Set column bounds of the unstructured matrix component */
   ijmatrix = hypre_SStructMatrixIJMatrix(matrices[terms[nterms - 1]]);
   if (trans[nterms - 1])
   {
      jlower = hypre_IJMatrixRowPartitioning(ijmatrix)[0];
      jupper = hypre_IJMatrixRowPartitioning(ijmatrix)[1] - 1;
   }
   else
   {
      jlower = hypre_IJMatrixColPartitioning(ijmatrix)[0];
      jupper = hypre_IJMatrixColPartitioning(ijmatrix)[1] - 1;
   }

   /* Create the unstructured matrix component (UMatrix) */
   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &ij_M);
   HYPRE_IJMatrixSetObjectType(ij_M, HYPRE_PARCSR);
   hypre_SStructMatrixIJMatrix(M) = ij_M;

   *M_ptr = M;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Run communication phase for computing the structured component of M
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultCommunicate( hypre_SStructMatmultData *mmdata )
{
   HYPRE_Int                   nparts     = (mmdata -> nparts);
   hypre_SStructPMatmultData **pmmdata    = (mmdata -> pmmdata);
   hypre_CommPkg              *comm_pkg   = (mmdata -> comm_pkg);
   HYPRE_Complex             **comm_data  = (mmdata -> comm_data);

   hypre_CommPkg             **comm_pkg_a;
   HYPRE_Complex            ***comm_data_a;
   HYPRE_Int                   num_comm_pkgs;

   HYPRE_Int                   part;
   hypre_CommHandle           *comm_handle;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatmultCommSetup(pmmdata[part]);
   }

   /* Allocate communication package and data arrays */
   comm_pkg_a  = hypre_TAlloc(hypre_CommPkg *, nparts, HYPRE_MEMORY_HOST);
   comm_data_a = hypre_TAlloc(HYPRE_Complex **, nparts, HYPRE_MEMORY_HOST);

   /* Update pointers to communication packages and data */
   num_comm_pkgs = 0;
   for (part = 0; part < nparts; part++)
   {
      if (pmmdata[part] && (pmmdata[part] -> comm_pkg))
      {
         comm_pkg_a[num_comm_pkgs]  = (pmmdata[part] -> comm_pkg);
         comm_data_a[num_comm_pkgs] = (pmmdata[part] -> comm_data);
         num_comm_pkgs++;
      }
   }

   if (num_comm_pkgs > 0)
   {
      /* Agglomerate communication packages if needed */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "CommSetup");
      if (!comm_pkg)
      {
         hypre_CommPkgAgglomerate(num_comm_pkgs, comm_pkg_a, &comm_pkg);
         hypre_CommPkgAgglomData(num_comm_pkgs, comm_pkg_a, comm_data_a, comm_pkg, &comm_data);
         (mmdata -> comm_pkg)  = comm_pkg;
         (mmdata -> comm_data) = comm_data;
      }
      HYPRE_ANNOTATE_REGION_END("%s", "CommSetup");

      hypre_StructCommunicationInitialize(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
      hypre_StructCommunicationFinalize(comm_handle);
   }

   /* Free up communication package and data arrays */
   hypre_TFree(comm_pkg_a, HYPRE_MEMORY_HOST);
   hypre_TFree(comm_data_a, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the structured component of the product of SStructMatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultComputeS( hypre_SStructMatmultData *mmdata,
                              hypre_SStructMatrix      *M )
{
   HYPRE_Int                   nparts  = (mmdata -> nparts);
   hypre_SStructPMatmultData **pmmdata = (mmdata -> pmmdata);

   hypre_SStructPMatrix       *pM;
   HYPRE_Int                   part;

   for (part = 0; part < nparts; part++)
   {
      pM = hypre_SStructMatrixPMatrix(M, part);
      hypre_SStructPMatmultCompute(pmmdata[part], pM);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the unstructured component of a PtAP product when both uA and uP
 * are nonzero, where sM = sP^T*sA*sP has already been computed natively.
 * Only the rows J of M that differ from sM are formed: complete rows of
 * P^T*A*P from the rows of A and P that reach J. They become uM, and the
 * caller must then set the stencil entries of sM from them with
 * hypre_SStructMatrixCompressUToS(M, -1). Support boxes are exchanged between
 * neighboring ranks only.
 *
 * Sets *supported to zero on all ranks when the layout is not supported,
 * and to one when the result is computed.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixPtAPBoundary( hypre_SStructMatrix  *A,
                                 hypre_SStructMatrix  *P,
                                 hypre_SStructMatrix  *M,
                                 hypre_ParCSRMatrix  **result,
                                 HYPRE_Int            *supported )
{
   MPI_Comm                       comm   = hypre_SStructMatrixComm(A);
   HYPRE_Int                      nparts = hypre_SStructMatrixNParts(A);
   HYPRE_Int                      ndim   = hypre_SStructMatrixNDim(A);
   hypre_SStructGrid             *fine   = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(A));
   hypre_SStructGrid             *pgrid  = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(P));
   hypre_SStructGrid             *coarse = hypre_SStructGraphDomGrid(hypre_SStructMatrixGraph(P));
   hypre_SStructGrid             *mgrid  = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(M));
   hypre_ParCSRMatrix            *uA     = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParCSRMatrix            *uP     = hypre_SStructMatrixParCSRMatrix(P);
   HYPRE_MemoryLocation           location = hypre_ParCSRMatrixMemoryLocation(uP);

   /* Per-part matrix data */
   hypre_SStructPMatrix          *pa, *pp, *pm;
   hypre_SStructStencil          *astencil, *pstencil, *mstencil;
   hypre_StructMatrix            *sm;
   hypre_StructGrid              *sgrid;
   hypre_IndexRef                 stride;
   hypre_BoxArray                *fboxes, *cboxes, *refined;
   hypre_Box                     *box;
   HYPRE_Int                     *msplit;

   /* reach[0:6] is the stencil reach of A, reach[6:12] the distance from the
      local fine boxes up to which support boxes can extend */
   HYPRE_Int                      reach[12] = {0};
   HYPRE_Int                      global_reach[12];
   HYPRE_Int                     *global_halo = global_reach;
   HYPRE_Int                     *nbr_halo    = global_reach + 6;
   HYPRE_Int                      valid, all_valid, known, preach;

   /* Support boxes and intermediate matrices */
   hypre_SStructGridRemoteBoxes  *rboxes;
   hypre_BoxArray              ***pu_rows, ***boxes, ***right, ***left, ***rows_J;
   hypre_IJMatrix                *ij_s = NULL, *ij_a = NULL, *ij_p = NULL;
   hypre_ParCSRMatrix            *a, *p, *restricted, *rap;
   HYPRE_Int                     *marker;
   HYPRE_Int                      ncols;
   HYPRE_Int                      part, d, e, b;

   *supported = 0;

   if (A == P || ndim < 1 || ndim > 3 ||
       nparts != hypre_SStructMatrixNParts(P) ||
       ndim != hypre_SStructMatrixNDim(P) ||
       hypre_SStructGraphDomGrid(hypre_SStructMatrixGraph(A)) != fine)
   {
      return hypre_error_flag;
   }
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(hypre_SStructMatrixMemoryLocation(A)) == HYPRE_EXEC_DEVICE)
   {
      return hypre_error_flag;
   }
#endif

   /*------------------------------------------------------
    * Check the layout of A, P and M
    *------------------------------------------------------*/

   box   = hypre_BoxCreate(ndim);
   hypre_SStructGridMatch(fine, pgrid, &valid);
   if (valid)
   {
      hypre_SStructGridMatch(coarse, mgrid, &valid);
   }
   if (hypre_ParCSRMatrixMemoryLocation(uA) != location ||
       hypre_ParCSRMatrixNumRows(uA) != hypre_ParCSRMatrixNumRows(uP) ||
       hypre_ParCSRMatrixFirstRowIndex(uA) != hypre_ParCSRMatrixFirstRowIndex(uP) ||
       hypre_IJMatrixRowPartitioning(hypre_SStructMatrixIJMatrix(M))[0] !=
       hypre_ParCSRMatrixFirstColDiag(uP))
   {
      valid = 0;
   }

   for (part = 0; valid && part < nparts; part++)
   {
      pa = hypre_SStructMatrixPMatrix(A, part);
      pp = hypre_SStructMatrixPMatrix(P, part);
      pm = hypre_SStructMatrixPMatrix(M, part);
      if (hypre_SStructPMatrixNVars(pa) != 1 ||
          hypre_SStructPMatrixNVars(pp) != 1 ||
          hypre_SStructPMatrixNVars(pm) != 1)
      {
         valid = 0;
         break;
      }

      stride   = hypre_SStructPMatrixDomainStride(pp);
      sm       = hypre_SStructPMatrixSMatrix(pm, 0, 0);
      astencil = hypre_SStructPMatrixStencil(pa, 0);
      pstencil = hypre_SStructPMatrixStencil(pp, 0);
      mstencil = hypre_SStructPMatrixStencil(pm, 0);
      fboxes   = hypre_StructGridBoxes(hypre_SStructPGridSGrid(hypre_SStructGridPGrid(fine, part), 0));
      cboxes   = hypre_StructGridBoxes(hypre_SStructPGridSGrid(hypre_SStructGridPGrid(coarse, part), 0));
      msplit   = hypre_SStructMatrixSplit(M, part, 0);

      /* Strides */
      for (d = 0; d < ndim; d++)
      {
         if (hypre_SStructPMatrixDomainStride(pa)[d] != 1 ||
             hypre_SStructPMatrixRangeStride(pa)[d] != 1 ||
             hypre_SStructPMatrixRangeStride(pp)[d] != 1 || stride[d] < 1 ||
             hypre_SStructPMatrixDomainStride(pm)[d] != 1 ||
             hypre_SStructPMatrixRangeStride(pm)[d] != 1 ||
             hypre_StructMatrixDataStride(sm)[d] != 1)
         {
            valid = 0;
         }
      }

      /* Stencil reach of A and P */
      for (e = 0; e < hypre_SStructStencilSize(astencil); e++)
      {
         for (d = 0; d < ndim; d++)
         {
            reach[2 * d] = hypre_max(reach[2 * d],
                                     hypre_abs(hypre_SStructStencilShape(astencil)[e][d]));
         }
      }
      for (d = 0; d < ndim; d++)
      {
         preach = 0;
         for (e = 0; e < hypre_SStructStencilSize(pstencil); e++)
         {
            preach = hypre_max(preach, hypre_abs(hypre_SStructStencilShape(pstencil)[e][d]));
         }
         reach[2 * d + 1] = reach[2 * d];
         reach[6 + 2 * d] = reach[6 + 2 * d + 1] =
                               hypre_max(reach[6 + 2 * d], stride[d] - 1 + preach + reach[2 * d]);
      }

      /* Rows J of sM are overwritten by hypre_SStructMatrixCompressUToS */
      for (e = 0; e < hypre_StructStencilSize(hypre_StructMatrixStencil(sm)); e++)
      {
         if (hypre_StructMatrixConstEntry(sm, e) || hypre_StructMatrixSymmEntries(sm)[e] >= 0)
         {
            valid = 0;
         }
      }
      for (e = 0; e < hypre_SStructStencilSize(mstencil); e++)
      {
         if (msplit[e] < 0 || hypre_SStructStencilVars(mstencil)[e])
         {
            valid = 0;
         }
      }

      /* The fine rows of P with columns in the local coarse box must be near the local fine box */
      if (hypre_BoxArraySize(cboxes))
      {
         hypre_CopyBox(hypre_BoxArrayBox(cboxes, 0), box);
         hypre_RefineBox(box, NULL, stride);
         for (d = 0; valid && d < ndim; d++)
         {
            if (!hypre_BoxArraySize(fboxes) ||
                hypre_BoxIMinD(box, d) < hypre_BoxIMinD(hypre_BoxArrayBox(fboxes, 0), d) - stride[d] + 1 ||
                hypre_BoxIMaxD(box, d) > hypre_BoxIMaxD(hypre_BoxArrayBox(fboxes, 0), d) + stride[d] - 1)
            {
               valid = 0;
            }
         }
      }
   }
   hypre_MPI_Allreduce(reach, global_reach, 12, hypre_MPI_INT, hypre_MPI_MAX, comm);

   /* The box managers must know the boxes within nbr_halo of the local boxes.
      They hold the remote boxes known to the box managers of the struct grids. */
   for (part = 0; part < nparts; part++)
   {
      sgrid = hypre_SStructPGridSGrid(hypre_SStructGridPGrid(fine, part), 0);
      hypre_BoxManGetAllGlobalKnown(hypre_StructGridBoxMan(sgrid), &known);
      for (d = 0; !known && d < ndim; d++)
      {
         if (nbr_halo[2 * d] > hypre_StructGridMaxDistance(sgrid)[d])
         {
            valid = 0;
         }
      }
   }
   hypre_MPI_Allreduce(&valid, &all_valid, 1, hypre_MPI_INT, hypre_MPI_MIN, comm);
   if (!all_valid)
   {
      hypre_BoxDestroy(box);

      return hypre_error_flag;
   }

   /*------------------------------------------------------
    * Find the rows J of M that differ from sM
    *
    * uM has nonzero rows J: the columns of uP, and the columns of sP over the
    * rows of uA and over the rows coupled by sA to the rows of uP
    *------------------------------------------------------*/

   rboxes = hypre_SStructGridRemoteBoxesCreate(fine, nbr_halo);
   ncols  = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(uP));
   marker = hypre_CTAlloc(HYPRE_Int, ncols, location);

   /* Mark the rows of uA once, retaining both row-only and row/column support */
   hypre_SStructMatrixCollectSupport(pgrid, uP, 0, &pu_rows, NULL);
   hypre_SStructMatrixCollectSupport(fine, uA, 1, &right, &boxes);
   for (part = 0; part < nparts; part++)
   {
      pa = hypre_SStructMatrixPMatrix(A, part);
      hypre_SStructStencilAppendCoupledBoxes(hypre_SStructPMatrixStencil(pa, 0),
                                             pu_rows[part][0], boxes[part][0]);
   }
   hypre_SStructGridRemoteBoxesShare(fine, rboxes, boxes);
   hypre_SStructMatrixBoxesToUMatrix(P, pgrid, &ij_s, boxes);
   hypre_SStructMatrixFreeSupport(fine, boxes);
   hypre_ParCSRMatrixMarkColumns((hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij_s), marker);
   hypre_ParCSRMatrixMarkColumns(uP, marker);
   HYPRE_IJMatrixDestroy(ij_s);
   hypre_SStructGridMarkerBoxes(coarse, marker, ncols, location, &rows_J);

   /*------------------------------------------------------
    * Rows of P with columns in J (left), and rows of P reached from them
    * by A (right)
    *------------------------------------------------------*/

   left = hypre_TAlloc(hypre_BoxArray **, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pp      = hypre_SStructMatrixPMatrix(P, part);
      refined = hypre_BoxArrayClone(rows_J[part][0]);
      for (b = 0; b < hypre_BoxArraySize(refined); b++)
      {
         hypre_RefineBox(hypre_BoxArrayBox(refined, b), NULL,
                         hypre_SStructPMatrixDomainStride(pp));
      }
      left[part]    = hypre_TAlloc(hypre_BoxArray *, 1, HYPRE_MEMORY_HOST);
      left[part][0] = hypre_BoxArrayCreate(0, ndim);
      hypre_SStructStencilAppendCoupledBoxes(hypre_SStructPMatrixStencil(pp, 0),
                                             refined, left[part][0]);
      hypre_AppendBoxArray(pu_rows[part][0], left[part][0]);
      hypre_BoxArrayDestroy(refined);
   }
   hypre_SStructMatrixFreeSupport(fine, pu_rows);
   hypre_SStructGridRemoteBoxesShare(fine, rboxes, left);

   for (part = 0; part < nparts; part++)
   {
      for (b = 0; b < hypre_BoxArraySize(left[part][0]); b++)
      {
         hypre_CopyBox(hypre_BoxArrayBox(left[part][0], b), box);
         hypre_BoxGrowByArray(box, global_halo);
         hypre_AppendBox(box, right[part][0]);
      }
   }
   hypre_SStructGridRemoteBoxesShare(fine, rboxes, right);

   /*------------------------------------------------------
    * Compute the complete rows J of P^T*A*P
    *------------------------------------------------------*/

   hypre_SStructMatrixBoxesWithU(A, left, &ij_a);
   hypre_SStructMatrixBoxesWithU(P, right, &ij_p);
   hypre_SStructMatrixFreeSupport(fine, left);
   hypre_SStructMatrixFreeSupport(fine, right);

   a = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij_a);
   p = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij_p);
   hypre_ParCSRMatrixRestrictColumns(p, marker, &restricted);
   if (!hypre_ParCSRMatrixCommPkg(p))
   {
      hypre_MatvecCommPkgCreate(p);
   }
   rap = hypre_ParCSRMatrixRAPKT(restricted, a, p, 1, 0);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(restricted);
   HYPRE_IJMatrixDestroy(ij_a);
   HYPRE_IJMatrixDestroy(ij_p);
   hypre_TFree(marker, location);
   hypre_SStructMatrixFreeSupport(coarse, rows_J);
   hypre_SStructGridRemoteBoxesDestroy(rboxes);
   hypre_BoxDestroy(box);

   *result = rap;
   *supported = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the unstructured component of the product of SStructMatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultComputeU( hypre_SStructMatmultData *mmdata,
                              hypre_SStructMatrix      *M )
{
   HYPRE_Int                nmatrices = (mmdata -> nmatrices);
   HYPRE_Int                nterms    = (mmdata -> nterms);
   HYPRE_Int               *terms     = (mmdata -> terms);
   HYPRE_Int               *trans     = (mmdata -> transposes);
   hypre_SStructMatrix    **matrices  = (mmdata -> matrices);

   /* M matrix variables */
   hypre_IJMatrix          *ij_M;

   /* Temporary variables */
   hypre_ParCSRMatrix     **parcsr_tmp;
   hypre_ParCSRMatrix      *parcsr_sA;
   hypre_ParCSRMatrix      *parcsr_uA;
   hypre_ParCSRMatrix      *parcsr_sP;
   hypre_ParCSRMatrix      *parcsr_uP;
   hypre_ParCSRMatrix      *parcsr_sM = NULL;
   hypre_ParCSRMatrix      *parcsr_uM;
   hypre_ParCSRMatrix      *parcsr_uMold;
   hypre_ParCSRMatrix      *parcsr_sMold;
   hypre_IJMatrix          *ijmatrix;
   hypre_IJMatrix          *ij_tmp = NULL;
   hypre_IJMatrix         **ij_sA;

   HYPRE_Int                m, t;
   HYPRE_BigInt             num_nonzeros_uP;
   HYPRE_Int                urap2 = 0; /* Split unstructured RAP into 2 stages */

   HYPRE_Int                special_case_PTAP = 0;
   HYPRE_Int                boundary_PTAP     = 0;
   HYPRE_Int                compress_action   = 1; /* Add uM to sM where the pattern permits */
   HYPRE_Int                is_PTAP;

   is_PTAP = (nterms == 3 && matrices[terms[0]] == matrices[terms[2]] &&
              trans[0] && !trans[1] && !trans[2]);

   HYPRE_ANNOTATE_FUNC_BEGIN;


   if (is_PTAP)
   {
      m = terms[2];
      ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
      HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uP);
      num_nonzeros_uP = hypre_ParCSRMatrixNumNonzeros(parcsr_uP);

      /* The following ensures that NumNonzeros is initialized (a negative value
       * means uninitialized).  RDF: Not sure if this is needed for anything other
       * than matching the subsequent if test for the special PtAP case. */
      if (num_nonzeros_uP < 0)
      {
         hypre_ParCSRMatrixSetNumNonzeros(parcsr_uP);
         num_nonzeros_uP = hypre_ParCSRMatrixNumNonzeros(parcsr_uP);
      }
      if (num_nonzeros_uP == 0)
      {
         special_case_PTAP = 1;
      }
   }

   if (is_PTAP && !special_case_PTAP)
   {
      hypre_SStructMatrixPtAPBoundary(matrices[terms[1]], matrices[terms[2]],
                                      M, &parcsr_uM, &boundary_PTAP);
   }

   if (special_case_PTAP)
   {
      /* Specialization for PtAP when P has only the structured component. */
      m = terms[1];
      ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
      HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uA);

      /* Convert only the rows of sP that multiply entries of uA */
      m = terms[2];
      hypre_SStructMatrixHaloToUMatrix(matrices[m], matrices[terms[1]], &ij_tmp);
      HYPRE_IJMatrixGetObject(ij_tmp, (void **) &parcsr_sP);

      if (!hypre_ParCSRMatrixCommPkg(parcsr_sP))
      {
         hypre_MatvecCommPkgCreate(parcsr_sP);
      }

      /* Compute unstructured RAP */
      if (!urap2)
      {
         /* Note: this call does NOT add structural zeros to the diagonal */
         parcsr_uM = hypre_ParCSRMatrixRAPKT(parcsr_sP, parcsr_uA, parcsr_sP, 1, 0);
      }
      else
      {
         hypre_ParCSRMatrix *Q = hypre_ParCSRMatMat(parcsr_uA, parcsr_sP);
         parcsr_uM = hypre_ParCSRTMatMatKT(parcsr_sP, Q, 1);
         hypre_MatvecCommPkgCreate(parcsr_uM);
      }

      HYPRE_IJMatrixDestroy(ij_tmp);
   }
   else if (boundary_PTAP)
   {
      /* Specialization for PtAP when both A and P have unstructured components.
       * uM, computed by the call above, holds complete rows, so these rows
       * of sM are overwritten instead of added to. */
      compress_action = -1;  /* Sparse overwrite: retain S at entries absent from U */
   }
   else
   {
      /* Temporary work matrices */
      parcsr_tmp = hypre_TAlloc(hypre_ParCSRMatrix *, 3, HYPRE_MEMORY_HOST);
      ij_sA = hypre_TAlloc(hypre_IJMatrix *, nmatrices, HYPRE_MEMORY_HOST);
      for (m = 0; m < nmatrices; m++)
      {
         ij_sA[m] = NULL;
      }

      /* Set initial data */
      m = terms[nterms - 1];
      ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
      HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uMold);
      /* General products convert the whole structured component */
      ij_sA[m] = hypre_SStructMatrixToUMatrix(matrices[m], 0);

      HYPRE_IJMatrixGetObject(ij_sA[m], (void **) &parcsr_sMold);

#if defined(DEBUG_MATMULT)
      char matname[64];

      hypre_ParCSRMatrixPrintIJ(parcsr_uMold, 0, 0, "parcsr_uP");
      hypre_ParCSRMatrixPrintIJ(parcsr_sMold, 0, 0, "parcsr_sP");
#endif

      /* Compute uM iteratively */
      for (t = (nterms - 2); t >= 0; t--)
      {
         m = terms[t];

         /* Convert sA_n to IJMatrix */
         if (ij_sA[m] == NULL)
         {
            ij_sA[m] = hypre_SStructMatrixToUMatrix(matrices[m], 0);
         }
         HYPRE_IJMatrixGetObject(ij_sA[m], (void **) &parcsr_sA);
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_sA_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_sA, 0, 0, matname);
#endif

         /* 1) Compute sA_n*uMold */
         if (trans[t])
         {
#if defined(HYPRE_USING_GPU)
            if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_sA),
                                     hypre_ParCSRMatrixMemoryLocation(parcsr_uMold)) ==
                HYPRE_EXEC_DEVICE)
            {
               if (!hypre_ParCSRMatrixCommPkg(parcsr_sA))
               {
                  hypre_MatvecCommPkgCreate(parcsr_sA);
               }
               parcsr_tmp[0] = hypre_ParCSRTMatMat(parcsr_sA, parcsr_uMold);
            }
            else
#endif
            {
               parcsr_tmp[0] = hypre_ParTMatmul(parcsr_sA, parcsr_uMold);
            }
         }
         else
         {
#if defined(HYPRE_USING_GPU)
            if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_sA),
                                     hypre_ParCSRMatrixMemoryLocation(parcsr_uMold)) ==
                HYPRE_EXEC_DEVICE)
            {
               parcsr_tmp[0] = hypre_ParCSRMatMat(parcsr_sA, parcsr_uMold);
            }
            else
#endif
            {
               parcsr_tmp[0] = hypre_ParMatmul(parcsr_sA, parcsr_uMold);
            }
         }
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_0a_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_tmp[0], 0, 0, matname);
#endif

         /* 2) Compute uA_n*uMold */
         ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
         HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uA);
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_uA_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_uA, 0, 0, matname);
#endif
         if (trans[t])
         {
#if defined(HYPRE_USING_GPU)
            if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_uA),
                                     hypre_ParCSRMatrixMemoryLocation(parcsr_uMold)) ==
                HYPRE_EXEC_DEVICE)
            {
               if (!hypre_ParCSRMatrixCommPkg(parcsr_uA))
               {
                  hypre_MatvecCommPkgCreate(parcsr_uA);
               }
               parcsr_tmp[1] = hypre_ParCSRTMatMat(parcsr_uA, parcsr_uMold);
            }
            else
#endif
            {
               parcsr_tmp[1] = hypre_ParTMatmul(parcsr_uA, parcsr_uMold);
            }
         }
         else
         {
#if defined(HYPRE_USING_GPU)
            if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_uA),
                                     hypre_ParCSRMatrixMemoryLocation(parcsr_uMold)) ==
                HYPRE_EXEC_DEVICE)
            {
               parcsr_tmp[1] = hypre_ParCSRMatMat(parcsr_uA, parcsr_uMold);
            }
            else
#endif
            {
               parcsr_tmp[1] = hypre_ParMatmul(parcsr_uA, parcsr_uMold);
            }
         }
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_1_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_tmp[1], 0, 0, matname);
#endif

         if (t != (nterms - 2))
         {
            hypre_ParCSRMatrixDestroy(parcsr_uMold);
         }

         /* 3) Compute (sA_n*uMold + uA_n*uMold) */
         hypre_ParCSRMatrixAdd(1.0, parcsr_tmp[0], 1.0, parcsr_tmp[1], &parcsr_tmp[2]);
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_2_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_tmp[2], 0, 0, matname);
#endif

         /* Free sA_n*uMold and uA_n*uMold */
         hypre_ParCSRMatrixDestroy(parcsr_tmp[0]);
         hypre_ParCSRMatrixDestroy(parcsr_tmp[1]);

         /* 4) Compute uA_n*sMold */
         if (trans[t])
         {
#if defined(HYPRE_USING_GPU)
            if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_uA),
                                     hypre_ParCSRMatrixMemoryLocation(parcsr_sMold)) ==
                HYPRE_EXEC_DEVICE)
            {
               if (!hypre_ParCSRMatrixCommPkg(parcsr_uA))
               {
                  hypre_MatvecCommPkgCreate(parcsr_uA);
               }
               parcsr_tmp[0] = hypre_ParCSRTMatMat(parcsr_uA, parcsr_sMold);
            }
            else
#endif
            {
               parcsr_tmp[0] = hypre_ParTMatmul(parcsr_uA, parcsr_sMold);
            }
         }
         else
         {
#if defined(HYPRE_USING_GPU)
            if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_uA),
                                     hypre_ParCSRMatrixMemoryLocation(parcsr_sMold)) ==
                HYPRE_EXEC_DEVICE)
            {
               parcsr_tmp[0] = hypre_ParCSRMatMat(parcsr_uA, parcsr_sMold);
            }
            else
#endif
            {
               parcsr_tmp[0] = hypre_ParMatmul(parcsr_uA, parcsr_sMold);
            }
         }
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_0b_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_tmp[0], 0, 0, matname);
#endif

         /* 5) Compute (uA_n*uMold + sA_n*uMold + uA_n*uMold) */
         hypre_ParCSRMatrixAdd(1.0, parcsr_tmp[0], 1.0, parcsr_tmp[2], &parcsr_uM);
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_uM_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_uM, 0, 0, matname);
#endif

         /* Free temporary work matrices */
         hypre_ParCSRMatrixDestroy(parcsr_tmp[0]);
         hypre_ParCSRMatrixDestroy(parcsr_tmp[2]);

         /* Intermediate structured products feed the next U update. The final
          * structured product has already been computed by ComputeS. */
         parcsr_sM = NULL;
         if (t > 0)
         {
            if (trans[t])
            {
#if defined(HYPRE_USING_GPU)
               if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_sA),
                                        hypre_ParCSRMatrixMemoryLocation(parcsr_sMold)) ==
                   HYPRE_EXEC_DEVICE)
               {
                  if (!hypre_ParCSRMatrixCommPkg(parcsr_sA))
                  {
                     hypre_MatvecCommPkgCreate(parcsr_sA);
                  }
                  parcsr_sM = hypre_ParCSRTMatMat(parcsr_sA, parcsr_sMold);
               }
               else
#endif
               {
                  parcsr_sM = hypre_ParTMatmul(parcsr_sA, parcsr_sMold);
               }
            }
            else
            {
#if defined(HYPRE_USING_GPU)
               if (hypre_GetExecPolicy2(hypre_ParCSRMatrixMemoryLocation(parcsr_sA),
                                        hypre_ParCSRMatrixMemoryLocation(parcsr_sMold)) ==
                   HYPRE_EXEC_DEVICE)
               {
                  parcsr_sM = hypre_ParCSRMatMat(parcsr_sA, parcsr_sMold);
               }
               else
#endif
               {
                  parcsr_sM = hypre_ParMatmul(parcsr_sA, parcsr_sMold);
               }
            }
#if defined(DEBUG_MATMULT)
            hypre_sprintf(matname, "parcsr_sM_%d", t);
            hypre_ParCSRMatrixPrintIJ(parcsr_sM, 0, 0, matname);
#endif
         }

         if (t < (nterms - 2))
         {
            hypre_ParCSRMatrixDestroy(parcsr_sMold);
         }

         /* 7) Update pointers */
         parcsr_sMold = parcsr_sM;
         parcsr_uMold = parcsr_uM;
      }

      /* Free temporary work matrices */
      hypre_TFree(parcsr_tmp, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixDestroy(parcsr_sM);
      for (m = 0; m < nmatrices; m++)
      {
         if (ij_sA[m] != NULL)
         {
            HYPRE_IJMatrixDestroy(ij_sA[m]);
         }
      }
      hypre_TFree(ij_sA, HYPRE_MEMORY_HOST);
   }

   /* Update pointer to unstructured matrix component of M */
   ij_M = hypre_SStructMatrixIJMatrix(M);
   hypre_IJMatrixDestroyParCSR(ij_M);
   hypre_IJMatrixSetObject(ij_M, parcsr_uM);
   hypre_SStructMatrixParCSRMatrix(M) = parcsr_uM;
   hypre_IJMatrixAssembleFlag(ij_M) = 1;
   hypre_ParCSRMatrixSetNumNonzeros(parcsr_uM);

   /* Sparse overwrite has the same meaning in either compression backend. */
   hypre_SStructMatrixCompressUToS(M, compress_action);
   hypre_ParCSRMatrixSetNumNonzeros(parcsr_uM);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Iterative multiplication of SStructMatrices A_i (i=1,...,n) computed as
 *
 * M_1 = A_1           = (sA_1 + uA_1)
 * M_2 = A_2 * M_1     = (sA_2 + uA_2) * (sM_1 + uM_1)
 *                     = sA_2*sM_1 + (sA_2*uM_1 + uA_2 * (sM_1 + uM_1))
 *                          \.../                 \.../
 *                           \./                   \./
 *                            |                     |
 *                     =    sM_2   +              uM_2
 * ...
 * M_n = A_n * M_{n-1} = (sA_n + uA_n) * (sM_{n-1} + uM_{n-1})
 *                     = sA_n*sM_{n-1} + (sA_n*uM_{n-1} + uA_n * (sM_{n-1} + uM_{n-1}))
 *                           \.../                         \.../
 *                            \./                           \./
 *                             |                             |
 *                     =    sM_n       +                   uM_n
 *
 * Notes:
 *         1) A is transposed in each call to hypre_ParTMatmul. This operation
 *            could be done only once and At reused...
 *         2) Should we phase out domain grid and have only a base grid?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultCompute( hypre_SStructMatmultData *mmdata,
                             hypre_SStructMatrix      *M )
{
   /* Computes the structured component */
   hypre_SStructMatmultComputeS(mmdata, M);

   /* Computes the unstructured component */
   hypre_SStructMatmultComputeU(mmdata, M);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the product of several SStructMatrix matrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmult(HYPRE_Int             nmatrices,
                     hypre_SStructMatrix **matrices,
                     HYPRE_Int             nterms,
                     HYPRE_Int            *terms,
                     HYPRE_Int            *trans,
                     hypre_SStructMatrix **M_ptr )
{
   hypre_SStructMatmultData *mmdata;

   hypre_SStructMatmultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_SStructMatmultInitialize(mmdata, M_ptr);
   hypre_SStructMatmultCommunicate(mmdata);
   hypre_SStructMatmultCompute(mmdata, *M_ptr);
   hypre_SStructMatmultDestroy(mmdata);

   HYPRE_SStructMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes the product of two SStructMatrix matrices: M = A*B
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmat( hypre_SStructMatrix  *A,
                     hypre_SStructMatrix  *B,
                     hypre_SStructMatrix **M_ptr )
{
   HYPRE_Int            nmatrices   = 2;
   HYPRE_SStructMatrix  matrices[2] = {A, B};
   HYPRE_Int            nterms      = 2;
   HYPRE_Int            terms[3]    = {0, 1};
   HYPRE_Int            trans[2]    = {0, 0};

   hypre_SStructMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes M = P^T*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixPtAP( hypre_SStructMatrix  *A,
                         hypre_SStructMatrix  *P,
                         hypre_SStructMatrix **M_ptr )
{
   HYPRE_Int            nmatrices   = 2;
   HYPRE_SStructMatrix  matrices[2] = {A, P};
   HYPRE_Int            nterms      = 3;
   HYPRE_Int            terms[3]    = {1, 0, 1};
   HYPRE_Int            trans[3]    = {1, 0, 0};

   hypre_SStructMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes M = R*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixRAP( hypre_SStructMatrix  *R,
                        hypre_SStructMatrix  *A,
                        hypre_SStructMatrix  *P,
                        hypre_SStructMatrix **M_ptr )
{
   HYPRE_Int            nmatrices   = 3;
   HYPRE_SStructMatrix  matrices[3] = {A, P, R};
   HYPRE_Int            nterms      = 3;
   HYPRE_Int            terms[3]    = {2, 0, 1};
   HYPRE_Int            trans[3]    = {0, 0, 0};

   hypre_SStructMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Computes M = RT^T*A*P
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixRTtAP( hypre_SStructMatrix  *RT,
                          hypre_SStructMatrix  *A,
                          hypre_SStructMatrix  *P,
                          hypre_SStructMatrix **M_ptr )
{
   HYPRE_Int            nmatrices   = 3;
   HYPRE_SStructMatrix  matrices[3] = {A, P, RT};
   HYPRE_Int            nterms      = 3;
   HYPRE_Int            terms[3]    = {2, 0, 1};
   HYPRE_Int            trans[3]    = {1, 0, 0};

   hypre_SStructMatmult(nmatrices, matrices, nterms, terms, trans, M_ptr);

   return hypre_error_flag;
}
