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
 * hypre_SStructPMatmultCreate
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

   max_matmults = (nterms - 1) * hypre_pow(nvars, (nterms + 1));
   max_matrices = nmatrices * nvars * nvars;
   hypre_StructMatmultCreate(max_matmults, max_matrices, &smmdata);
   smmid = hypre_TAlloc(HYPRE_Int **, nvars, HYPRE_MEMORY_HOST);
   smmsz = hypre_TAlloc(HYPRE_Int * , nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smmid[vi] = hypre_TAlloc(HYPRE_Int *, nvars, HYPRE_MEMORY_HOST);
      smmsz[vi] = hypre_TAlloc(HYPRE_Int  , nvars, HYPRE_MEMORY_HOST);
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
               if (trans[k])
               {
                  /* Use the transpose matrix (reverse the indices) */
                  smatrices[k] = hypre_SStructPMatrixSMatrix(pmatrices[terms[k]], i[k + 1], i[k]);
               }
               else
               {
                  smatrices[k] = hypre_SStructPMatrixSMatrix(pmatrices[terms[k]], i[k], i[k + 1]);
               }
               if (smatrices[k] == NULL)
               {
                  zero_product = 1;
                  break;
               }
            }
            if (!zero_product)
            {
               hypre_StructMatmultSetup(smmdata, nterms, smatrices, nterms, sterms, trans,
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
   (pmmdata -> comm_pkg_a) = NULL;
   (pmmdata -> comm_data)  = NULL;
   (pmmdata -> comm_data_a) = NULL;
   (pmmdata -> num_comm_pkgs)   = 0;
   (pmmdata -> num_comm_blocks) = 0;

   *pmmdata_ptr = pmmdata;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatmultDestroy
 *
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
 * hypre_SStructPMatmultSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultSetup( hypre_SStructPMatmultData  *pmmdata,
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

   HYPRE_Int                    vi, vj, e, cnt, k;
   HYPRE_Int                    pstencil_size;
   HYPRE_Int                    max_stencil_size;

   /* Initialize variables */
   ndim      = hypre_SStructPMatrixNDim(pmatrix);
   comm      = hypre_SStructPMatrixComm(pmatrix);
   pfgrid    = hypre_SStructPMatrixPGrid(pmatrix); /* Same grid for all input matrices */
   vartypes  = hypre_SStructPGridVarTypes(pfgrid);
   hypre_SetIndex(origin, 0);

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
   /* NOTE: This does not assemble the struct grids.  They are assembled below
    * or in HYPRE_SStructGridAssemble(Mgrid) to reduce box manager overhead.
    * It's not clear if this is useful anymore. */
   hypre_StructMatmultInit(smmdata, 0);

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
            hypre_StructMatrix  **smatrices;

            /* Destroy the default struct matrix for the (vi,vj)-block of the PMatrix */
            sM = hypre_SStructPMatrixSMatrix(pM, vi, vj);
            hypre_StructMatrixDestroy(sM);

            /* Replace the default struct matrix for the (vi,vj)-block of the PMatrix */
            smatrices = hypre_CTAlloc(hypre_StructMatrix *, smmsz[vi][vj], HYPRE_MEMORY_HOST);
            for (k = 0; k < smmsz[vi][vj]; k++)
            {
               hypre_StructMatmultGetMatrix(smmdata, smmid[vi][vj][k], &smatrices[k]);
            }
            hypre_StructMatrixAddInit(smmsz[vi][vj], smatrices, &sM);
            hypre_SStructPMatrixSMatrix(pM, vi, vj) = sM;
            hypre_TFree(smatrices, HYPRE_MEMORY_HOST);

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
                     hypre_CoarsenBoxArrayArrayNeg(fpbnd_boxaa, grid_boxes, origin,
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

   /* Point to the smmdata communication fields (RDF: Remove later, it's redundant ) */
   (pmmdata -> comm_pkg_a)      = (smmdata -> comm_pkg_a);
   (pmmdata -> comm_data_a)     = (smmdata -> comm_data_a);
   (pmmdata -> num_comm_pkgs)   = (smmdata -> num_comm_pkgs);
   (pmmdata -> num_comm_blocks) = (smmdata -> num_comm_blocks);

   /* Point to resulting matrix */
   *pM_ptr = pM;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatmultCommunicate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultCommunicate( hypre_SStructPMatmultData *pmmdata )
{
   hypre_CommPkg           *comm_pkg      = (pmmdata -> comm_pkg);
   HYPRE_Complex          **comm_data     = (pmmdata -> comm_data);
   hypre_CommPkg          **comm_pkg_a    = (pmmdata -> comm_pkg_a);
   HYPRE_Complex         ***comm_data_a   = (pmmdata -> comm_data_a);
   HYPRE_Int                num_comm_pkgs = (pmmdata -> num_comm_pkgs);
   hypre_CommHandle        *comm_handle;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* RDF: We could just call hypre_StructMatmultCommunicate() here */

   if (num_comm_pkgs > 0)
   {
      /* Agglomerate communication packages if needed */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "CommSetup");
      if (!comm_pkg)
      {
         hypre_CommPkgAgglomerate(num_comm_pkgs, comm_pkg_a, &comm_pkg);
         hypre_CommPkgAgglomData(num_comm_pkgs, comm_pkg_a, comm_data_a, comm_pkg, &comm_data);
         hypre_CommPkgAgglomDestroy(num_comm_pkgs, comm_pkg_a, comm_data_a);
         (pmmdata -> comm_pkg_a)  = NULL;
         (pmmdata -> comm_data_a) = NULL;
         (pmmdata -> comm_pkg)    = comm_pkg;
         (pmmdata -> comm_data)   = comm_data;
      }
      HYPRE_ANNOTATE_REGION_END("%s", "CommSetup");

      hypre_InitializeCommunication(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
      hypre_FinalizeCommunication(comm_handle);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatmultCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatmultCompute( hypre_SStructPMatmultData *pmmdata,
                              hypre_SStructPMatrix      *pM )
{
   hypre_StructMatmultData    *smmdata = (pmmdata -> smmdata);
   HYPRE_Int                ***smmid   = (pmmdata -> smmid);
   HYPRE_Int                 **smmsz   = (pmmdata -> smmsz);
   HYPRE_Int                   nvars   = (pmmdata -> nvars);

   hypre_StructMatrix         *sM, *sMk;
   HYPRE_Int                   vi, vj, k;

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         /* This computes the coefficients of the (vi,vj)-block of the PMatrix */
         sM = hypre_SStructPMatrixSMatrix(pM, vi, vj);
         for (k = 0; k < smmsz[vi][vj]; k++)
         {
            hypre_StructMatmultCompute(smmdata, smmid[vi][vj][k]);
            hypre_StructMatmultGetMatrix(smmdata, smmid[vi][vj][k], &sMk);
            hypre_StructMatrixAddMat(sM, 1.0, sMk);  /* Compute sM += alpha * sMk */
            hypre_StructMatrixDestroy(sMk);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatmult
 *
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
   hypre_SStructPMatmultSetup(mmdata, 1, M_ptr); /* Make sure to assemble the grid */
   hypre_SStructPMatmultCommunicate(mmdata);
   hypre_SStructPMatmultCompute(mmdata, *M_ptr);
   hypre_SStructPMatmultDestroy(mmdata);

   hypre_SStructPMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatmat
 *
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
 * hypre_SStructPMatrixPtAP
 *
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
 * hypre_SStructPMatrixRAP
 *
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
 * hypre_SStructPMatrixRTtAP
 *
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
 * hypre_SStructMatmultCreate
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
   (mmdata -> comm_pkg_a)  = NULL;
   (mmdata -> comm_data)   = NULL;
   (mmdata -> comm_data_a) = NULL;
   (mmdata -> num_comm_pkgs)   = 0;
   (mmdata -> num_comm_blocks) = 0;

   *mmdata_ptr = mmdata;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatmultDestroy
 *
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
 * hypre_SStructMatmultSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultSetup( hypre_SStructMatmultData   *mmdata,
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
   HYPRE_Int                   ilower, iupper;
   HYPRE_Int                   jlower, jupper;

   /* Input matrices variables */
   hypre_SStructPMatrix       *pmatrix;
   HYPRE_SStructVariable      *vartypes;
   hypre_SStructPGrid         *pgrid;

   /* Communication variables */
   HYPRE_Int                   np, num_comm_pkgs, num_comm_blocks;
   hypre_CommPkg             **comm_pkg_a;
   HYPRE_Complex            ***comm_data_a;

   /* Local variables */
   MPI_Comm                    comm;
   HYPRE_Int                   ndim;
   HYPRE_Int                   part;
   HYPRE_Int                   i, vi, vj, nvars;

   /* TODO: sanity check for input matrices */

   /* Initialize variables */
   comm   = hypre_SStructMatrixComm(matrices[0]);
   ndim   = hypre_SStructMatrixNDim(matrices[0]);
   nparts = hypre_SStructMatrixNParts(matrices[0]);

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
      hypre_SStructPMatmultSetup(pmmdata[part], 0, &pM); /* Don't assemble the grid */

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

   /* Find total number of communication packages and blocks */
   num_comm_pkgs = num_comm_blocks = 0;
   for (part = 0; part < nparts; part++)
   {
      if (pmmdata[part])
      {
         num_comm_pkgs   += (pmmdata[part] -> num_comm_pkgs);
         num_comm_blocks += (pmmdata[part] -> num_comm_blocks);
      }
   }
   (mmdata -> num_comm_pkgs)   = num_comm_pkgs;
   (mmdata -> num_comm_blocks) = num_comm_blocks;

   /* Allocate communication packages and data */
   comm_pkg_a  = hypre_TAlloc(hypre_CommPkg *, num_comm_pkgs, HYPRE_MEMORY_HOST);
   comm_data_a = hypre_TAlloc(HYPRE_Complex **, num_comm_pkgs, HYPRE_MEMORY_HOST);
   (mmdata -> comm_pkg_a)  = comm_pkg_a;
   (mmdata -> comm_data_a) = comm_data_a;

   /* Update pointers to communication packages and data */
   num_comm_pkgs = num_comm_blocks = 0;
   for (part = 0; part < nparts; part++)
   {
      if (pmmdata[part])
      {
         for (np = 0; np < (pmmdata[part] -> num_comm_pkgs); np++)
         {
            comm_pkg_a[num_comm_pkgs]  = (pmmdata[part] -> comm_pkg_a[np]);
            comm_data_a[num_comm_pkgs] = (pmmdata[part] -> comm_data_a[np]);
            num_comm_pkgs++;
         }

         hypre_TFree(pmmdata[part] -> comm_pkg_a, HYPRE_MEMORY_HOST);
         hypre_TFree(pmmdata[part] -> comm_data_a, HYPRE_MEMORY_HOST);
      }
   }

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
 * hypre_SStructMatmultCommunicate
 *
 * Run communication phase for computing the structured component of M
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatmultCommunicate( hypre_SStructMatmultData *mmdata )
{
   hypre_CommPkg           *comm_pkg      = (mmdata -> comm_pkg);
   HYPRE_Complex          **comm_data     = (mmdata -> comm_data);
   hypre_CommPkg          **comm_pkg_a    = (mmdata -> comm_pkg_a);
   HYPRE_Complex         ***comm_data_a   = (mmdata -> comm_data_a);
   HYPRE_Int                num_comm_pkgs = (mmdata -> num_comm_pkgs);

   hypre_CommHandle        *comm_handle;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (num_comm_pkgs > 0)
   {
      /* Agglomerate communication packages if needed */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "CommSetup");
      if (!comm_pkg)
      {
         hypre_CommPkgAgglomerate(num_comm_pkgs, comm_pkg_a, &comm_pkg);
         hypre_CommPkgAgglomData(num_comm_pkgs, comm_pkg_a, comm_data_a, comm_pkg, &comm_data);
         hypre_CommPkgAgglomDestroy(num_comm_pkgs, comm_pkg_a, comm_data_a);
         (mmdata -> comm_pkg_a)  = NULL;
         (mmdata -> comm_data_a) = NULL;
         (mmdata -> comm_pkg)    = comm_pkg;
         (mmdata -> comm_data)   = comm_data;
      }
      HYPRE_ANNOTATE_REGION_END("%s", "CommSetup");

      hypre_InitializeCommunication(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
      hypre_FinalizeCommunication(comm_handle);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatmultComputeS
 *
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
 * hypre_SStructMatmultComputeU
 *
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
   hypre_SStructGraph      *graph;
   hypre_SStructGrid       *grid;
   hypre_ParCSRMatrix     **parcsr_tmp;
   hypre_ParCSRMatrix      *parcsr_sA;
   hypre_ParCSRMatrix      *parcsr_uA;
   hypre_ParCSRMatrix      *parcsr_sP;
   hypre_ParCSRMatrix      *parcsr_uP;
   hypre_ParCSRMatrix      *parcsr_sM;
   hypre_ParCSRMatrix      *parcsr_uM;
   hypre_ParCSRMatrix      *parcsr_uMold;
   hypre_ParCSRMatrix      *parcsr_sMold;
   hypre_IJMatrix          *ijmatrix;
   hypre_IJMatrix          *ij_tmp = NULL;
   hypre_IJMatrix         **ij_sA;

   HYPRE_Int                m, t;
   HYPRE_Int                num_nonzeros_uP;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   m = terms[2];
   ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
   HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uP);
   num_nonzeros_uP = hypre_ParCSRMatrixNumNonzeros(parcsr_uP);
   if (nterms == 3 && (num_nonzeros_uP == 0))
   {
      /* Specialization for RAP when P has only the structured component */
      m = terms[1];
      graph = hypre_SStructMatrixGraph(matrices[m]);
      grid  = hypre_SStructGraphGrid(graph);
      ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
      HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_sA);

      m = terms[2];
      hypre_SStructMatrixHaloToUMatrix(matrices[m], grid, &ij_tmp, 2);
      HYPRE_IJMatrixGetObject(ij_tmp, (void **) &parcsr_sP);

      if (!hypre_ParCSRMatrixCommPkg(parcsr_sP))
      {
         hypre_MatvecCommPkgCreate(parcsr_sP);
      }

      //hypre_BoomerAMGBuildCoarseOperator(parcsr_sP, parcsr_sA, parcsr_sP, &parcsr_uM);
      parcsr_uM = hypre_ParCSRMatrixRAP(parcsr_sP, parcsr_sA, parcsr_sP);

      HYPRE_IJMatrixDestroy(ij_tmp);
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
      m = terms[nterms - 2];
      graph = hypre_SStructMatrixGraph(matrices[m]);
      grid  = hypre_SStructGraphGrid(graph);

      m = terms[nterms - 1];
      ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
      HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uMold);
      /* WM: todo - converting the whole matrix for now to be safe... */
      /* hypre_SStructMatrixHaloToUMatrix(matrices[m], grid, &ij_sA[m], 2); */
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
            graph = hypre_SStructMatrixGraph(matrices[terms[t + 1]]);
            grid  = hypre_SStructGraphGrid(graph);

            /* WM: todo - converting the whole matrix for now to be safe... */
            /* hypre_SStructMatrixHaloToUMatrix(matrices[m], grid, &ij_sA[m], 2); */
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
            parcsr_tmp[0] = hypre_ParTMatmul(parcsr_sA, parcsr_uMold);
         }
         else
         {
            parcsr_tmp[0] = hypre_ParMatmul(parcsr_sA, parcsr_uMold);
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
            parcsr_tmp[1] = hypre_ParTMatmul(parcsr_uA, parcsr_uMold);
         }
         else
         {
            parcsr_tmp[1] = hypre_ParMatmul(parcsr_uA, parcsr_uMold);
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
            parcsr_tmp[0] = hypre_ParTMatmul(parcsr_uA, parcsr_sMold);
         }
         else
         {
            parcsr_tmp[0] = hypre_ParMatmul(parcsr_uA, parcsr_sMold);
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

         /* 6) Compute sA_n*sMold */
         if (trans[t])
         {
            parcsr_sM = hypre_ParTMatmul(parcsr_sA, parcsr_sMold);
         }
         else
         {
            parcsr_sM = hypre_ParMatmul(parcsr_sA, parcsr_sMold);
         }
#if defined(DEBUG_MATMULT)
         hypre_sprintf(matname, "parcsr_sM_%d", t);
         hypre_ParCSRMatrixPrintIJ(parcsr_sM, 0, 0, matname);
#endif

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

   /* WM: extra delete zeros... I'm getting zero diagonals everywhere in the U matrices? */
   hypre_CSRMatrix *delete_zeros = hypre_CSRMatrixDeleteZeros(hypre_ParCSRMatrixDiag(parcsr_uM),
                                                              HYPRE_REAL_MIN);
   if (delete_zeros)
   {
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(parcsr_uM));
      hypre_ParCSRMatrixDiag(parcsr_uM) = delete_zeros;
      hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixDiag(parcsr_uM));
   }
   delete_zeros = hypre_CSRMatrixDeleteZeros(hypre_ParCSRMatrixOffd(parcsr_uM), HYPRE_REAL_MIN);
   if (delete_zeros)
   {
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(parcsr_uM));
      hypre_ParCSRMatrixOffd(parcsr_uM) = delete_zeros;
      hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixOffd(parcsr_uM));
   }
   hypre_ParCSRMatrixSetNumNonzeros(parcsr_uM);

   hypre_SStructMatrixCompressUToS(M, 1);

   /* WM: should I do this here? What tolerance? Smarter way to avoid a bunch of zero entries? */
   /*     note that once I'm not converting the entire struct matrix, most of these should go away, I think... */
   /* WM: todo - CAREFUL HERE! This can screw things up if you throw away entries that are actually non-trivial and should be there... */
   delete_zeros = hypre_CSRMatrixDeleteZeros(hypre_ParCSRMatrixDiag(parcsr_uM), HYPRE_REAL_MIN);
   if (delete_zeros)
   {
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(parcsr_uM));
      hypre_ParCSRMatrixDiag(parcsr_uM) = delete_zeros;
      hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixDiag(parcsr_uM));
   }
   delete_zeros = hypre_CSRMatrixDeleteZeros(hypre_ParCSRMatrixOffd(parcsr_uM), HYPRE_REAL_MIN);
   if (delete_zeros)
   {
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(parcsr_uM));
      hypre_ParCSRMatrixOffd(parcsr_uM) = delete_zeros;
      hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixOffd(parcsr_uM));
   }
   hypre_ParCSRMatrixSetNumNonzeros(parcsr_uM);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatmultCompute
 *
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
 * hypre_SStructMatmult
 *
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
   hypre_SStructMatmultSetup(mmdata, M_ptr);
   hypre_SStructMatmultCommunicate(mmdata);
   hypre_SStructMatmultCompute(mmdata, *M_ptr);
   hypre_SStructMatmultDestroy(mmdata);

   HYPRE_SStructMatrixAssemble(*M_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatmat
 *
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
 * hypre_SStructMatrixPtAP
 *
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
 * hypre_SStructMatrixRAP
 *
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
 * hypre_SStructMatrixRTtAP
 *
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
