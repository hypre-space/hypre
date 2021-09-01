/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "sstruct_matmult.h"

//#define DEBUG_MATMULT

/*==========================================================================
 * SStructPMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixMultCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixMultCreate(HYPRE_Int               nmatrices_input,
                               hypre_SStructPMatrix  **pmatrices_input,
                               HYPRE_Int               nterms,
                               HYPRE_Int              *terms_input,
                               HYPRE_Int              *trans_input,
                               hypre_SStructPMMData  **pmmdata_ptr)
{
   hypre_SStructPMMData   *pmmdata;
   hypre_StructMMData     *smmdata;

   hypre_SStructPMatrix  **pmatrices;
   hypre_StructMatrix    **smatrices;

   HYPRE_Int              *terms;
   HYPRE_Int              *trans;
   HYPRE_Int              *matmap;
   HYPRE_Int               nmatrices;
   HYPRE_Int               nvars;
   HYPRE_Int               m, t, vi, vj;

   pmmdata = hypre_CTAlloc(hypre_SStructPMMData, 1, HYPRE_MEMORY_HOST);

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
   smatrices = hypre_CTAlloc(hypre_StructMatrix *, nmatrices, HYPRE_MEMORY_HOST);
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

   /* Create SStructPMMData object */
   (pmmdata -> smmdata) = hypre_TAlloc(hypre_StructMMData **, nvars, HYPRE_MEMORY_HOST);

   /* TODO: This won't work for cases with inter-variable coupling */
   for (vi = 0; vi < nvars; vi++)
   {
      (pmmdata -> smmdata)[vi] = hypre_TAlloc(hypre_StructMMData *, nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         for (m = 0; m < nmatrices; m++)
         {
            smatrices[m] = hypre_SStructPMatrixSMatrix(pmatrices[m], vi, vj);
         }

         if (smatrices[0])
         {
            hypre_StructMatmultCreate(nmatrices, smatrices, nterms, terms, trans, &smmdata);
            (pmmdata -> smmdata)[vi][vj] = smmdata;
         }
         else
         {
            (pmmdata -> smmdata)[vi][vj] = NULL;
         }
      }
   }
   hypre_TFree(smatrices, HYPRE_MEMORY_HOST);

   /* Set SStructPMMData object */
   (pmmdata -> nterms)     = nterms;
   (pmmdata -> nmatrices)  = nmatrices;
   (pmmdata -> pmatrices)  = pmatrices;
   (pmmdata -> terms)      = terms;
   (pmmdata -> transposes) = trans;

   *pmmdata_ptr = pmmdata;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixMultDestroy
 *
 * Destroys an object of type hypre_SStructPMMData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixMultDestroy( hypre_SStructPMMData *pmmdata )
{
   HYPRE_Int vi, vj, nvars;

   if (pmmdata)
   {
      nvars = (pmmdata -> nvars);
      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            hypre_StructMatmultDestroy((pmmdata -> smmdata)[vi][vj]);
         }
         hypre_TFree(pmmdata -> smmdata[vi], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(pmmdata -> smmdata, HYPRE_MEMORY_HOST);

      hypre_TFree(pmmdata -> pmatrices, HYPRE_MEMORY_HOST);
      hypre_TFree(pmmdata -> transposes, HYPRE_MEMORY_HOST);
      hypre_TFree(pmmdata -> terms, HYPRE_MEMORY_HOST);

      hypre_TFree(pmmdata, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixMultSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixMultSetup( hypre_SStructPMMData   *pmmdata,
                               hypre_SStructPMatrix  **pM_ptr )
{
   hypre_StructMMData   ***smmdata = (pmmdata -> smmdata);
   HYPRE_Int               nvars   = (pmmdata -> nvars);
   hypre_SStructPMatrix   *pmatrix = pmmdata -> pmatrices[0];

   MPI_Comm                comm;
   HYPRE_Int               ndim;
   hypre_StructStencil    *stencil;
   hypre_Index            *offset;

   hypre_BoxArrayArray    *fpbnd_boxaa;
   hypre_BoxArrayArray    *cpbnd_boxaa;
   hypre_Index             origin;
   hypre_IndexRef          coarsen_stride;
   HYPRE_Int               coarsen;
   HYPRE_Int               num_boxes;
   hypre_BoxArray         *grid_boxes;

   HYPRE_SStructVariable  *vartp;
   hypre_SStructStencil  **pstencils;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructPGrid     *pfgrid;
   hypre_SStructPMatrix   *pM;
   hypre_StructMatrix     *sM;
   hypre_StructGrid       *sgrid;
   HYPRE_Int             **smaps;
   HYPRE_Int              *sentries;

   HYPRE_Int               vi, vj, e, cnt;
   HYPRE_Int               pstencil_size;
   HYPRE_Int               max_stencil_size;

   /* Initialize variables */
   ndim   = hypre_SStructPMatrixNDim(pmatrix);
   comm   = hypre_SStructPMatrixComm(pmatrix);
   pfgrid = hypre_SStructPMatrixPGrid(pmatrix); /* Same grid for all input matrices */
   vartp  = hypre_SStructPGridVarTypes(pfgrid);
   hypre_SetIndex(origin, 0);

   /* Create temporary semi-struct stencil data structure */
   pstencils = hypre_TAlloc(hypre_SStructStencil *, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      HYPRE_SStructStencilCreate(ndim, 0, &pstencils[vi]);
   }

   /* Create part grid data structure */
   hypre_SStructPGridCreate(comm, ndim, &pgrid);
   hypre_SStructPGridSetVariables(pgrid, nvars, vartp);

   /* Create part matrix data structure */
   hypre_SStructPMatrixCreate(comm, pgrid, pstencils, &pM);
   smaps = hypre_SStructPMatrixSMaps(pM);

   /* Setup part matrix data structure */
   max_stencil_size = cnt = 0;
   for (vi = 0; vi < nvars; vi++)
   {
      pstencil_size = 0;
      coarsen = 0;
      coarsen_stride = NULL;
      for (vj = 0; vj < nvars; vj++)
      {
         /* Check if this SMatrix exists */
         if (smmdata[vi][vj])
         {
            /* Destroy placeholder data */
            sM = hypre_SStructPMatrixSMatrix(pM, vi, vj);
            hypre_StructMatrixDestroy(sM);

            /* This sets up the grid and stencil of the (vi,vj)-block of the PMatrix */
            hypre_StructMatmultSetup(smmdata[vi][vj], &sM);
            hypre_SStructPMatrixSMatrix(pM, vi, vj) = sM;

            /* Update struct stencil of the (vi,vj)-block with actual stencils */
            stencil = hypre_StructMatrixStencil(sM);
            hypre_SStructPMatrixSStencil(pM, vi, vj) = hypre_StructStencilRef(stencil);

            /* Update the part stencil size */
            pstencil_size += hypre_StructStencilSize(stencil);

            /* Get coarsening information from the diagonal block */
            if (vi == vj)
            {
               coarsen = (smmdata[vi][vj] -> coarsen);
               coarsen_stride = (smmdata[vi][vj] -> coarsen_stride);
            }
         }
      }
      max_stencil_size = hypre_max(pstencil_size, max_stencil_size);

      /* Destroy placeholder grid and update with new StructGrid */
      sgrid = hypre_StructMatrixGrid(sM);
      hypre_SStructPGridSetSGrid(sgrid, pgrid, vi);

      /* Build part boundaries array */
      num_boxes   = hypre_StructGridNumBoxes(sgrid);
      grid_boxes  = hypre_StructGridBoxes(sgrid);
      fpbnd_boxaa = hypre_SStructPGridPBndBoxArrayArray(pfgrid, vi);
      if (num_boxes)
      {
         if (coarsen)
         {
            hypre_CoarsenBoxArrayArrayNeg(fpbnd_boxaa, grid_boxes, origin,
                                          coarsen_stride, &cpbnd_boxaa);
         }
         else
         {
            cpbnd_boxaa = hypre_BoxArrayArrayClone(fpbnd_boxaa);
         }
         hypre_SStructPGridPBndBoxArrayArray(pgrid, vi) = cpbnd_boxaa;
      }

      /* Update smaps array */
      smaps[vi] = hypre_TReAlloc(smaps[vi], HYPRE_Int, pstencil_size, HYPRE_MEMORY_HOST);

      /* Destroy placeholder semi-struct stencil and update with actual one */
      HYPRE_SStructStencilDestroy(pstencils[vi]);
      HYPRE_SStructStencilCreate(ndim, pstencil_size, &pstencils[vi]);
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

   /* Point to resulting matrix */
   *pM_ptr = pM;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixMultCommunicate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixMultCommunicate( hypre_SStructPMMData *pmmdata )
{
   HYPRE_Int                nvars   = (pmmdata -> nvars);
   hypre_StructMMData    ***smmdata = (pmmdata -> smmdata);

   HYPRE_Int               vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         if (smmdata[vi][vj])
         {
            hypre_StructMatmultCommunicate(smmdata[vi][vj]);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixMultCompute
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixMultCompute( hypre_SStructPMMData  *pmmdata,
                                 hypre_SStructPMatrix  *M )
{
   HYPRE_Int                nvars   = (pmmdata -> nvars);
   hypre_StructMMData    ***smmdata = (pmmdata -> smmdata);

   hypre_StructMatrix      *sM;
   HYPRE_Int                vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         /* This computes the coefficients of the (vi,vj)-block of the PMatrix */
         if (smmdata[vi][vj])
         {
            sM = hypre_SStructPMatrixSMatrix(M, vi, vj);
            hypre_StructMatmultCompute(smmdata[vi][vj], sM);
         }
      }
   }

   return hypre_error_flag;
}

/*==========================================================================
 * SStructMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixMultCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMultCreate(HYPRE_Int             nmatrices_input,
                              hypre_SStructMatrix **matrices_input,
                              HYPRE_Int             nterms,
                              HYPRE_Int            *terms_input,
                              HYPRE_Int            *trans_input,
                              hypre_SStructMMData **mmdata_ptr)
{
   hypre_SStructMMData    *mmdata;
   hypre_SStructPMMData   *pmmdata;

   hypre_SStructPMatrix  **pmatrices;
   hypre_SStructMatrix   **matrices;

   HYPRE_Int              *terms;
   HYPRE_Int              *trans;
   HYPRE_Int              *matmap;
   HYPRE_Int               nmatrices;
   HYPRE_Int               part, nparts;
   HYPRE_Int               m, t;

   mmdata = hypre_CTAlloc(hypre_SStructMMData, 1, HYPRE_MEMORY_HOST);

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

   /* Create SStructPMMData object */
   (mmdata -> pmmdata) = hypre_TAlloc(hypre_SStructPMMData *, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      for (m = 0; m < nmatrices; m++)
      {
         pmatrices[m] = hypre_SStructMatrixPMatrix(matrices[m], part);
      }
      hypre_SStructPMatrixMultCreate(nmatrices, pmatrices, nterms, terms, trans, &pmmdata);
      (mmdata -> pmmdata)[part] = pmmdata;
   }
   hypre_TFree(pmatrices, HYPRE_MEMORY_HOST);

   /* Set SStructMMData object */
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
 * hypre_SStructMatrixMultDestroy
 *
 * Destroys an object of type hypre_SStructMMData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMultDestroy( hypre_SStructMMData *mmdata )
{
   HYPRE_Int part, nparts;

   if (mmdata)
   {
      hypre_TFree(mmdata -> matrices, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> transposes, HYPRE_MEMORY_HOST);
      hypre_TFree(mmdata -> terms, HYPRE_MEMORY_HOST);

      nparts = (mmdata -> nparts);
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPMatrixMultDestroy((mmdata -> pmmdata)[part]);
      }
      hypre_TFree(mmdata -> pmmdata, HYPRE_MEMORY_HOST);

      hypre_CommPkgDestroy(mmdata -> comm_pkg);
      hypre_TFree(mmdata -> comm_data, HYPRE_MEMORY_HOST);

      hypre_TFree(mmdata, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixMultSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMultSetup( hypre_SStructMMData   *mmdata,
                              hypre_SStructMatrix  **M_ptr )
{
   HYPRE_Int                nparts   = (mmdata -> nparts);
   HYPRE_Int                nterms   = (mmdata -> nterms);
   HYPRE_Int               *terms    = (mmdata -> terms);
   HYPRE_Int               *trans    = (mmdata -> transposes);
   hypre_SStructPMMData   **pmmdata  = (mmdata -> pmmdata);
   hypre_SStructMatrix    **matrices = (mmdata -> matrices);

   /* M matrix variables */
   hypre_SStructMatrix     *M;
   hypre_SStructGrid       *Mgrid;
   hypre_SStructGraph      *Mgraph;
   hypre_SStructPMatrix    *pM;
   hypre_SStructStencil    *stencil;
   HYPRE_Int                stencil_size;
   HYPRE_Int                pstencil_size;
   HYPRE_Int                max_stencil_size;
   HYPRE_Int             ***splits;
   HYPRE_Int               *sentries;
   HYPRE_Int               *uentries;

   /* Unstructured component variables */
   hypre_IJMatrix          *ij_M;
   hypre_IJMatrix          *ijmatrix;
   HYPRE_Int                ilower, iupper;
   HYPRE_Int                jlower, jupper;

   /* Input matrices variables */
   hypre_SStructPMatrix    *pmatrix;
   HYPRE_SStructVariable   *vartypes;
   hypre_SStructPGrid      *pgrid;

   /* Communication variables */
   hypre_StructMMData      *smmdata;
   HYPRE_Int                np, num_comm_pkgs, num_comm_blocks;
   hypre_CommPkg          **comm_pkg_a;
   HYPRE_Complex         ***comm_data_a;
   HYPRE_Complex          **comm_data;

   /* Local variables */
   MPI_Comm                 comm;
   HYPRE_Int                ndim;
   HYPRE_Int                part;
   HYPRE_Int                i, vi, vj, nvars;

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
      hypre_SStructPMatrixMultSetup(pmmdata[part], &pM);

      /* Update part matrix of M */
      hypre_SStructMatrixPMatrix(M, part) = pM;

      /* Update part grid of M */
      hypre_SStructPGridDestroy(pgrid);
      pgrid = hypre_SStructPMatrixPGrid(pM);
      hypre_SStructGridPGrid(Mgrid, part) = pgrid;

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
      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smmdata = pmmdata[part] -> smmdata[vi][vj];
            if (smmdata)
            {
               num_comm_pkgs   += (smmdata -> num_comm_pkgs);
               num_comm_blocks += (smmdata -> num_comm_blocks);
            }
         }
      }
   }
   (mmdata -> num_comm_pkgs)   = num_comm_pkgs;
   (mmdata -> num_comm_blocks) = num_comm_blocks;

   /* Allocate communication packages and data */
   comm_pkg_a  = hypre_TAlloc(hypre_CommPkg *, num_comm_pkgs, HYPRE_MEMORY_HOST);
   comm_data_a = hypre_TAlloc(HYPRE_Complex **, num_comm_pkgs, HYPRE_MEMORY_HOST);
   comm_data   = hypre_TAlloc(HYPRE_Complex *, num_comm_blocks, HYPRE_MEMORY_HOST);
   (mmdata -> comm_pkg_a)  = comm_pkg_a;
   (mmdata -> comm_data_a) = comm_data_a;
   (mmdata -> comm_data)   = comm_data;

   /* Update pointers to communication packages and data */
   num_comm_pkgs = num_comm_blocks = 0;
   for (part = 0; part < nparts; part++)
   {
      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smmdata = pmmdata[part] -> smmdata[vi][vj];
            if (smmdata)
            {
               for (np = 0; np < (smmdata -> num_comm_pkgs); np++)
               {
                  comm_pkg_a[num_comm_pkgs]  = smmdata -> comm_pkg_a[np];
                  comm_data_a[num_comm_pkgs] = smmdata -> comm_data_a[np];
                  num_comm_pkgs++;
               }

               hypre_TFree(smmdata -> comm_pkg_a, HYPRE_MEMORY_HOST);
               hypre_TFree(smmdata -> comm_data_a, HYPRE_MEMORY_HOST);
            }
         }
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
   ijmatrix = hypre_SStructMatrixIJMatrix(matrices[terms[nterms-1]]);
   if (trans[nterms-1])
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
 * hypre_SStructMatrixMultCommunicate
 *
 * Run communication phase for computing the structured component of M
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMultCommunicate( hypre_SStructMMData *mmdata )
{
   hypre_CommPkg           *comm_pkg      = (mmdata -> comm_pkg);
   HYPRE_Complex          **comm_data     = (mmdata -> comm_data);
   hypre_CommPkg          **comm_pkg_a    = (mmdata -> comm_pkg_a);
   HYPRE_Complex         ***comm_data_a   = (mmdata -> comm_data_a);
   HYPRE_Int                num_comm_pkgs = (mmdata -> num_comm_pkgs);

   hypre_CommHandle        *comm_handle;
   HYPRE_Int                i, j, nb;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (num_comm_pkgs > 0)
   {
      /* Agglomerate communication packages if needed */
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "CommSetup");
      if (!comm_pkg)
      {
         hypre_CommPkgAgglomerate(num_comm_pkgs, comm_pkg_a, &comm_pkg);

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

         mmdata -> comm_pkg  = comm_pkg;
         mmdata -> comm_data = comm_data;
      }
      HYPRE_ANNOTATE_REGION_END("%s", "CommSetup");

      hypre_InitializeCommunication(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
      hypre_FinalizeCommunication(comm_handle);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixMultComputeS
 *
 * Computes the structured component of the product of SStructMatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMultComputeS( hypre_SStructMMData  *mmdata,
                                 hypre_SStructMatrix  *M )
{
   HYPRE_Int                nparts  = (mmdata -> nparts);
   hypre_SStructPMMData   **pmmdata = (mmdata -> pmmdata);

   hypre_SStructPMatrix    *pM;
   HYPRE_Int                part;

   for (part = 0; part < nparts; part++)
   {
      pM = hypre_SStructMatrixPMatrix(M, part);
      hypre_SStructPMatrixMultCompute(pmmdata[part], pM);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixMultComputeU
 *
 * Computes the unstructured component of the product of SStructMatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMultComputeU( hypre_SStructMMData *mmdata,
                                 hypre_SStructMatrix *M )
{
   HYPRE_Int                nmatrices = (mmdata -> nmatrices);
   HYPRE_Int                nterms    = (mmdata -> nterms);
   HYPRE_Int               *terms     = (mmdata -> terms);
   HYPRE_Int               *trans     = (mmdata -> transposes);
   hypre_SStructMatrix    **matrices  = (mmdata -> matrices);

   /* M matrix variables */
   hypre_IJMatrix          *ij_M;

   /* Temporary variables */
   hypre_ParCSRMatrix     **parcsr;
   hypre_ParCSRMatrix      *parcsr_sA;
   hypre_ParCSRMatrix      *parcsr_uA;
   hypre_ParCSRMatrix      *parcsr_sM;
   hypre_ParCSRMatrix      *parcsr_uM;
   hypre_ParCSRMatrix      *parcsr_uMold;
   hypre_ParCSRMatrix      *parcsr_sMold;
   hypre_IJMatrix          *ijmatrix;
   hypre_IJMatrix         **ij_sA;

   HYPRE_Int                m, t;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Temporary work matrices */
   parcsr = hypre_TAlloc(hypre_ParCSRMatrix *, 3, HYPRE_MEMORY_HOST);
   ij_sA  = hypre_TAlloc(hypre_IJMatrix *, nmatrices, HYPRE_MEMORY_HOST);
   for (m = 0; m < nmatrices; m++)
   {
      ij_sA[m] = NULL;
   }

   /* Set initial data */
   m = terms[nterms - 2];
   ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
   HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uM);

   m = terms[nterms - 1];
   ijmatrix = hypre_SStructMatrixIJMatrix(matrices[m]);
   HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uMold);
   hypre_SStructMatrixBoundaryToUMatrix(matrices[m], parcsr_uM, &ij_sA[m]);
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
         hypre_SStructMatrixBoundaryToUMatrix(matrices[m], parcsr_uMold, &ij_sA[m]);
      }
      HYPRE_IJMatrixGetObject(ij_sA[m], (void **) &parcsr_sA);
#if defined(DEBUG_MATMULT)
      hypre_sprintf(matname, "parcsr_sA_%d", t);
      hypre_ParCSRMatrixPrintIJ(parcsr_sA, 0, 0, matname);
#endif

      /* 1) Compute sA_n*uMold */
      if (trans[t])
      {
         parcsr[0] = hypre_ParTMatmul(parcsr_sA, parcsr_uMold);
      }
      else
      {
         parcsr[0] = hypre_ParMatmul(parcsr_sA, parcsr_uMold);
      }
#if defined(DEBUG_MATMULT)
      hypre_sprintf(matname, "parcsr_0a_%d", t);
      hypre_ParCSRMatrixPrintIJ(parcsr[0], 0, 0, matname);
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
         parcsr[1] = hypre_ParTMatmul(parcsr_uA, parcsr_uMold);
      }
      else
      {
         parcsr[1] = hypre_ParMatmul(parcsr_uA, parcsr_uMold);
      }
#if defined(DEBUG_MATMULT)
      hypre_sprintf(matname, "parcsr_1_%d", t);
      hypre_ParCSRMatrixPrintIJ(parcsr[1], 0, 0, matname);
#endif

      if (t != (nterms - 2))
      {
         hypre_ParCSRMatrixDestroy(parcsr_uMold);
      }

      /* 3) Compute (sA_n*uMold + uA_n*uMold) */
      hypre_ParCSRMatrixAdd(1.0, parcsr[0], 1.0, parcsr[1], &parcsr[2]);
#if defined(DEBUG_MATMULT)
      hypre_sprintf(matname, "parcsr_2_%d", t);
      hypre_ParCSRMatrixPrintIJ(parcsr[2], 0, 0, matname);
#endif

      /* Free sA_n*uMold and uA_n*uMold */
      hypre_ParCSRMatrixDestroy(parcsr[0]);
      hypre_ParCSRMatrixDestroy(parcsr[1]);

      /* 4) Compute uA_n*sMold */
      if (trans[t])
      {
         parcsr[0] = hypre_ParTMatmul(parcsr_uA, parcsr_sMold);
      }
      else
      {
         parcsr[0] = hypre_ParMatmul(parcsr_uA, parcsr_sMold);
      }
#if defined(DEBUG_MATMULT)
      hypre_sprintf(matname, "parcsr_0b_%d", t);
      hypre_ParCSRMatrixPrintIJ(parcsr[0], 0, 0, matname);
#endif

      /* 5) Compute (uA_n*uMold + sA_n*uMold + uA_n*uMold) */
      hypre_ParCSRMatrixAdd(1.0, parcsr[0], 1.0, parcsr[2], &parcsr_uM);
#if defined(DEBUG_MATMULT)
      hypre_sprintf(matname, "parcsr_uM_%d", t);
      hypre_ParCSRMatrixPrintIJ(parcsr_uM, 0, 0, matname);
#endif

      /* Free temporary work matrices */
      hypre_ParCSRMatrixDestroy(parcsr[0]);
      hypre_ParCSRMatrixDestroy(parcsr[2]);

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
   hypre_TFree(parcsr, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixDestroy(parcsr_sM);
   for (m = 0; m < nmatrices; m++)
   {
      if (ij_sA[m] != NULL)
      {
         HYPRE_IJMatrixDestroy(ij_sA[m]);
      }
   }
   hypre_TFree(ij_sA, HYPRE_MEMORY_HOST);

   /* Update pointer to unstructured matrix component of M */
   ij_M = hypre_SStructMatrixIJMatrix(M);
   hypre_IJMatrixDestroyParCSR(ij_M);
   hypre_IJMatrixSetObject(ij_M, parcsr_uM);
   hypre_IJMatrixAssembleFlag(ij_M) = 1;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixMultCompute
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
hypre_SStructMatrixMultCompute( hypre_SStructMMData *mmdata,
                                hypre_SStructMatrix *M )
{
   /* Computes the structured component */
   hypre_SStructMatrixMultComputeS(mmdata, M);

   /* Computes the unstructured component */
   hypre_SStructMatrixMultComputeU(mmdata, M);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixMultGroup
 *
 * Computes the product of a group of SStructMatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMultGroup(HYPRE_Int             nmatrices,
                             hypre_SStructMatrix **matrices,
                             HYPRE_Int             nterms,
                             HYPRE_Int            *terms,
                             HYPRE_Int            *trans,
                             hypre_SStructMatrix **M_ptr )
{
   hypre_SStructMMData *mmdata;

   hypre_SStructMatrixMultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_SStructMatrixMultSetup(mmdata, M_ptr);
   hypre_SStructMatrixMultCommunicate(mmdata);
   hypre_SStructMatrixMultCompute(mmdata, *M_ptr);
   hypre_SStructMatrixMultDestroy(mmdata);

   HYPRE_SStructMatrixAssemble(*M_ptr);

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
   hypre_SStructMMData *mmdata;
   hypre_SStructMatrix *M;

   HYPRE_Int            nmatrices   = 2;
   HYPRE_SStructMatrix  matrices[2] = {A, P};
   HYPRE_Int            nterms      = 3;
   HYPRE_Int            terms[3]    = {1, 0, 1};
   HYPRE_Int            trans[3]    = {1, 0, 0};

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Compute resulting matrix M */
   hypre_SStructMatrixMultCreate(nmatrices, matrices, nterms, terms, trans, &mmdata);
   hypre_SStructMatrixMultSetup(mmdata, &M);
   hypre_SStructMatrixMultCommunicate(mmdata);
   hypre_SStructMatrixMultCompute(mmdata, M);
   hypre_SStructMatrixMultDestroy(mmdata);

   /* Assemble matrix M */
   HYPRE_SStructMatrixAssemble(M);

   /* Point to resulting matrix */
   *M_ptr = M;

   HYPRE_ANNOTATE_FUNC_END;

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
// TODO
   return hypre_error_flag;
}
