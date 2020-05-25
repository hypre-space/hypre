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

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_SStructMatmult
 *
 * Recursive multiplication of SStructMatrices A_i (i=1,...,n) computed as
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
 *         3) Info about neighboring parts on grid_M?
 *         4) Do not build grid_M, use a reference to existing grids from
 *            ssmatrices instead
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatmult( HYPRE_Int             nmatrices,
                      hypre_SStructMatrix **ssmatrices,
                      HYPRE_Int             nterms,
                      HYPRE_Int            *terms,
                      HYPRE_Int            *transposes,
                      hypre_SStructMatrix **M_ptr )
{
   MPI_Comm                 comm;
   hypre_SStructMatrix     *M;
   hypre_SStructGraph      *graph;
   hypre_SStructGraph      *graph_M;
   hypre_SStructGrid       *grid;
   hypre_SStructGrid       *grid_M;
   hypre_SStructGrid       *dom_grid;
   hypre_SStructGrid       *ran_grid;
   hypre_SStructPGrid      *pgrid;
   hypre_StructGrid        *sgrid;
   hypre_SStructPMatrix    *pmatrix;
   hypre_StructMatrix     **smatrices;   /* nmatrices array */
   hypre_StructMatrix     **smatrices_M; /* nparts array */
   HYPRE_SStructVariable   *vartypes;
   HYPRE_Int               *pids;
   HYPRE_Int               *pids_M;
   HYPRE_Int               *pids_all;

   /* Temporary data structures */
   hypre_ParCSRMatrix     **parcsr;
   hypre_ParCSRMatrix      *parcsr_sA;
   hypre_ParCSRMatrix      *parcsr_uA;
   hypre_ParCSRMatrix      *parcsr_uM;
   hypre_ParCSRMatrix      *parcsr_uMold;
   hypre_ParCSRMatrix      *parcsr_sM;
   hypre_ParCSRMatrix      *parcsr_sMold;
   hypre_IJMatrix          *ijmatrix;
   hypre_IJMatrix         **ij_sA;
   hypre_IJMatrix          *ij_M;
   hypre_BoxArray          *boxes;
   hypre_Box               *box;

   /* Stencil data */
   hypre_StructStencil     *stencil_M;
   hypre_SStructStencil   **stencils_M;
   hypre_Index             *stencil_shape_M;
   HYPRE_Int                stencil_size_M;

   /* This function works for a single variable type only */
   HYPRE_Int                vi = 0, vj = 0;
   HYPRE_Int                ndim;
   HYPRE_Int                i, m, p, q, s, t;
   HYPRE_Int                pid, part, part_cnt;
   HYPRE_Int                dom_nparts, ran_nparts;
   HYPRE_Int                nparts, nparts_M, nparts_all;
   HYPRE_Int                nvars_M;
   hypre_IndexRef           imin, imax;

   /*-------------------------------------------------------
    * Safety checks
    *-------------------------------------------------------*/

   /* TODO: add a check for the number and types of variables */

   /* TODO: check if we can multiply each of the matrices */

   /* Set some data */
   comm = hypre_SStructMatrixComm(ssmatrices[0]);
   ndim = hypre_SStructMatrixNDim(ssmatrices[0]);

   t = terms[0];
   graph = hypre_SStructMatrixGraph(ssmatrices[t]);
   if (transposes[t])
   {
      ran_grid = hypre_SStructGraphDomGrid(graph);
   }
   else
   {
      ran_grid = hypre_SStructGraphRanGrid(graph);
   }
   ran_nparts = hypre_SStructGridNParts(ran_grid);

   t = terms[nterms - 1];
   graph = hypre_SStructMatrixGraph(ssmatrices[t]);
   if (transposes[t])
   {
      dom_grid = hypre_SStructGraphRanGrid(graph);
   }
   else
   {
      dom_grid = hypre_SStructGraphDomGrid(graph);
   }
   dom_nparts = hypre_SStructGridNParts(dom_grid);
   nparts_M   = hypre_min(dom_nparts, ran_nparts);
   pids_M     = hypre_TAlloc(HYPRE_Int, nparts_M);

   /* Compute part ids of M */
   nparts_all = 0;
   for (m = 0; m < nmatrices; m++)
   {
      t = terms[m];
      nparts_all += hypre_SStructMatrixNParts(ssmatrices[t]);
   }
   pids_all = hypre_TAlloc(HYPRE_Int, nparts_all);

   part_cnt = 0;
   for (m = 0; m < nmatrices; m++)
   {
      t = terms[m];
      nparts = hypre_SStructMatrixNParts(ssmatrices[t]);
      graph  = hypre_SStructMatrixGraph(ssmatrices[t]);
      grid   = hypre_SStructGraphGrid(graph);

      for (part = 0; part < nparts; part++)
      {
         pids_all[part_cnt++] = hypre_SStructGridPartID(grid, part);
      }
   }

   hypre_qsort0(pids_all, 0, nparts_all - 1);

   part = pids_all[0];
   part_cnt = 1;
   nparts_M = 0;
   for (p = 1; p < nparts_all; p++)
   {
      if (pids_all[p] == part)
      {
         part_cnt++;
         if (part_cnt == nmatrices)
         {
            pids_M[nparts_M++] = part;
            part_cnt = 1;
            part = pids_all[++p];
         }
      }
      else
      {
         part = pids_all[p];
         part_cnt = 1;
      }
   }
   hypre_TFree(pids_all);

   /*-------------------------------------------------------
    * Compute structured component
    *-------------------------------------------------------*/
   smatrices   = hypre_TAlloc(hypre_StructMatrix *, nmatrices);
   smatrices_M = hypre_TAlloc(hypre_StructMatrix *, nparts_M);
   stencils_M  = hypre_TAlloc(hypre_SStructStencil *, nparts_M);
   for (part = 0; part < nparts_M; part++)
   {
      pid = pids_M[part];

      for (m = 0; m < nmatrices; m++)
      {
         nparts = hypre_SStructMatrixNParts(ssmatrices[m]);
         graph  = hypre_SStructMatrixGraph(ssmatrices[m]);
         grid   = hypre_SStructGraphGrid(graph);
         pids   = hypre_SStructGridPartIDs(grid);

         q = hypre_BinarySearch(pids, pid, nparts);

         pmatrix = hypre_SStructMatrixPMatrix(ssmatrices[m], q);
         smatrices[m] = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
      }

      /* Multiply StructMatrices (part, vi, vj)-block */
      hypre_StructMatmult(nmatrices, smatrices, nterms, terms,
                          transposes, &smatrices_M[part]);

      /* Create SStructStencil object for M */
      stencil_M       = hypre_StructMatrixStencil(smatrices_M[part]);
      stencil_size_M  = hypre_StructStencilSize(stencil_M);
      stencil_shape_M = hypre_StructStencilShape(stencil_M);

      HYPRE_SStructStencilCreate(ndim, stencil_size_M, &stencils_M[part]);
      for (s = 0; s < stencil_size_M; s++)
      {
         HYPRE_SStructStencilSetEntry(stencils_M[part], s, stencil_shape_M[s], vj);
      }
   }

   /*-------------------------------------------------------
    * Compute unstructured component
    *-------------------------------------------------------*/

   /* Temporary work matrices */
   parcsr = hypre_TAlloc(hypre_ParCSRMatrix *, 3);
   ij_sA  = hypre_TAlloc(hypre_IJMatrix *, nmatrices);
   for (m = 0; m < nmatrices; m++)
   {
      ij_sA[m] = NULL;
   }

   /* Set initial data */
   t = terms[nmatrices - 1];
   ijmatrix = hypre_SStructMatrixIJMatrix(ssmatrices[t]);
   HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uMold);
   ij_sA[t] = hypre_SStructMatrixToUMatrix(ssmatrices[t]);
   HYPRE_IJMatrixGetObject(ij_sA[t], (void **) &parcsr_sMold);

   /* Compute M recursively */
   for (m = (nmatrices - 2); m >= 0; m--)
   {
      t = terms[m];

      // Convert sA_n to IJMatrix
      if (ij_sA[t] == NULL)
      {
         ij_sA[t] = hypre_SStructMatrixToUMatrix(ssmatrices[t]);
      }
      HYPRE_IJMatrixGetObject(ij_sA[t], (void **) &parcsr_sA);

      // Compute sA_n*uMold
      if (transposes[m])
      {
         parcsr[0] = hypre_ParTMatmul(parcsr_sA, parcsr_uMold);
      }
      else
      {
         parcsr[0] = hypre_ParMatmul(parcsr_sA, parcsr_uMold);
      }

      // Compute uA_n*uMold
      ijmatrix = hypre_SStructMatrixIJMatrix(ssmatrices[t]);
      HYPRE_IJMatrixGetObject(ijmatrix, (void **) &parcsr_uA);
      if (transposes[m])
      {
         parcsr[1] = hypre_ParTMatmul(parcsr_uA, parcsr_uMold);
      }
      else
      {
         parcsr[1] = hypre_ParMatmul(parcsr_uA, parcsr_uMold);
         if (!parcsr[1])
         {
            hypre_ParCSRMatrixPrintIJ(parcsr_uA, 0, 0, "parcsr_uA");
            hypre_ParCSRMatrixPrintIJ(parcsr_uMold, 0, 0, "parcsr_uMold");
         }
      }

      // Note: Cannot free parcsr_uMold here since it holds col_starts info of parcsr[0].
      // Free uMold
      if (m > 1)
      {
         hypre_ParCSRMatrixDestroy(parcsr_uMold);
      }

      // Compute (sA_n*uMold + uA_n*uMold)
      hypre_ParcsrAdd(1.0, parcsr[0], 1.0, parcsr[1], &parcsr[2]);

      // Free sA_n*uMold and uA_n*uMold
      hypre_ParCSRMatrixDestroy(parcsr[0]);
      hypre_ParCSRMatrixDestroy(parcsr[1]);

      // Compute uA_n*sMold
      if (transposes[m])
      {
         parcsr[0] = hypre_ParTMatmul(parcsr_uA, parcsr_sMold);
      }
      else
      {
         parcsr[0] = hypre_ParMatmul(parcsr_uA, parcsr_sMold);
      }

      // Compute (uA_n*sMold + sA_n*uMold + uA_n*uMold)
      hypre_ParcsrAdd(1.0, parcsr[0], 1.0, parcsr[2], &parcsr_uM);

      // Free temporary work matrices
      hypre_ParCSRMatrixDestroy(parcsr[0]);
      hypre_ParCSRMatrixDestroy(parcsr[2]);

      // Compute sA_n*sMold
      if (transposes[m])
      {
         parcsr_sM = hypre_ParTMatmul(parcsr_sA, parcsr_sMold);
      }
      else
      {
         parcsr_sM = hypre_ParMatmul(parcsr_sA, parcsr_sMold);
      }

      if (m < (nmatrices - 2))
      {
         hypre_ParCSRMatrixDestroy(parcsr_sMold);
      }

      // Update pointers
      parcsr_sMold = parcsr_sM;
      parcsr_uMold = parcsr_uM;
   }

   // Free temporary work matrices
   hypre_TFree(parcsr);
   for (m = 0; m < nmatrices; m++)
   {
      if (ij_sA[m] != NULL)
      {
         HYPRE_IJMatrixDestroy(ij_sA[m]);
      }
   }

   /*-------------------------------------------------------
    * Create the resulting SStructMatrix
    *-------------------------------------------------------*/

   // Create grid
   HYPRE_SStructGridCreate(comm, ndim, nparts_M, &grid_M);
   graph = hypre_SStructMatrixGraph(ssmatrices[0]);
   grid  = hypre_SStructGraphGrid(graph);
   for (part = 0; part < nparts_M; part++)
   {
      if (smatrices_M[part])
      {
         sgrid = hypre_StructMatrixGrid(smatrices_M[part]);
         boxes = hypre_StructGridBoxes(sgrid);

         hypre_ForBoxI(i, boxes)
         {
            box  = hypre_BoxArrayBox(boxes, i);
            imin = hypre_BoxIMin(box);
            imax = hypre_BoxIMax(box);

            HYPRE_SStructGridSetExtents(grid_M, part, imin, imax);
         }

         pgrid = hypre_SStructGridPGrid(grid, part);
         vartypes = hypre_SStructPGridVarTypes(pgrid);
         HYPRE_SStructGridSetVariables(grid_M, part, 1, vartypes);
      }
   }
   hypre_SStructGridSetPartIDs(grid_M, pids_M);
   HYPRE_SStructGridAssemble(grid_M);

   // Create graph
   HYPRE_SStructGraphCreate(comm, grid_M, grid_M, (HYPRE_SStructGraph*) &graph_M);
   HYPRE_SStructGraphSetObjectType(graph_M, HYPRE_SSTRUCT);
   for (part = 0; part < nparts_M; part++)
   {
      if (stencils_M[part])
      {
         HYPRE_SStructGraphSetStencil(graph_M, part, 0, stencils_M[part]);
      }
   }

   // Create matrix
   HYPRE_SStructMatrixCreate(comm, graph_M, &M);
   HYPRE_SStructMatrixInitialize(M);
   for (part = 0; part < nparts_M; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(M, part);
      nvars_M = hypre_SStructPMatrixNVars(pmatrix);
      if (nvars_M)
      {
         hypre_SStructPMatrixSMatrix(pmatrix, vi, vj) = hypre_StructMatrixRef(smatrices_M[part]);
      }
   }

   ij_M = hypre_SStructMatrixIJMatrix(M);
   hypre_IJMatrixDestroyParCSR(ij_M);
   hypre_IJMatrixObject(ij_M) = NULL;
   hypre_IJMatrixTranslator(ij_M) = NULL;
   hypre_IJMatrixAssembleFlag(ij_M) = 1;
   hypre_IJMatrixSetObject(ij_M, parcsr_uM);
   HYPRE_IJMatrixGetObject(ij_M, (void **) &hypre_SStructMatrixParCSRMatrix(M));
   HYPRE_SStructMatrixAssemble(M);

   /*-------------------------------------------------------
    * Free memory
    *-------------------------------------------------------*/
   hypre_TFree(smatrices);

   /* Set pointer to output matrix */
   *M_ptr = M;

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_SStructMatmult
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatPtAP( hypre_SStructMatrix   *P,
                      hypre_SStructMatrix   *A,
                      hypre_SStructMatrix  **PtAP_ptr )
{
   hypre_SStructMatrix  *matrices[3] = {A, P, P};
   HYPRE_Int             nmatrices   = 3;
   HYPRE_Int             nterms      = 3;
   HYPRE_Int             terms[3]    = {1, 0, 1};
   HYPRE_Int             trans[3]    = {1, 0, 0};

   hypre_SStructMatmult(nmatrices, matrices, nterms, terms, trans, PtAP_ptr);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Notes:
 *         *) Consider a single variable type for now.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructPMatrixBoundaryToUMatrix( HYPRE_Int             order,
                                       hypre_SStructPMatrix *pMat,
                                       hypre_ParCSRMatrix   *uMat,
                                       hypre_ParCSRMatrix   *oMat )
{
   hypre_StructMatrix  *sMat;
   HYPRE_Int            vi = 0;
   HYPRE_Int            vj = 0;

   sMat = hypre_SStructPMatrixSMatrix(pMat, vi, vj);
   if (order)
   {
      // Convert to ParCSR the boxes of sMat that are needed when computing sMat*uMat
   }
   else
   {
      // Convert to ParCSR the boxes of sMat that are needed when computing uMat*sMat
   }

   return hypre_error_flag;
}
