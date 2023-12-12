/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Are private static arrays a problem?
 *
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fac.h"

#define MapStencilRank(stencil, rank)           \
   {                                            \
      HYPRE_Int ii,jj,kk;                       \
      ii = hypre_IndexX(stencil);               \
      jj = hypre_IndexY(stencil);               \
      kk = hypre_IndexZ(stencil);               \
      if (ii==-1)                               \
         ii=2;                                  \
      if (jj==-1)                               \
         jj=2;                                  \
      if (kk==-1)                               \
         kk=2;                                  \
      rank = ii + 3*jj + 9*kk;                  \
   }

#define InverseMapStencilRank(rank, stencil)    \
   {                                            \
      HYPRE_Int ij,ii,jj,kk;                    \
      ij = (rank%9);                            \
      ii = (ij%3);                              \
      jj = (ij-ii)/3;                           \
      kk = (rank-3*jj-ii)/9;                    \
      if (ii==2)                                \
         ii= -1;                                \
      if (jj==2)                                \
         jj= -1;                                \
      if (kk==2)                                \
         kk= -1;                                \
      hypre_SetIndex3(stencil, ii, jj, kk);     \
   }


#define AbsStencilShape(stencil, abs_shape)                     \
   {                                                            \
      HYPRE_Int ii,jj,kk;                                       \
      ii = hypre_IndexX(stencil);                               \
      jj = hypre_IndexY(stencil);                               \
      kk = hypre_IndexZ(stencil);                               \
      abs_shape= hypre_abs(ii) + hypre_abs(jj) + hypre_abs(kk); \
   }

/*--------------------------------------------------------------------------
 * hypre_AMR_CFCoarsen: Coarsens the CF interface to get the stencils
 * reaching into a coarsened fbox. Also sets the centre coefficient of CF
 * interface nodes to have "preserved" row sum.
 *
 * On entry, fac_A already has all the coefficient values of the cgrid
 * chunks that are not underlying a fbox.  Note that A & fac_A have the
 * same grid & graph. Therefore, we will use A's grid & graph.
 *
 * ASSUMING ONLY LIKE-VARIABLES COUPLE THROUGH CF CONNECTIONS.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMR_CFCoarsen( hypre_SStructMatrix  *   A,
                     hypre_SStructMatrix  *   fac_A,
                     hypre_Index              refine_factors,
                     HYPRE_Int                level )

{
   MPI_Comm                comm       = hypre_SStructMatrixComm(A);
   hypre_SStructGraph     *graph      = hypre_SStructMatrixGraph(A);
   HYPRE_Int               graph_type = hypre_SStructGraphObjectType(graph);
   hypre_SStructGrid      *grid       = hypre_SStructGraphGrid(graph);
   HYPRE_Int               nUventries = hypre_SStructGraphNUVEntries(graph);
   HYPRE_IJMatrix          ij_A       = hypre_SStructMatrixIJMatrix(A);
   HYPRE_Int               matrix_type = hypre_SStructMatrixObjectType(A);
   HYPRE_Int               ndim       = hypre_SStructMatrixNDim(A);

   hypre_SStructPMatrix   *A_pmatrix;
   hypre_StructMatrix     *smatrix_var;
   hypre_StructStencil    *stencils;
   HYPRE_Int               stencil_size;
   hypre_Index             stencil_shape_i;
   hypre_Index             loop_size;
   hypre_Box               refined_box;
   HYPRE_Real            **a_ptrs;
   hypre_Box              *A_dbox;

   HYPRE_Int               part_crse = level - 1;
   HYPRE_Int               part_fine = level;

   hypre_BoxManager       *fboxman;
   hypre_BoxManEntry     **boxman_entries, *boxman_entry;
   HYPRE_Int               nboxman_entries;
   hypre_Box               boxman_entry_box;

   hypre_BoxArrayArray  ***fgrid_cinterface_extents;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_Index             node_extents;
   hypre_Index             stridec, stridef;

   hypre_BoxArrayArray    *cinterface_arrays;
   hypre_BoxArray         *cinterface_array;
   hypre_Box              *fgrid_cinterface;

   HYPRE_Int               centre;

   HYPRE_Int               ci, fi, boxi;
   HYPRE_Int               max_stencil_size = 27;
   HYPRE_Int               falseV = 0;
   HYPRE_Int               trueV = 1;
   HYPRE_Int               found;
   HYPRE_Int              *stencil_ranks, *rank_stencils;
   HYPRE_BigInt            rank, startrank;
   HYPRE_Real             *vals;

   HYPRE_Int               i, j;
   HYPRE_Int               nvars, var1;

   hypre_Index             lindex, zero_index;
   hypre_Index             index1, index2;
   hypre_Index             index_temp;

   hypre_SStructUVEntry   *Uventry;
   HYPRE_Int               nUentries, cnt1;
   HYPRE_Int               box_array_size;

   HYPRE_Int              *ncols;
   HYPRE_BigInt           *rows, *cols;

   HYPRE_Int              *temp1, *temp2;

   HYPRE_Int               myid;

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_SetIndex3(zero_index, 0, 0, 0);
   hypre_SetIndex3(lindex, 0, 0, 0);

   hypre_BoxInit(&refined_box, ndim);
   hypre_BoxInit(&boxman_entry_box, ndim);

   /*--------------------------------------------------------------------------
    *  Task: Coarsen the CF interface connections of A into fac_A so that
    *  fac_A will have the stencil coefficients extending into a coarsened
    *  fbox. The centre coefficient is constructed to preserve the row sum.
    *--------------------------------------------------------------------------*/

   if (graph_type == HYPRE_SSTRUCT)
   {
      startrank = hypre_SStructGridGhstartRank(grid);
   }
   else if (graph_type == HYPRE_PARCSR)
   {
      startrank = hypre_SStructGridStartRank(grid);
   }
   else
   {
      startrank = 0;
   }

   /*--------------------------------------------------------------------------
    * Fine grid strides by the refinement factors.
    *--------------------------------------------------------------------------*/
   hypre_SetIndex3(stridec, 1, 1, 1);
   for (i = 0; i < ndim; i++)
   {
      stridef[i] = refine_factors[i];
   }
   for (i = ndim; i < 3; i++)
   {
      stridef[i] = 1;
   }

   /*--------------------------------------------------------------------------
    *  Determine the c/f interface index boxes: fgrid_cinterface_extents.
    *  These are between fpart= level and cpart= (level-1). The
    *  fgrid_cinterface_extents are indexed by cboxes, but fboxes that
    *  abutt a given cbox must be considered. Moreover, for each fbox,
    *  we can have a c/f interface from a number of different stencil
    *  directions- i.e., we have a boxarrayarray for each cbox, each
    *  fbox leading to a boxarray.
    *
    *  Algo.: For each cbox:
    *    1) refine & stretch by a unit in each dimension.
    *    2) boxman_intersect with the fgrid boxman to get all fboxes contained
    *       or abutting this cbox.
    *    3) get the fgrid_cinterface_extents for each of these fboxes.
    *
    *  fgrid_cinterface_extents[var1][ci]
    *--------------------------------------------------------------------------*/
   A_pmatrix =  hypre_SStructMatrixPMatrix(fac_A, part_crse);
   nvars    =  hypre_SStructPMatrixNVars(A_pmatrix);

   fgrid_cinterface_extents = hypre_TAlloc(hypre_BoxArrayArray **,  nvars, HYPRE_MEMORY_HOST);
   for (var1 = 0; var1 < nvars; var1++)
   {
      fboxman = hypre_SStructGridBoxManager(grid, part_fine, var1);
      stencils = hypre_SStructPMatrixSStencil(A_pmatrix, var1, var1);

      cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);
      fgrid_cinterface_extents[var1] = hypre_TAlloc(hypre_BoxArrayArray *,
                                                    hypre_BoxArraySize(cgrid_boxes), HYPRE_MEMORY_HOST);

      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

         hypre_StructMapCoarseToFine(hypre_BoxIMin(cgrid_box), zero_index,
                                     refine_factors, hypre_BoxIMin(&refined_box));
         hypre_SetIndex3(index1, refine_factors[0] - 1, refine_factors[1] - 1,
                         refine_factors[2] - 1);
         hypre_StructMapCoarseToFine(hypre_BoxIMax(cgrid_box), index1,
                                     refine_factors, hypre_BoxIMax(&refined_box));

         /*------------------------------------------------------------------------
          * Stretch the refined_box so that a BoxManIntersect will get abutting
          * fboxes.
          *------------------------------------------------------------------------*/
         for (i = 0; i < ndim; i++)
         {
            hypre_BoxIMin(&refined_box)[i] -= 1;
            hypre_BoxIMax(&refined_box)[i] += 1;
         }

         hypre_BoxManIntersect(fboxman, hypre_BoxIMin(&refined_box),
                               hypre_BoxIMax(&refined_box), &boxman_entries,
                               &nboxman_entries);

         fgrid_cinterface_extents[var1][ci] = hypre_BoxArrayArrayCreate(nboxman_entries, ndim);

         /*------------------------------------------------------------------------
          * Get the  fgrid_cinterface_extents using var1-var1 stencil (only like-
          * variables couple).
          *------------------------------------------------------------------------*/
         if (stencils != NULL)
         {
            for (i = 0; i < nboxman_entries; i++)
            {
               hypre_BoxManEntryGetExtents(boxman_entries[i],
                                           hypre_BoxIMin(&boxman_entry_box),
                                           hypre_BoxIMax(&boxman_entry_box));
               hypre_CFInterfaceExtents2(&boxman_entry_box, cgrid_box, stencils, refine_factors,
                                         hypre_BoxArrayArrayBoxArray(fgrid_cinterface_extents[var1][ci], i) );
            }
         }
         hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);

      }  /* hypre_ForBoxI(ci, cgrid_boxes) */
   }     /* for (var1= 0; var1< nvars; var1++) */

   /*--------------------------------------------------------------------------
    *  STEP 1:
    *        ADJUST THE ENTRIES ALONG THE C/F BOXES SO THAT THE COARSENED
    *        C/F CONNECTION HAS THE APPROPRIATE ROW SUM.
    *        WE ARE ASSUMING ONLY LIKE VARIABLES COUPLE.
    *--------------------------------------------------------------------------*/
   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);
      stencils =  hypre_SStructPMatrixSStencil(A_pmatrix, var1, var1);

      /*----------------------------------------------------------------------
       * Extract only where variables couple.
       *----------------------------------------------------------------------*/
      if (stencils != NULL)
      {
         stencil_size = hypre_StructStencilSize(stencils);

         /*------------------------------------------------------------------
          *  stencil_ranks[i]      =  rank of stencil entry i.
          *  rank_stencils[i]      =  stencil entry of rank i.
          *
          * These are needed in collapsing the unstructured connections to
          * a stencil connection.
          *------------------------------------------------------------------*/
         stencil_ranks = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
         rank_stencils = hypre_TAlloc(HYPRE_Int,  max_stencil_size, HYPRE_MEMORY_HOST);
         for (i = 0; i < max_stencil_size; i++)
         {
            rank_stencils[i] = -1;
            if (i < stencil_size)
            {
               stencil_ranks[i] = -1;
            }
         }

         for (i = 0; i < stencil_size; i++)
         {
            hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape_i);
            MapStencilRank(stencil_shape_i, j);
            stencil_ranks[i] = j;
            rank_stencils[stencil_ranks[i]] = i;
         }
         centre = rank_stencils[0];

         smatrix_var = hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var1);

         a_ptrs   = hypre_TAlloc(HYPRE_Real *,  stencil_size, HYPRE_MEMORY_HOST);
         hypre_ForBoxI(ci, cgrid_boxes)
         {
            cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

            cinterface_arrays = fgrid_cinterface_extents[var1][ci];
            A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix_var), ci);

            /*-----------------------------------------------------------------
             * Ptrs to the correct data location.
             *-----------------------------------------------------------------*/
            for (i = 0; i < stencil_size; i++)
            {
               hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape_i);
               a_ptrs[i] = hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                   ci,
                                                                   stencil_shape_i);
            }

            /*-------------------------------------------------------------------
             * Loop over the c/f interface boxes and set the centre to be the row
             * sum. Coarsen the c/f connection and set the centre to preserve
             * the row sum of the composite operator along the c/f interface.
             *-------------------------------------------------------------------*/
            hypre_ForBoxArrayI(fi, cinterface_arrays)
            {
               cinterface_array = hypre_BoxArrayArrayBoxArray(cinterface_arrays, fi);
               box_array_size  = hypre_BoxArraySize(cinterface_array);
               for (boxi = stencil_size; boxi < box_array_size; boxi++)
               {
                  fgrid_cinterface = hypre_BoxArrayBox(cinterface_array, boxi);
                  hypre_CopyIndex(hypre_BoxIMin(fgrid_cinterface), node_extents);
                  hypre_BoxGetSize(fgrid_cinterface, loop_size);

                  hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                            A_dbox, node_extents, stridec, iA);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     for (i = 0; i < stencil_size; i++)
                     {
                        if (i != centre)
                        {
                           a_ptrs[centre][iA] += a_ptrs[i][iA];
                        }
                     }

                     /*-----------------------------------------------------------------
                      * Search for unstructured connections for this coarse node. Need
                      * to compute the index of the node. We will "collapse" the
                      * unstructured connections to the appropriate stencil entry. Thus
                      * we need to serch for the stencil entry.
                      *-----------------------------------------------------------------*/
                     index_temp[0] = node_extents[0] + lindex[0];
                     index_temp[1] = node_extents[1] + lindex[1];
                     index_temp[2] = node_extents[2] + lindex[2];

                     hypre_SStructGridFindBoxManEntry(grid, part_crse, index_temp, var1,
                                                      &boxman_entry);
                     hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index_temp, &rank,
                                                           matrix_type);
                     if (nUventries > 0)
                     {
                        found = falseV;
                        if ((rank - startrank) >= hypre_SStructGraphIUVEntry(graph, 0) &&
                            (rank - startrank) <= hypre_SStructGraphIUVEntry(graph, nUventries - 1))
                        {
                           found = trueV;
                        }
                     }

                     /*-----------------------------------------------------------------
                      * The graph has Uventries only if (nUventries > 0). Therefore,
                      * check this. Only like variables contribute to the row sum.
                      *-----------------------------------------------------------------*/
                     if (nUventries > 0 && found == trueV)
                     {
                        Uventry = hypre_SStructGraphUVEntry(graph, rank - startrank);

                        if (Uventry != NULL)
                        {
                           nUentries = hypre_SStructUVEntryNUEntries(Uventry);

                           /*-----------------------------------------------------------
                            * extract only the connections to level part_fine and the
                            * correct variable.
                            *-----------------------------------------------------------*/
                           temp1 = hypre_CTAlloc(HYPRE_Int,  nUentries, HYPRE_MEMORY_HOST);
                           cnt1 = 0;
                           for (i = 0; i < nUentries; i++)
                           {
                              if (hypre_SStructUVEntryToPart(Uventry, i) == part_fine
                                  &&  hypre_SStructUVEntryToVar(Uventry, i) == var1)
                              {
                                 temp1[cnt1++] = i;
                              }
                           }

                           ncols = hypre_TAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
                           rows = hypre_TAlloc(HYPRE_BigInt,  cnt1, HYPRE_MEMORY_HOST);
                           cols = hypre_TAlloc(HYPRE_BigInt,  cnt1, HYPRE_MEMORY_HOST);
                           temp2 = hypre_TAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
                           vals = hypre_CTAlloc(HYPRE_Real,  cnt1, HYPRE_MEMORY_HOST);

                           for (i = 0; i < cnt1; i++)
                           {
                              ncols[i] = 1;
                              rows[i] = rank;
                              cols[i] = hypre_SStructUVEntryToRank(Uventry, temp1[i]);

                              /* determine the stencil connection pattern */
                              hypre_StructMapFineToCoarse(
                                 hypre_SStructUVEntryToIndex(Uventry, temp1[i]),
                                 zero_index, stridef, index2);
                              hypre_SubtractIndexes(index2, index_temp,
                                                    ndim, index1);
                              MapStencilRank(index1, temp2[i]);

                              /* zero off this stencil connection into the fbox */
                              if (temp2[i] < max_stencil_size)
                              {
                                 j = rank_stencils[temp2[i]];
                                 if (j >= 0)
                                 {
                                    a_ptrs[j][iA] = 0.0;
                                 }
                              }
                           }  /* for (i= 0; i< cnt1; i++) */

                           hypre_TFree(temp1, HYPRE_MEMORY_HOST);

                           HYPRE_IJMatrixGetValues(ij_A, cnt1, ncols, rows, cols, vals);
                           for (i = 0; i < cnt1; i++)
                           {
                              a_ptrs[centre][iA] += vals[i];
                           }

                           hypre_TFree(ncols, HYPRE_MEMORY_HOST);
                           hypre_TFree(rows, HYPRE_MEMORY_HOST);
                           hypre_TFree(cols, HYPRE_MEMORY_HOST);

                           /* compute the connection to the coarsened fine box */
                           for (i = 0; i < cnt1; i++)
                           {
                              if (temp2[i] < max_stencil_size)
                              {
                                 j = rank_stencils[temp2[i]];
                                 if (j >= 0)
                                 {
                                    a_ptrs[j][iA] += vals[i];
                                 }
                              }
                           }
                           hypre_TFree(vals, HYPRE_MEMORY_HOST);
                           hypre_TFree(temp2, HYPRE_MEMORY_HOST);

                           /* centre connection which preserves the row sum */
                           for (i = 0; i < stencil_size; i++)
                           {
                              if (i != centre)
                              {
                                 a_ptrs[centre][iA] -= a_ptrs[i][iA];
                              }
                           }

                        }   /* if (Uventry != NULL) */
                     }       /* if (nUventries > 0) */
                  }
                  hypre_SerialBoxLoop1End(iA);
               }  /* for (boxi= stencil_size; boxi< box_array_size; boxi++) */
            }     /* hypre_ForBoxArrayI(fi, cinterface_arrays) */
         }        /* hypre_ForBoxI(ci, cgrid_boxes) */

         hypre_TFree(a_ptrs, HYPRE_MEMORY_HOST);
         hypre_TFree(stencil_ranks, HYPRE_MEMORY_HOST);
         hypre_TFree(rank_stencils, HYPRE_MEMORY_HOST);
      }   /* if (stencils != NULL) */
   }      /* end var1 */


   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);

      hypre_ForBoxI(ci, cgrid_boxes)
      {
         hypre_BoxArrayArrayDestroy(fgrid_cinterface_extents[var1][ci]);
      }
      hypre_TFree(fgrid_cinterface_extents[var1], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(fgrid_cinterface_extents, HYPRE_MEMORY_HOST);

   return 0;
}
