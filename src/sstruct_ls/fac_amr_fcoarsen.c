/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   vals
 *
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "_hypre_struct_mv.hpp"
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
 * hypre_AMR_FCoarsen: Coarsen the fbox and f/c connections. Forms the
 * coarse operator by averaging neighboring connections in the refinement
 * patch.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMR_FCoarsen( hypre_SStructMatrix  *   A,
                    hypre_SStructMatrix  *   fac_A,
                    hypre_SStructPMatrix *   A_crse,
                    hypre_Index              refine_factors,
                    HYPRE_Int                level )

{
   hypre_Box               fine_box;
   hypre_Box               intersect_box;

   MPI_Comm                comm       = hypre_SStructMatrixComm(A);

   hypre_SStructGraph     *graph      = hypre_SStructMatrixGraph(A);
   HYPRE_Int               graph_type = hypre_SStructGraphObjectType(graph);
   hypre_SStructGrid      *grid       = hypre_SStructGraphGrid(graph);
   HYPRE_IJMatrix          ij_A       = hypre_SStructMatrixIJMatrix(A);
   HYPRE_Int               matrix_type = hypre_SStructMatrixObjectType(A);
   HYPRE_Int               ndim       = hypre_SStructMatrixNDim(A);

   hypre_SStructPMatrix   *A_pmatrix  = hypre_SStructMatrixPMatrix(fac_A, level);

   hypre_StructMatrix     *smatrix_var;
   hypre_StructStencil    *stencils, *stencils_last;
   HYPRE_Int               stencil_size = 0, stencil_last_size;
   hypre_Index             stencil_shape_i, stencil_last_shape_i;
   hypre_Index             loop_size;
   hypre_Box               loop_box;
   HYPRE_Real            **a_ptrs;
   hypre_Box              *A_dbox;

   HYPRE_Int               part_crse = level - 1;
   HYPRE_Int               part_fine = level;

   hypre_StructMatrix     *crse_smatrix;
   HYPRE_Real             *crse_ptr;
   HYPRE_Real            **crse_ptrs;
   hypre_Box              *crse_dbox;

   hypre_StructGrid       *cgrid;
   hypre_BoxArray         *cgrid_boxes;
   hypre_Box              *cgrid_box;
   hypre_Index             cstart;
   hypre_Index             fstart, fend;
   hypre_Index             stridec, stridef;

   hypre_StructGrid       *fgrid;
   hypre_BoxArray         *fgrid_boxes;
   hypre_Box              *fgrid_box;
   hypre_BoxArray       ***fgrid_crse_extents;
   hypre_BoxArray       ***fbox_interior;
   hypre_BoxArrayArray  ***fbox_bdy;
   HYPRE_Int            ***interior_fboxi;
   HYPRE_Int            ***bdy_fboxi;
   HYPRE_Int            ***cboxi_fboxes;
   HYPRE_Int             **cboxi_fcnt;

   hypre_BoxArray         *fbox_interior_ci, *fbox_bdy_ci_fi;
   hypre_BoxArrayArray    *fbox_bdy_ci;
   HYPRE_Int              *interior_fboxi_ci;
   HYPRE_Int              *bdy_fboxi_ci;

   HYPRE_Int               centre;

   hypre_BoxArray         *data_space;

   HYPRE_Int               ci, fi, arrayi;
   HYPRE_Int               max_stencil_size = 27;
   HYPRE_Int               trueV = 1;
   HYPRE_Int               falseV = 0;
   HYPRE_Int               found, sort;
   HYPRE_Int               stencil_marker;
   HYPRE_Int              *stencil_ranks = NULL, *rank_stencils = NULL;
   HYPRE_Int              *stencil_contrib_cnt = NULL;
   HYPRE_Int             **stencil_contrib_i = NULL;
   HYPRE_Real            **weight_contrib_i = NULL;
   HYPRE_Real              weights[4] = {1.0, 0.25, 0.125, 0.0625};
   HYPRE_Real              sum;
   HYPRE_Int               abs_stencil_shape;
   hypre_Box             **shift_box = NULL;
   hypre_Box               coarse_cell_box;
   HYPRE_Int               volume_coarse_cell_box;
   HYPRE_Int              *volume_shift_box = NULL;
   HYPRE_Int               max_contribut_size = 0, stencil_i;
   HYPRE_BigInt            startrank, rank;
   HYPRE_Real             *vals = NULL, *vals2 = NULL;

   HYPRE_Int               i, j, k, l, m, n, ll, kk, jj;
   HYPRE_Int               nvars, var1, var2, var2_start;
   HYPRE_Int               iA_shift_z, iA_shift_zy, iA_shift_zyx;

   hypre_Index             lindex;
   hypre_Index             index1, index2;
   hypre_Index             index_temp;

   HYPRE_Int             **box_graph_indices;
   HYPRE_Int              *box_graph_cnts;
   HYPRE_Int              *box_ranks, *box_ranks_cnt, *box_to_ranks_cnt;
   HYPRE_Int              *cdata_space_ranks, *box_starts, *box_ends;
   HYPRE_Int              *box_connections;
   HYPRE_Int             **coarse_contrib_Uv;
   HYPRE_Int              *fine_interface_ranks;
   HYPRE_Int               nUventries = hypre_SStructGraphNUVEntries(graph);
   HYPRE_Int              *iUventries  = hypre_SStructGraphIUVEntries(graph);
   hypre_SStructUVEntry  **Uventries   = hypre_SStructGraphUVEntries(graph);
   hypre_SStructUVEntry   *Uventry;
   HYPRE_Int               nUentries, cnt1;
   hypre_Index             index, *cindex, *Uv_cindex;
   HYPRE_Int               box_array_size, cbox_array_size;

   HYPRE_Int               nrows;
   HYPRE_BigInt            to_rank;
   HYPRE_Int              *ncols;
   HYPRE_BigInt           *rows, *cols;
   HYPRE_Int             **interface_max_stencil_ranks;
   HYPRE_Int             **interface_max_stencil_cnt;
   HYPRE_Int             **interface_rank_stencils;
   HYPRE_Int             **interface_stencil_ranks;
   HYPRE_Int              *coarse_stencil_cnt;
   HYPRE_Real             *stencil_vals;
   HYPRE_Int              *common_rank_stencils, *common_stencil_ranks;
   HYPRE_Int              *common_stencil_i;
   hypre_BoxManEntry      *boxman_entry;

   HYPRE_Int              *temp1, *temp2;
   HYPRE_Real             *temp3;
   HYPRE_Real              sum_contrib, scaling;

   HYPRE_Int             **OffsetA;

   HYPRE_Int              *parents;
   HYPRE_Int              *parents_cnodes;

   HYPRE_Int               myid;

   hypre_MPI_Comm_rank(comm, &myid);

   hypre_BoxInit(&fine_box, ndim);
   hypre_BoxInit(&intersect_box, ndim);
   hypre_BoxInit(&loop_box, ndim);
   hypre_BoxInit(&coarse_cell_box, ndim);

   /*--------------------------------------------------------------------------
    * Task: Coarsen the fbox and f/c connections to form the coarse grid
    * operator inside the fgrid.
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported graph_type!");
      return hypre_error_flag;
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
    * Scaling for averaging row sum.
    *--------------------------------------------------------------------------*/
   scaling = 1.0;
   for (i = 0; i < ndim - 2; i++)
   {
      scaling *= refine_factors[0];
   }

   /*--------------------------------------------------------------------------
    *  Determine the coarsened fine grid- fgrid_crse_extents.
    *  These are between fpart= level and cpart= (level-1). The
    *  fgrid_crse_extents will be indexed by cboxes- the boxarray of coarsened
    *  fboxes FULLY in a given cbox.
    *
    *  Also, determine the interior and boundary boxes of each fbox. Having
    *  these will allow us to determine the f/c interface nodes without
    *  extensive checking. These are also indexed by the cboxes.
    *    fgrid_interior- for each cbox, we have a collection of child fboxes,
    *                    each leading to an interior=> boxarray
    *    fgrid_bdy     - for each cbox, we have a collection of child fboxes,
    *                    each leading to a boxarray of bdies=> boxarrayarray.
    *  Because we need to know the fbox id for these boxarray/boxarrayarray,
    *  we will need one for each fbox.
    *
    *  And, determine which cboxes contain a given fbox. That is, given a
    *  fbox, find all cboxes that contain a chunk of it.
    *--------------------------------------------------------------------------*/
   nvars    =  hypre_SStructPMatrixNVars(A_pmatrix);

   fgrid_crse_extents      = hypre_TAlloc(hypre_BoxArray **,  nvars, HYPRE_MEMORY_HOST);
   fbox_interior           = hypre_TAlloc(hypre_BoxArray **,  nvars, HYPRE_MEMORY_HOST);
   fbox_bdy                = hypre_TAlloc(hypre_BoxArrayArray **,  nvars, HYPRE_MEMORY_HOST);
   interior_fboxi          = hypre_TAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);
   bdy_fboxi               = hypre_TAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);
   cboxi_fboxes            = hypre_TAlloc(HYPRE_Int **,  nvars, HYPRE_MEMORY_HOST);
   cboxi_fcnt              = hypre_TAlloc(HYPRE_Int *,  nvars, HYPRE_MEMORY_HOST);

   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);
      fgrid_crse_extents[var1] = hypre_TAlloc(hypre_BoxArray *,
                                              hypre_BoxArraySize(cgrid_boxes), HYPRE_MEMORY_HOST);
      fbox_interior[var1] = hypre_TAlloc(hypre_BoxArray *,
                                         hypre_BoxArraySize(cgrid_boxes), HYPRE_MEMORY_HOST);
      fbox_bdy[var1]     = hypre_TAlloc(hypre_BoxArrayArray *,
                                        hypre_BoxArraySize(cgrid_boxes), HYPRE_MEMORY_HOST);
      interior_fboxi[var1] = hypre_TAlloc(HYPRE_Int *,  hypre_BoxArraySize(cgrid_boxes),
                                          HYPRE_MEMORY_HOST);
      bdy_fboxi[var1]     = hypre_TAlloc(HYPRE_Int *,  hypre_BoxArraySize(cgrid_boxes),
                                         HYPRE_MEMORY_HOST);

      fgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      fgrid_boxes = hypre_StructGridBoxes(fgrid);

      cboxi_fboxes[var1] = hypre_CTAlloc(HYPRE_Int *,  hypre_BoxArraySize(fgrid_boxes),
                                         HYPRE_MEMORY_HOST);
      cboxi_fcnt[var1]  = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(fgrid_boxes), HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------------------
       *  Determine the fine grid boxes that are underlying a coarse grid box.
       *  Coarsen the indices to determine the looping extents of these
       *  boxes. Also, find the looping extents for the extended coarsened
       *  boxes, and the interior and boundary extents of a fine_grid box.
       *  The fine_grid boxes must be adjusted so that only the coarse nodes
       *  inside these boxes are included. Only the lower bound needs to be
       *  adjusted.
       *-----------------------------------------------------------------------*/
      hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
         hypre_CopyIndex(hypre_BoxIMin(cgrid_box), cstart);

         cnt1 = 0;
         temp1 = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(fgrid_boxes), HYPRE_MEMORY_HOST);

         hypre_ClearIndex(index_temp);
         hypre_ForBoxI(fi, fgrid_boxes)
         {
            fgrid_box = hypre_BoxArrayBox(fgrid_boxes, fi);
            hypre_CopyIndex(hypre_BoxIMin(fgrid_box), fstart);
            for (i = 0; i < ndim; i++)
            {
               j = fstart[i] % refine_factors[i];
               if (j)
               {
                  fstart[i] += refine_factors[i] - j;
               }
            }

            hypre_StructMapFineToCoarse(fstart, index_temp,
                                        refine_factors, hypre_BoxIMin(&fine_box));
            hypre_StructMapFineToCoarse(hypre_BoxIMax(fgrid_box), index_temp,
                                        refine_factors, hypre_BoxIMax(&fine_box));

            hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);
            if (hypre_BoxVolume(&intersect_box) > 0)
            {
               temp1[cnt1++] = fi;
            }
         }

         fgrid_crse_extents[var1][ci] = hypre_BoxArrayCreate(cnt1, ndim);
         fbox_interior[var1][ci]  = hypre_BoxArrayCreate(cnt1, ndim);
         fbox_bdy[var1][ci]       = hypre_BoxArrayArrayCreate(cnt1, ndim);
         interior_fboxi[var1][ci] = hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
         bdy_fboxi[var1][ci]      = hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);

         for (fi = 0; fi < cnt1; fi++)
         {
            fgrid_box = hypre_BoxArrayBox(fgrid_boxes, temp1[fi]);
            hypre_CopyIndex(hypre_BoxIMin(fgrid_box), fstart);
            hypre_CopyIndex(hypre_BoxIMax(fgrid_box), fend);

            /*--------------------------------------------------------------------
             * record which sides will be adjusted- fstart adjustments will
             * decrease the box size, whereas fend adjustments will increase the
             * box size. Since we fstart decreases the box size, we cannot
             * have an f/c interface at an adjusted fstart end. fend may
             * correspond to an f/c interface whether it has been adjusted or not.
             *--------------------------------------------------------------------*/
            hypre_SetIndex3(index1, 1, 1, 1);
            for (i = 0; i < ndim; i++)
            {
               j = fstart[i] % refine_factors[i];
               if (j)
               {
                  fstart[i] += refine_factors[i] - j;
                  index1[i] = 0;
               }

               j = fend[i] % refine_factors[i];
               if (refine_factors[i] - 1 - j)
               {
                  fend[i] += (refine_factors[i] - 1) - j;
               }
            }

            hypre_StructMapFineToCoarse(fstart, index_temp,
                                        refine_factors, hypre_BoxIMin(&fine_box));
            hypre_StructMapFineToCoarse(hypre_BoxIMax(fgrid_box), index_temp,
                                        refine_factors, hypre_BoxIMax(&fine_box));
            hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);

            hypre_CopyBox(&intersect_box,
                          hypre_BoxArrayBox(fgrid_crse_extents[var1][ci], fi));

            /*--------------------------------------------------------------------
             * adjust the fine intersect_box so that we get the interior and
             * boundaries separately.
             *--------------------------------------------------------------------*/
            hypre_StructMapCoarseToFine(hypre_BoxIMin(&intersect_box), index_temp,
                                        refine_factors, hypre_BoxIMin(&fine_box));

            /* the following index2 shift for ndim<3 is no problem since
               refine_factors[j]= 1 for j>=ndim. */
            hypre_SetIndex3(index2, refine_factors[0] - 1, refine_factors[1] - 1,
                            refine_factors[2] - 1);
            hypre_StructMapCoarseToFine(hypre_BoxIMax(&intersect_box), index2,
                                        refine_factors, hypre_BoxIMax(&fine_box));

            hypre_SetIndex3(index2, 1, 1, 1);
            hypre_CopyBox(&fine_box, &loop_box);
            for (i = 0; i < ndim; i++)
            {
               hypre_BoxIMin(&loop_box)[i] += refine_factors[i] * index1[i];
               hypre_BoxIMax(&loop_box)[i] -= refine_factors[i] * index2[i];
            }
            hypre_CopyBox(&loop_box,
                          hypre_BoxArrayBox(fbox_interior[var1][ci], fi));
            interior_fboxi[var1][ci][fi] = temp1[fi];

            hypre_SubtractBoxes(&fine_box, &loop_box,
                                hypre_BoxArrayArrayBoxArray(fbox_bdy[var1][ci], fi));
            bdy_fboxi[var1][ci][fi] = temp1[fi];
         }
         hypre_TFree(temp1, HYPRE_MEMORY_HOST);

      }  /* hypre_ForBoxI(ci, cgrid_boxes) */

      /*--------------------------------------------------------------------
       * Determine the cboxes that contain a chunk of a given fbox.
       *--------------------------------------------------------------------*/
      hypre_ForBoxI(fi, fgrid_boxes)
      {
         fgrid_box = hypre_BoxArrayBox(fgrid_boxes, fi);
         hypre_CopyIndex(hypre_BoxIMin(fgrid_box), fstart);
         for (i = 0; i < ndim; i++)
         {
            j = fstart[i] % refine_factors[i];
            if (j)
            {
               fstart[i] += refine_factors[i] - j;
            }
         }

         hypre_StructMapFineToCoarse(fstart, index_temp,
                                     refine_factors, hypre_BoxIMin(&fine_box));
         hypre_StructMapFineToCoarse(hypre_BoxIMax(fgrid_box), index_temp,
                                     refine_factors, hypre_BoxIMax(&fine_box));

         temp1 = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(cgrid_boxes), HYPRE_MEMORY_HOST);
         hypre_ForBoxI(i, cgrid_boxes)
         {
            cgrid_box = hypre_BoxArrayBox(cgrid_boxes, i);
            hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);
            if (hypre_BoxVolume(&intersect_box) > 0)
            {
               temp1[cboxi_fcnt[var1][fi]] = i;
               cboxi_fcnt[var1][fi]++;
            }
         }

         cboxi_fboxes[var1][fi] = hypre_TAlloc(HYPRE_Int,  cboxi_fcnt[var1][fi], HYPRE_MEMORY_HOST);
         for (i = 0; i < cboxi_fcnt[var1][fi]; i++)
         {
            cboxi_fboxes[var1][fi][i] = temp1[i];
         }
         hypre_TFree(temp1, HYPRE_MEMORY_HOST);
      }
   }     /* for (var1= 0; var1< nvars; var1++) */

   /*--------------------------------------------------------------------------
    *  STEP 1:
    *        COMPUTE THE COARSE LEVEL OPERATOR INSIDE OF A REFINED BOX.
    *
    *  We assume that the coarse and fine grid variables are of the same type.
    *
    *  Coarse stencils in the refinement patches are obtained by averaging the
    *  fine grid coefficients. Since we are assuming cell-centred discretization,
    *  we apply a weighted averaging of ONLY the fine grid coefficients along
    *  interfaces of adjacent agglomerated coarse cells.
    *
    *  Since the stencil pattern is assumed arbitrary, we must determine the
    *  stencil pattern of each var1-var2 struct_matrix to get the correct
    *  contributing stencil coefficients, averaging weights, etc.
    *--------------------------------------------------------------------------*/

   /*--------------------------------------------------------------------------
    *  Agglomerated coarse cell info. These are needed in defining the looping
    *  extents for averaging- i.e., we loop over extents determined by the
    *  size of the agglomerated coarse cell.
    *  Note that the agglomerated coarse cell is constructed correctly for
    *  any dimensions (1, 2, or 3).
    *--------------------------------------------------------------------------*/
   hypre_ClearIndex(index_temp);
   hypre_CopyIndex(index_temp, hypre_BoxIMin(&coarse_cell_box));
   hypre_SetIndex3(index_temp, refine_factors[0] - 1, refine_factors[1] - 1,
                   refine_factors[2] - 1 );
   hypre_CopyIndex(index_temp, hypre_BoxIMax(&coarse_cell_box));

   volume_coarse_cell_box = hypre_BoxVolume(&coarse_cell_box);


   /*--------------------------------------------------------------------------
    * Offsets in y & z directions for refinement patches. These will be used
    * for pointing to correct coarse stencil location.
    *--------------------------------------------------------------------------*/
   OffsetA =  hypre_CTAlloc(HYPRE_Int *,  2, HYPRE_MEMORY_HOST);
   for (i = 0; i < 2; i++)
   {
      OffsetA[i] = hypre_CTAlloc(HYPRE_Int,  refine_factors[i + 1], HYPRE_MEMORY_HOST);
   }

   /*--------------------------------------------------------------------------
    *  Stencil contribution cnts, weights, etc are computed only if we have
    *  a new stencil pattern. If the pattern is the same, the previously
    *  computed stencil contribution cnts, weights, etc can be used.
    *
    *  Mark the stencil_marker so that the first time the stencil is non-null,
    *  the stencil contribution cnts, weights, etc are computed.
    *--------------------------------------------------------------------------*/
   stencil_marker = trueV;
   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);

      fgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      fgrid_boxes = hypre_StructGridBoxes(fgrid);


      for (var2 = 0; var2 < nvars; var2++)
      {
         stencils = hypre_SStructPMatrixSStencil(A_crse, var1, var2);
         if (stencils != NULL)
         {
            stencil_size = hypre_StructStencilSize(stencils);

            /*-----------------------------------------------------------------
             * When stencil_marker== true, form the stencil contributions cnts,
             * weights, etc. This occurs for the first non-null stencil or
             * when the stencil shape of the current non-null stencil has a
             * different stencil shape from that of the latest non-null stencil.
             *
             * But when  stencil_marker== false, we must check to see if we
             * need new stencil contributions cnts, weights, etc. Thus, find
             * the latest non-null stencil for comparison.
             *-----------------------------------------------------------------*/
            if (stencil_marker == falseV)
            {
               /* search for the first previous non-null stencil */
               found     = falseV;
               var2_start = var2 - 1;
               for (j = var1; j >= 0; j--)
               {
                  for (i = var2_start; i >= 0; i--)
                  {
                     stencils_last = hypre_SStructPMatrixSStencil(A_crse, j, i);
                     if (stencils_last != NULL)
                     {
                        found = trueV;
                        break;
                     }
                  }
                  if (found)
                  {
                     break;
                  }
                  else
                  {
                     var2_start = nvars - 1;
                  }
               }

               /*--------------------------------------------------------------
                * Compare the stencil shape.
                *--------------------------------------------------------------*/
               stencil_last_size = hypre_StructStencilSize(stencils_last);
               if (stencil_last_size != stencil_size)
               {
                  stencil_marker = trueV;
                  break;
               }
               else
               {
                  found = falseV;
                  for (i = 0; i < stencil_size; i++)
                  {
                     hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                                     stencil_shape_i);
                     hypre_CopyIndex(hypre_StructStencilElement(stencils_last, i),
                                     stencil_last_shape_i);

                     hypre_SetIndex3(index_temp,
                                     stencil_shape_i[0] - stencil_last_shape_i[0],
                                     stencil_shape_i[1] - stencil_last_shape_i[1],
                                     stencil_shape_i[2] - stencil_last_shape_i[2]);

                     AbsStencilShape(index_temp, abs_stencil_shape);
                     if (abs_stencil_shape)
                     {
                        found = trueV;
                        stencil_marker = trueV;
                        hypre_TFree(stencil_contrib_cnt, HYPRE_MEMORY_HOST);
                        hypre_TFree(stencil_ranks, HYPRE_MEMORY_HOST);
                        for (i = 0; i < stencil_size; i++)
                        {
                           hypre_BoxDestroy(shift_box[i]);
                        }
                        hypre_TFree(shift_box, HYPRE_MEMORY_HOST);
                        hypre_TFree(volume_shift_box, HYPRE_MEMORY_HOST);
                        hypre_TFree(vals, HYPRE_MEMORY_HOST);

                        for (j = 1; j < max_stencil_size; j++)
                        {
                           stencil_i = rank_stencils[j];
                           if (stencil_i != -1)
                           {
                              hypre_TFree(stencil_contrib_i[stencil_i], HYPRE_MEMORY_HOST);
                              hypre_TFree(weight_contrib_i[stencil_i], HYPRE_MEMORY_HOST);
                           }
                        }
                        hypre_TFree(stencil_contrib_i, HYPRE_MEMORY_HOST);
                        hypre_TFree(weight_contrib_i, HYPRE_MEMORY_HOST);
                        hypre_TFree(rank_stencils, HYPRE_MEMORY_HOST);
                     }

                     if (found)
                     {
                        break;
                     }
                  }   /* for (i= 0; i< stencil_size; i++) */
               }      /* else */
            }         /* if (stencil_marker == false) */

            /*-----------------------------------------------------------------
             *  If stencil_marker==true, form the contribution structures.
             *  Since the type of averaging is determined by the stencil shapes,
             *  we need a ranking of the stencil shape to allow for easy
             *  determination.
             *
             *  top:  14  12  13    centre:  5  3  4     bottom 23   21   22
             *        11   9  10             2  0  1            20   18   19
             *        17  15  16             8  6  7            26   24   25
             *
             *  for stencil of max. size 27.
             *
             *  stencil_contrib_cnt[i]=  no. of fine stencils averaged to
             *                           form stencil entry i.
             *  stencil_contrib_i[i]  =  rank of fine stencils contributing
             *                           to form stencil entry i.
             *  weight_contrib_i[i]   =  array of weights for weighting
             *                           the contributions to stencil entry i.
             *  stencil_ranks[i]      =  rank of stencil entry i.
             *  rank_stencils[i]      =  stencil entry of rank i.
             *-----------------------------------------------------------------*/

            if (stencil_marker == trueV)
            {

               /* mark stencil_marker for the next stencil */
               stencil_marker = falseV;

               stencil_contrib_cnt = hypre_CTAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
               stencil_contrib_i  = hypre_TAlloc(HYPRE_Int *,  stencil_size, HYPRE_MEMORY_HOST);
               weight_contrib_i   = hypre_TAlloc(HYPRE_Real *,  stencil_size, HYPRE_MEMORY_HOST);
               stencil_ranks      = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
               rank_stencils      = hypre_TAlloc(HYPRE_Int,  max_stencil_size, HYPRE_MEMORY_HOST);
               shift_box          = hypre_TAlloc(hypre_Box *,  stencil_size, HYPRE_MEMORY_HOST);
               volume_shift_box   = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);

               for (i = 0; i < max_stencil_size; i++)
               {
                  rank_stencils[i] = -1;
                  if (i < stencil_size)
                  {
                     stencil_ranks[i] = -1;
                  }
               }

               /*-----------------------------------------------------------------
                *  Get mappings between stencil entries and ranks and vice versa;
                *  fine grid looping extents for averaging of the fine coefficients;
                *  and the number of fine grid values to be averaged.
                *  Note that the shift_boxes are constructed correctly for any
                *  dimensions. For j>=ndim,
                *  hypre_BoxIMin(shift_box[i])[j]=hypre_BoxIMax(shift_box[i])[j]= 0.
                *-----------------------------------------------------------------*/
               for (i = 0; i < stencil_size; i++)
               {
                  shift_box[i] = hypre_BoxCreate(ndim);
                  hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                                  stencil_shape_i);
                  MapStencilRank(stencil_shape_i, j);
                  stencil_ranks[i] = j;
                  rank_stencils[stencil_ranks[i]] = i;

                  hypre_SetIndex3(hypre_BoxIMin(shift_box[i]),
                                  (refine_factors[0] - 1)*stencil_shape_i[0],
                                  (refine_factors[1] - 1)*stencil_shape_i[1],
                                  (refine_factors[2] - 1)*stencil_shape_i[2]);

                  hypre_AddIndexes(hypre_BoxIMin(shift_box[i]),
                                   hypre_BoxIMax(&coarse_cell_box), 3,
                                   hypre_BoxIMax(shift_box[i]));

                  hypre_IntersectBoxes(&coarse_cell_box, shift_box[i], shift_box[i]);

                  volume_shift_box[i] = hypre_BoxVolume(shift_box[i]);
               }

               /*-----------------------------------------------------------------
                *  Derive the contribution info.
                *  The above rank table is used to determine the direction indices.
                *  Weight construction procedure valid for any dimensions.
                *-----------------------------------------------------------------*/

               /* east */
               stencil_i = rank_stencils[1];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 4; i <= 7; i += 3)
                  {
                     if (rank_stencils[i] != -1)       /* ne or se */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 1; i <= 7; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = stencil_contrib_cnt[stencil_i];
               }

               /* fill up the east contribution stencil indices */
               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 4; i <= 7; i += 3)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 1; i <= 7; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }


               /* west */
               stencil_i = rank_stencils[2];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 5; i <= 8; i += 3)
                  {
                     if (rank_stencils[i] != -1)       /* nw or sw */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 2; i <= 8; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = hypre_max( max_contribut_size,
                                                  stencil_contrib_cnt[stencil_i] );
               }

               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 5; i <= 8; i += 3)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 2; i <= 8; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }


               /* north */
               stencil_i = rank_stencils[3];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 4; i <= 5; i++)
                  {
                     if (rank_stencils[i] != -1)       /* ne or nw */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 3; i <= 5; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = hypre_max( max_contribut_size,
                                                  stencil_contrib_cnt[stencil_i] );
               }

               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 4; i <= 5; i++)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 3; i <= 5; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }

               /* south */
               stencil_i = rank_stencils[6];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 7; i <= 8; i++)
                  {
                     if (rank_stencils[i] != -1)       /* ne or nw */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 6; i <= 8; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = hypre_max( max_contribut_size,
                                                  stencil_contrib_cnt[stencil_i] );
               }


               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 7; i <= 8; i++)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 6; i <= 8; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }

               /*-----------------------------------------------------------------
                *  If only 2-d, extract the corner indices.
                *-----------------------------------------------------------------*/
               if (ndim == 2)
               {
                  /* corners: ne  & nw */
                  for (i = 4; i <= 5; i++)
                  {
                     stencil_i = rank_stencils[i];
                     if (stencil_i != -1)
                     {
                        stencil_contrib_cnt[stencil_i]++;
                        stencil_contrib_i[stencil_i] = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
                        weight_contrib_i[stencil_i] =  hypre_TAlloc(HYPRE_Real,  1, HYPRE_MEMORY_HOST);
                        stencil_contrib_i[stencil_i][0] = stencil_i;
                        weight_contrib_i[stencil_i][0] = weights[0];
                     }
                  }

                  /* corners: se  & sw */
                  for (i = 7; i <= 8; i++)
                  {
                     stencil_i = rank_stencils[i];
                     if (stencil_i != -1)
                     {
                        stencil_contrib_cnt[stencil_i]++;
                        stencil_contrib_i[stencil_i] = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
                        weight_contrib_i[stencil_i] =  hypre_TAlloc(HYPRE_Real,  1, HYPRE_MEMORY_HOST);
                        stencil_contrib_i[stencil_i][0] = stencil_i;
                        weight_contrib_i[stencil_i][0] = weights[0];
                     }
                  }
               }

               /*-----------------------------------------------------------------
                *  Additional directions for 3-dim case
                *-----------------------------------------------------------------*/
               if (ndim > 2)
               {
                  /* sides: top */
                  stencil_i = rank_stencils[9];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[9 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[9 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[9 + i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[9 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* sides: bottom */
                  stencil_i = rank_stencils[18];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[18 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[18 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[18 + i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[18 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: cne */
                  stencil_i = rank_stencils[4];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 4] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 4] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 4];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 4]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: cse */
                  stencil_i = rank_stencils[7];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 7] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 7] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 7];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 7]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: cnw */
                  stencil_i = rank_stencils[5];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 5] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 5] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 5];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 5]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: csw */
                  stencil_i = rank_stencils[8];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 8] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 8] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 8];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 8]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top east */
                  stencil_i = rank_stencils[10];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[10 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[10 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[10 + i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[10 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top west */
                  stencil_i = rank_stencils[11];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[11 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[11 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[11 + i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[11 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top north */
                  stencil_i = rank_stencils[12];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 13; i <= 14; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 13; i <= 14; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top south*/
                  stencil_i = rank_stencils[15];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 16; i <= 17; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 16; i <= 17; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom east */
                  stencil_i = rank_stencils[19];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[19 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[19 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[19 + i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[19 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom west */
                  stencil_i = rank_stencils[20];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[20 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[20 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[20 + i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[20 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom north */
                  stencil_i = rank_stencils[21];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 22; i <= 23; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 22; i <= 23; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom south*/
                  stencil_i = rank_stencils[24];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 25; i <= 26; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Int,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        hypre_TAlloc(HYPRE_Real,  stencil_contrib_cnt[stencil_i], HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 25; i <= 26; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* corners*/
                  for (j = 1; j <= 2; j++)
                  {
                     for (i = 4; i <= 5; i++)
                     {
                        stencil_i = rank_stencils[9 * j + i];
                        if (stencil_i != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                           stencil_contrib_i[stencil_i] = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
                           weight_contrib_i[stencil_i] =  hypre_TAlloc(HYPRE_Real,  1, HYPRE_MEMORY_HOST);
                           stencil_contrib_i[stencil_i][0] = stencil_i;
                           weight_contrib_i[stencil_i][0] = weights[0];
                        }
                     }
                     for (i = 7; i <= 8; i++)
                     {
                        stencil_i = rank_stencils[9 * j + i];
                        if (stencil_i != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                           stencil_contrib_i[stencil_i] = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
                           weight_contrib_i[stencil_i] =  hypre_TAlloc(HYPRE_Real,  1, HYPRE_MEMORY_HOST);
                           stencil_contrib_i[stencil_i][0] = stencil_i;
                           weight_contrib_i[stencil_i][0] = weights[0];
                        }
                     }
                  }

               }       /* if ndim > 2 */
               /*-----------------------------------------------------------------
                *  Allocate for the temporary vector used in computing the
                *  averages.
                *-----------------------------------------------------------------*/
               vals = hypre_CTAlloc(HYPRE_Real,  max_contribut_size, HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                *  coarse grid stencil contributor structures have been formed.
                *-----------------------------------------------------------------*/
            }   /* if (stencil_marker == true) */

            /*---------------------------------------------------------------------
             *  Loop over gridboxes to average stencils
             *---------------------------------------------------------------------*/
            smatrix_var = hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
            crse_smatrix = hypre_SStructPMatrixSMatrix(A_crse, var1, var2);

            /*---------------------------------------------------------------------
             *  data ptrs to extract and fill in data.
             *---------------------------------------------------------------------*/
            a_ptrs   = hypre_TAlloc(HYPRE_Real *,  stencil_size, HYPRE_MEMORY_HOST);
            crse_ptrs = hypre_TAlloc(HYPRE_Real *,  stencil_size, HYPRE_MEMORY_HOST);

            hypre_ForBoxI(ci, cgrid_boxes)
            {
               cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
               fbox_interior_ci = fbox_interior[var1][ci];
               fbox_bdy_ci      = fbox_bdy[var1][ci];
               interior_fboxi_ci = interior_fboxi[var1][ci];
               bdy_fboxi_ci     = bdy_fboxi[var1][ci];

               crse_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(crse_smatrix),
                                             ci);
               /*------------------------------------------------------------------
                * grab the correct coarse grid pointers. These are the parent base
                * grids.
                *------------------------------------------------------------------*/
               for (i = 0; i < stencil_size; i++)
               {
                  hypre_CopyIndex(hypre_StructStencilElement(stencils, i), stencil_shape_i);
                  crse_ptrs[i] = hypre_StructMatrixExtractPointerByIndex(crse_smatrix,
                                                                         ci,
                                                                         stencil_shape_i);
               }
               /*------------------------------------------------------------------
                *  Loop over the interior of each patch inside cgrid_box.
                *------------------------------------------------------------------*/
               hypre_ForBoxI(fi, fbox_interior_ci)
               {
                  fgrid_box = hypre_BoxArrayBox(fbox_interior_ci, fi);
                  /*--------------------------------------------------------------
                   * grab the fine grid ptrs & create the offsets for the fine
                   * grid ptrs.
                   *--------------------------------------------------------------*/
                  A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix_var),
                                             interior_fboxi_ci[fi]);
                  for (i = 0; i < stencil_size; i++)
                  {
                     hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                                     stencil_shape_i);
                     a_ptrs[i] =
                        hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                interior_fboxi_ci[fi],
                                                                stencil_shape_i);
                  }

                  /*---------------------------------------------------------------
                   *  Compute the offsets for pointing to the correct data.
                   *  Note that for 1-d, OffsetA[j][i]= 0. Therefore, this ptr
                   *  will be correct for 1-d.
                   *---------------------------------------------------------------*/
                  for (j = 0; j < 2; j++)
                  {
                     OffsetA[j][0] = 0;
                     for (i = 1; i < refine_factors[j + 1]; i++)
                     {
                        if (j == 0)
                        {
                           hypre_SetIndex3(index_temp, 0, i, 0);
                        }
                        else
                        {
                           hypre_SetIndex3(index_temp, 0, 0, i);
                        }
                        OffsetA[j][i] = hypre_BoxOffsetDistance(A_dbox, index_temp);
                     }
                  }

                  hypre_CopyIndex(hypre_BoxIMin(fgrid_box), fstart);
                  hypre_CopyIndex(hypre_BoxIMax(fgrid_box), fend);

                  /* coarsen the interior patch box*/
                  hypre_ClearIndex(index_temp);
                  hypre_StructMapFineToCoarse(fstart, index_temp, stridef,
                                              hypre_BoxIMin(&fine_box));
                  hypre_StructMapFineToCoarse(fend, index_temp, stridef,
                                              hypre_BoxIMax(&fine_box));

                  hypre_CopyIndex(hypre_BoxIMin(&fine_box), cstart);

                  /*----------------------------------------------------------------
                   * Loop over interior grid box.
                   *----------------------------------------------------------------*/

                  hypre_BoxGetSize(&fine_box, loop_size);

                  hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                            A_dbox, fstart, stridef, iA,
                                            crse_dbox, cstart, stridec, iAc);
                  {
                     for (i = 0; i < stencil_size; i++)
                     {
                        rank =  stencil_ranks[i];

                        /*------------------------------------------------------------
                         *  Loop over refinement agglomeration making up a coarse cell
                         *  when a non-centre stencil.
                         *------------------------------------------------------------*/
                        if (rank)
                        {
                           /*--------------------------------------------------------
                            *  Loop over refinement agglomeration extents making up a
                            *  a coarse cell.
                            *--------------------------------------------------------*/
                           hypre_CopyIndex(hypre_BoxIMin(shift_box[i]), index1);
                           hypre_CopyIndex(hypre_BoxIMax(shift_box[i]), index2);

                           for (m = 0; m < stencil_contrib_cnt[i]; m++)
                           {
                              vals[m] = 0.0;
                           }

                           /*--------------------------------------------------------
                            * For 1-d, index1[l]= index2[l]= 0, l>=1. So
                            *    iA_shift_zyx= j,
                            * which is correct. Similarly, 2-d is correct.
                            *--------------------------------------------------------*/
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              iA_shift_z = iA + OffsetA[1][l];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 iA_shift_zy = iA_shift_z + OffsetA[0][k];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    iA_shift_zyx = iA_shift_zy + j;

                                    for (m = 0; m < stencil_contrib_cnt[i]; m++)
                                    {
                                       stencil_i = stencil_contrib_i[i][m];
                                       vals[m] += a_ptrs[stencil_i][iA_shift_zyx];
                                    }
                                 }
                              }
                           }
                           /*----------------------------------------------------------
                            *  average & weight the contributions and place into coarse
                            *  stencil entry.
                            *----------------------------------------------------------*/
                           crse_ptrs[i][iAc] = 0.0;
                           for (m = 0; m < stencil_contrib_cnt[i]; m++)
                           {
                              crse_ptrs[i][iAc] += vals[m] * weight_contrib_i[i][m];
                           }
                           crse_ptrs[i][iAc] /= volume_shift_box[i];

                        }  /* if (rank) */
                     }     /* for i */

                     /*------------------------------------------------------------------
                      *  centre stencil:
                      *  The centre stencil is computed so that the row sum is equal to
                      *  the sum of the row sums of the fine matrix. Uses the computed
                      *  coarse off-diagonal stencils.
                      *
                      *  No fine-coarse interface for the interior boxes.
                      *------------------------------------------------------------------*/
                     hypre_CopyIndex(hypre_BoxIMin(&coarse_cell_box), index1);
                     hypre_CopyIndex(hypre_BoxIMax(&coarse_cell_box), index2);

                     sum = 0.0;
                     for (l = index1[2]; l <= index2[2]; l++)
                     {
                        iA_shift_z = iA + OffsetA[1][l];
                        for (k = index1[1]; k <= index2[1]; k++)
                        {
                           iA_shift_zy = iA_shift_z + OffsetA[0][k];
                           for (j = index1[0]; j <= index2[0]; j++)
                           {
                              iA_shift_zyx = iA_shift_zy + j;
                              for (m = 0; m < stencil_size; m++)
                              {
                                 sum += a_ptrs[m][iA_shift_zyx];
                              }
                           }
                        }
                     }

                     /*---------------------------------------------------------------
                      * coarse centre coefficient- when away from the fine-coarse
                      * interface, the centre coefficient is the sum of the
                      * off-diagonal components.
                      *---------------------------------------------------------------*/
                     sum /= scaling;
                     for (m = 0; m < stencil_size; m++)
                     {
                        rank = stencil_ranks[m];
                        if (rank)
                        {
                           sum -= crse_ptrs[m][iAc];
                        }
                     }
                     crse_ptrs[ rank_stencils[0] ][iAc] = sum;
                  }
                  hypre_SerialBoxLoop2End(iA, iAc);
               }    /* end hypre_ForBoxI(fi, fbox_interior_ci) */

               /*------------------------------------------------------------------
                *  Loop over the boundaries of each patch inside cgrid_box.
                *------------------------------------------------------------------*/
               hypre_ForBoxArrayI(arrayi, fbox_bdy_ci)
               {
                  fbox_bdy_ci_fi = hypre_BoxArrayArrayBoxArray(fbox_bdy_ci, arrayi);
                  hypre_ForBoxI(fi, fbox_bdy_ci_fi)
                  {
                     fgrid_box = hypre_BoxArrayBox(fbox_bdy_ci_fi, fi);

                     /*-----------------------------------------------------------
                      * grab the fine grid ptrs & create the offsets for the fine
                      * grid ptrs.
                      *-----------------------------------------------------------*/
                     A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix_var),
                                                bdy_fboxi_ci[arrayi]);
                     for (i = 0; i < stencil_size; i++)
                     {
                        hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                                        stencil_shape_i);
                        a_ptrs[i] =
                           hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                   bdy_fboxi_ci[arrayi],
                                                                   stencil_shape_i);
                     }

                     /*--------------------------------------------------------------
                      *  Compute the offsets for pointing to the correct data.
                      *--------------------------------------------------------------*/
                     for (j = 0; j < 2; j++)
                     {
                        OffsetA[j][0] = 0;
                        for (i = 1; i < refine_factors[j + 1]; i++)
                        {
                           if (j == 0)
                           {
                              hypre_SetIndex3(index_temp, 0, i, 0);
                           }
                           else
                           {
                              hypre_SetIndex3(index_temp, 0, 0, i);
                           }
                           OffsetA[j][i] = hypre_BoxOffsetDistance(A_dbox, index_temp);
                        }
                     }

                     hypre_CopyIndex(hypre_BoxIMin(fgrid_box), fstart);
                     hypre_CopyIndex(hypre_BoxIMax(fgrid_box), fend);

                     /* coarsen the patch box*/
                     hypre_ClearIndex(index_temp);
                     hypre_StructMapFineToCoarse(fstart, index_temp, stridef,
                                                 hypre_BoxIMin(&fine_box));
                     hypre_StructMapFineToCoarse(fend, index_temp, stridef,
                                                 hypre_BoxIMax(&fine_box));

                     hypre_CopyIndex(hypre_BoxIMin(&fine_box), cstart);

                     /*--------------------------------------------------------------
                      * Loop over boundary grid box.
                      *--------------------------------------------------------------*/

                     hypre_BoxGetSize(&fine_box, loop_size);

                     hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                               A_dbox, fstart, stridef, iA,
                                               crse_dbox, cstart, stridec, iAc);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        for (i = 0; i < stencil_size; i++)
                        {
                           rank =  stencil_ranks[i];

                           /*--------------------------------------------------------
                            * Loop over refinement agglomeration making up a coarse
                            * cell when a non-centre stencil.
                            *--------------------------------------------------------*/
                           if (rank)
                           {
                              /*-----------------------------------------------------
                               * Loop over refinement agglomeration extents making up
                               * a coarse cell.
                               *-----------------------------------------------------*/
                              hypre_CopyIndex(hypre_BoxIMin(shift_box[i]), index1);
                              hypre_CopyIndex(hypre_BoxIMax(shift_box[i]), index2);

                              for (m = 0; m < stencil_contrib_cnt[i]; m++)
                              {
                                 vals[m] = 0.0;
                              }

                              for (l = index1[2]; l <= index2[2]; l++)
                              {
                                 iA_shift_z = iA + OffsetA[1][l];
                                 for (k = index1[1]; k <= index2[1]; k++)
                                 {
                                    iA_shift_zy = iA_shift_z + OffsetA[0][k];
                                    for (j = index1[0]; j <= index2[0]; j++)
                                    {
                                       iA_shift_zyx = iA_shift_zy + j;

                                       for (m = 0; m < stencil_contrib_cnt[i]; m++)
                                       {
                                          stencil_i = stencil_contrib_i[i][m];
                                          vals[m] += a_ptrs[stencil_i][iA_shift_zyx];
                                       }
                                    }
                                 }
                              }
                              /*---------------------------------------------------------
                               *  average & weight the contributions and place into coarse
                               *  stencil entry.
                               *---------------------------------------------------------*/
                              crse_ptrs[i][iAc] = 0.0;
                              for (m = 0; m < stencil_contrib_cnt[i]; m++)
                              {
                                 crse_ptrs[i][iAc] += vals[m] * weight_contrib_i[i][m];
                              }
                              crse_ptrs[i][iAc] /= volume_shift_box[i];

                           }  /* if (rank) */
                        }     /* for i */

                        /*---------------------------------------------------------------
                         *  centre stencil:
                         *  The centre stencil is computed so that the row sum is equal to
                         *  th sum of the row sums of the fine matrix. Uses the computed
                         *  coarse off-diagonal stencils.
                         *
                         *  Along the fine-coarse interface, we need to add the unstructured
                         *  connections.
                         *---------------------------------------------------------------*/
                        hypre_CopyIndex(hypre_BoxIMin(&coarse_cell_box), index1);
                        hypre_CopyIndex(hypre_BoxIMax(&coarse_cell_box), index2);

                        temp3 = hypre_CTAlloc(HYPRE_Real,  volume_coarse_cell_box, HYPRE_MEMORY_HOST);

                        /*---------------------------------------------------------------
                         *  iA_shift_zyx is computed correctly for 1 & 2-d. Also,
                         *  ll= 0 for 2-d, and ll= kk= 0 for 1-d. Correct ptrs.
                         *---------------------------------------------------------------*/
                        for (l = index1[2]; l <= index2[2]; l++)
                        {
                           iA_shift_z = iA + OffsetA[1][l];
                           ll        = l * refine_factors[1] * refine_factors[0];
                           for (k = index1[1]; k <= index2[1]; k++)
                           {
                              iA_shift_zy = iA_shift_z + OffsetA[0][k];
                              kk         = ll + k * refine_factors[0];
                              for (j = index1[0]; j <= index2[0]; j++)
                              {
                                 iA_shift_zyx = iA_shift_zy + j;
                                 jj          = kk + j;
                                 for (m = 0; m < stencil_size; m++)
                                 {
                                    temp3[jj] += a_ptrs[m][iA_shift_zyx];
                                 }
                              }
                           }
                        }

                        /*------------------------------------------------------------
                         * extract all unstructured connections. Note that we extract
                         * from sstruct_matrix A, which already has been assembled.
                         *------------------------------------------------------------*/
                        if (nUventries > 0)
                        {
                           temp2 = hypre_CTAlloc(HYPRE_Int,  volume_coarse_cell_box, HYPRE_MEMORY_HOST);
                           cnt1 = 0;
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              ll = l * refine_factors[1] * refine_factors[0];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 kk = ll + k * refine_factors[0];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    jj = kk + j;

                                    hypre_SetIndex3(index_temp,
                                                    j + lindex[0]*stridef[0],
                                                    k + lindex[1]*stridef[1],
                                                    l + lindex[2]*stridef[2]);
                                    hypre_AddIndexes(fstart, index_temp, 3, index_temp);

                                    hypre_SStructGridFindBoxManEntry(grid, part_fine, index_temp,
                                                                     var1, &boxman_entry);
                                    hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index_temp,
                                                                          &rank, matrix_type);

                                    found = falseV;
                                    i = hypre_SStructGraphIUVEntry(graph, 0);
                                    m = hypre_SStructGraphIUVEntry(graph, nUventries - 1);
                                    if ((rank - startrank) >= i && (rank - startrank) <= m)
                                    {
                                       found = trueV;
                                    }

                                    if (found)
                                    {
                                       Uventry = hypre_SStructGraphUVEntry(graph, rank - startrank);

                                       if (Uventry != NULL)
                                       {
                                          nUentries = hypre_SStructUVEntryNUEntries(Uventry);

                                          m = 0;
                                          for (i = 0; i < nUentries; i++)
                                          {
                                             if (hypre_SStructUVEntryToPart(Uventry, i) == part_crse)
                                             {
                                                m++;
                                             }
                                          }  /* for (i= 0; i< nUentries; i++) */

                                          temp2[jj] = m;
                                          cnt1    += m;

                                       }  /* if (Uventry != NULL) */
                                    }     /* if (found) */

                                 }   /* for (j= index1[0]; j<= index2[0]; j++) */
                              }      /* for (k= index1[1]; k<= index2[1]; k++) */
                           }         /* for (l= index1[2]; l<= index2[2]; l++) */

                           ncols = hypre_TAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
                           for (l = 0; l < cnt1; l++)
                           {
                              ncols[l] = 1;
                           }

                           rows = hypre_TAlloc(HYPRE_BigInt,  cnt1, HYPRE_MEMORY_HOST);
                           cols = hypre_TAlloc(HYPRE_BigInt,  cnt1, HYPRE_MEMORY_HOST);
                           vals2 = hypre_CTAlloc(HYPRE_Real,  cnt1, HYPRE_MEMORY_HOST);

                           cnt1 = 0;
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              ll = l * refine_factors[1] * refine_factors[0];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 kk = ll + k * refine_factors[0];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    jj = kk + j;

                                    hypre_SetIndex3(index_temp,
                                                    j + lindex[0]*stridef[0],
                                                    k + lindex[1]*stridef[1],
                                                    l + lindex[2]*stridef[2]);
                                    hypre_AddIndexes(fstart, index_temp, 3, index_temp);

                                    hypre_SStructGridFindBoxManEntry(grid, part_fine, index_temp,
                                                                     var1, &boxman_entry);
                                    hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index_temp,
                                                                          &rank, matrix_type);

                                    found = falseV;
                                    if (nUventries > 0)
                                    {
                                       i = hypre_SStructGraphIUVEntry(graph, 0);
                                       m = hypre_SStructGraphIUVEntry(graph, nUventries - 1);
                                       if ((HYPRE_Int)(rank - startrank) >= i && (HYPRE_Int)(rank - startrank) <= m)
                                       {
                                          found = trueV;
                                       }
                                    }

                                    if (found)
                                    {
                                       Uventry = hypre_SStructGraphUVEntry(graph, (HYPRE_Int)(rank - startrank));

                                       if (Uventry != NULL)
                                       {
                                          nUentries = hypre_SStructUVEntryNUEntries(Uventry);
                                          for (i = 0; i < nUentries; i++)
                                          {
                                             if (hypre_SStructUVEntryToPart(Uventry, i) == part_crse)
                                             {
                                                rows[cnt1] = rank;
                                                cols[cnt1++] = hypre_SStructUVEntryToRank(Uventry, i);
                                             }

                                          }  /* for (i= 0; i< nUentries; i++) */
                                       }     /* if (Uventry != NULL) */
                                    }        /* if (found) */

                                 }   /* for (j= index1[0]; j<= index2[0]; j++) */
                              }      /* for (k= index1[1]; k<= index2[1]; k++) */
                           }         /* for (l= index1[2]; l<= index2[2]; l++) */

                           HYPRE_IJMatrixGetValues(ij_A, cnt1, ncols, rows, cols, vals2);

                           cnt1 = 0;
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              ll = l * refine_factors[1] * refine_factors[0];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 kk = ll + k * refine_factors[0];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    jj = kk + j;
                                    for (m = 0; m < temp2[jj]; m++)
                                    {
                                       temp3[jj] += vals2[cnt1];
                                       cnt1++;
                                    }
                                    temp2[jj] = 0; /* zero off for next time */
                                 }       /* for (j= index1[0]; j<= index2[0]; j++) */
                              }           /* for (k= index1[1]; k<= index2[1]; k++) */
                           }              /* for (l= index1[2]; l<= index2[2]; l++) */

                           hypre_TFree(ncols, HYPRE_MEMORY_HOST);
                           hypre_TFree(rows, HYPRE_MEMORY_HOST);
                           hypre_TFree(cols, HYPRE_MEMORY_HOST);
                           hypre_TFree(vals2, HYPRE_MEMORY_HOST);
                           hypre_TFree(temp2, HYPRE_MEMORY_HOST);

                        }   /* if Uventries > 0 */

                        sum = 0.0;
                        for (l = index1[2]; l <= index2[2]; l++)
                        {
                           ll = l * refine_factors[1] * refine_factors[0];
                           for (k = index1[1]; k <= index2[1]; k++)
                           {
                              kk = ll + k * refine_factors[0];
                              for (j = index1[0]; j <= index2[0]; j++)
                              {
                                 jj = kk + j;
                                 sum += temp3[jj];
                              }
                           }
                        }

                        sum /= scaling;
                        crse_ptrs[ rank_stencils[0] ][iAc] = sum;

                        hypre_TFree(temp3, HYPRE_MEMORY_HOST);

                     }
                     hypre_SerialBoxLoop2End(iA, iAc);

                  }  /* hypre_ForBoxI(fi, fbox_bdy_ci_fi) */
               }      /* hypre_ForBoxArrayI(arrayi, fbox_bdy_ci) */
            }          /* hypre_ForBoxI(ci, cgrid_boxes) */

            hypre_TFree(a_ptrs, HYPRE_MEMORY_HOST);
            hypre_TFree(crse_ptrs, HYPRE_MEMORY_HOST);

         }    /* if (stencils != NULL) */
      }       /* end var2 */
   }          /* end var1 */

   if (stencil_contrib_cnt)
   {
      hypre_TFree(stencil_contrib_cnt, HYPRE_MEMORY_HOST);
   }
   if (stencil_ranks)
   {
      hypre_TFree(stencil_ranks, HYPRE_MEMORY_HOST);
   }
   if (volume_shift_box)
   {
      hypre_TFree(volume_shift_box, HYPRE_MEMORY_HOST);
   }
   if (vals)
   {
      hypre_TFree(vals, HYPRE_MEMORY_HOST);
   }

   if (shift_box)
   {
      for (j = 0; j < stencil_size; j++)
      {
         if (shift_box[j])
         {
            hypre_BoxDestroy(shift_box[j]);
         }
      }
      hypre_TFree(shift_box, HYPRE_MEMORY_HOST);
   }

   if (stencil_contrib_i)
   {
      for (j = 1; j < max_stencil_size; j++)
      {
         stencil_i = rank_stencils[j];
         if (stencil_i != -1)
         {
            if (stencil_contrib_i[stencil_i])
            {
               hypre_TFree(stencil_contrib_i[stencil_i], HYPRE_MEMORY_HOST);
            }
         }
      }
      hypre_TFree(stencil_contrib_i, HYPRE_MEMORY_HOST);
   }

   if (weight_contrib_i)
   {
      for (j = 1; j < max_stencil_size; j++)
      {
         stencil_i = rank_stencils[j];
         if (stencil_i != -1)
         {
            if (weight_contrib_i[stencil_i])
            {
               hypre_TFree(weight_contrib_i[stencil_i], HYPRE_MEMORY_HOST);
            }
         }
      }
      hypre_TFree(weight_contrib_i, HYPRE_MEMORY_HOST);
   }

   if (rank_stencils)
   {
      hypre_TFree(rank_stencils, HYPRE_MEMORY_HOST);
   }

   if (OffsetA)
   {
      for (j = 0; j < 2; j++)
      {
         if (OffsetA[j])
         {
            hypre_TFree(OffsetA[j], HYPRE_MEMORY_HOST);
         }
      }
      hypre_TFree(OffsetA, HYPRE_MEMORY_HOST);
   }

   /*--------------------------------------------------------------------------
    *  STEP 2:
    *
    *  Interface coarsening: fine-to-coarse connections. We are
    *  assuming that only like-variables couple along interfaces.
    *
    *  The task is to coarsen all the fine-to-coarse unstructured
    *  connections and to compute coarse coefficients along the
    *  interfaces (coarse-to-fine coefficients are obtained from these
    *  computed values assuming symmetry). This involves
    *      1) scanning over the graph entries to find the locations of
    *         the unstructure connections;
    *      2) determining the stencil shape of the coarsened connections;
    *      3) averaging the unstructured coefficients to compute
    *         coefficient entries for the interface stencils;
    *      4) determining the weights of the interface stencil coefficients
    *         to construct the structured coarse grid matrix along the
    *         interfaces.
    *
    *  We perform this task by
    *      1) scanning over the graph entries to group the locations
    *         of the fine-to-coarse connections wrt the boxes of the
    *         fine grid. Temporary vectors storing the Uventries indices
    *         and the number of connections for each box will be created;
    *      2) for each fine grid box, group the fine-to-coarse connections
    *         with respect to the connected coarse nodes. Temporary vectors
    *         storing the Uventry indices and the Uentry indices for each
    *         coarse node will be created (i.e., for a fixed coarse grid node,
    *         record the fine node Uventries indices that connect to this
    *         coarse node and Uentry index of the Uventry that contains
    *         this coarse node.). The grouping is accomplished comparing the
    *         ranks of the coarse nodes;
    *      3) using the Uventries and Uentry indices for each coarse node,
    *         "coarsen" the fine grid connections to this coarse node to
    *         create interface stencils (wrt to the coarse nodes- i.e.,
    *         the centre of the stencil is at a coarse node). Also, find
    *         the IJ rows and columns corresponding to all the fine-to-coarse
    *         connections in a box, and extract the  unstructured coefficients;
    *      4) looping over all coarse grid nodes connected to a fixed fine box,
    *         compute the arithmetically averaged interface stencils;
    *      5) compare the underlying coarse grid structured stencil shape
    *         to the interface stencil shape to determine how to weight the
    *         averaged interface stencil coefficients.
    *
    *  EXCEPTION: A NODE CAN CONTAIN ONLY UNSTRUCTURED CONNECTIONS
    *  BETWEEN ONLY TWO AMR LEVELS- I.E., WE CANNOT HAVE A NODE THAT
    *  IS ON THE INTERFACE OF MORE THAN TWO AMR LEVELS. CHANGES TO
    *  HANDLE THIS LATTER CASE WILL INVOLVE THE SEARCH FOR f/c
    *  CONNECTIONS.
    *-----------------------------------------------------------------*/
   if (nUventries > 0)
   {
      nvars    =  hypre_SStructPMatrixNVars(A_pmatrix);

      for (var1 = 0; var1 < nvars; var1++)
      {
         /*-----------------------------------------------------------------
          *  Yank out the structured stencils for this variable (only like
          *  variables considered) and find their ranks.
          *-----------------------------------------------------------------*/
         stencils    = hypre_SStructPMatrixSStencil(A_crse, var1, var1);
         stencil_size = hypre_StructStencilSize(stencils);

         stencil_ranks = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
         rank_stencils = hypre_TAlloc(HYPRE_Int,  max_stencil_size, HYPRE_MEMORY_HOST);
         for (i = 0; i < stencil_size; i++)
         {
            hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                            stencil_shape_i);
            MapStencilRank( stencil_shape_i, stencil_ranks[i] );
            rank_stencils[ stencil_ranks[i] ] = i;
         }
         /*-----------------------------------------------------------------
          *  qsort the ranks into ascending order
          *-----------------------------------------------------------------*/
         hypre_qsort0(stencil_ranks, 0, stencil_size - 1);

         crse_smatrix = hypre_SStructPMatrixSMatrix(A_crse, var1, var1);
         cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_crse), var1);
         cgrid_boxes = hypre_StructGridBoxes(cgrid);

         fgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
         fgrid_boxes = hypre_StructGridBoxes(fgrid);

         box_starts = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(fgrid_boxes), HYPRE_MEMORY_HOST);
         box_ends  = hypre_CTAlloc(HYPRE_Int,  hypre_BoxArraySize(fgrid_boxes), HYPRE_MEMORY_HOST);
         hypre_SStructGraphFindSGridEndpts(graph, part_fine, var1, myid,
                                           0, box_starts);
         hypre_SStructGraphFindSGridEndpts(graph, part_fine, var1, myid,
                                           1, box_ends);

         /*-----------------------------------------------------------------
          *  Step 1: scanning over the graph entries to group the locations
          *          of the unstructured connections wrt to fine grid boxes.
          *
          *  Count the components that couple for each box.
          *
          *  box_graph_indices[fi]=   array of Uventries indices in box fi.
          *  box_graph_cnts[fi]   =   number of Uventries in box fi.
          *  cdata_space_rank[ci] =   begin offset rank of coarse data_space
          *                           box ci.
          *-----------------------------------------------------------------*/
         box_array_size   = hypre_BoxArraySize(fgrid_boxes);
         cbox_array_size  = hypre_BoxArraySize(cgrid_boxes);
         box_graph_indices = hypre_CTAlloc(HYPRE_Int *,  box_array_size, HYPRE_MEMORY_HOST);
         box_graph_cnts   = hypre_CTAlloc(HYPRE_Int,  box_array_size, HYPRE_MEMORY_HOST);

         data_space = hypre_StructMatrixDataSpace(crse_smatrix);
         cdata_space_ranks = hypre_CTAlloc(HYPRE_Int,  cbox_array_size, HYPRE_MEMORY_HOST);
         cdata_space_ranks[0] = 0;
         for (i = 1; i < cbox_array_size; i++)
         {
            cdata_space_ranks[i] = cdata_space_ranks[i - 1] +
                                   hypre_BoxVolume(hypre_BoxArrayBox(data_space, i - 1));
         }

         /*-----------------------------------------------------------------
          *  Scanning obtained by searching iUventries between the start
          *  and end of a fine box. Binary search used to find the interval
          *  between these two endpts. Index (-1) returned if no interval
          *  bounds found. Note that if start has positive index, then end
          *  must have a positive index also.
          *-----------------------------------------------------------------*/
         for (fi = 0; fi < box_array_size; fi++)
         {
            i = hypre_LowerBinarySearch(iUventries, box_starts[fi], nUventries);
            if (i >= 0)
            {
               j = hypre_UpperBinarySearch(iUventries, box_ends[fi], nUventries);
               box_graph_indices[fi] = hypre_TAlloc(HYPRE_Int,  j - i + 1, HYPRE_MEMORY_HOST);

               for (k = 0; k < (j - i + 1); k++)
               {
                  Uventry = hypre_SStructGraphUVEntry(graph,
                                                      iUventries[i + k]);

                  for (m = 0; m < hypre_SStructUVEntryNUEntries(Uventry); m++)
                  {
                     if (hypre_SStructUVEntryToPart(Uventry, m) == part_crse)
                     {
                        box_graph_indices[fi][box_graph_cnts[fi]] = iUventries[i + k];
                        box_graph_cnts[fi]++;
                        break;
                     }
                  }  /* for (m= 0; m< hypre_SStructUVEntryNUEntries(Uventry); m++) */
               }     /* for (k= 0; k< (j-i+1); k++) */
            }        /* if (i >= 0) */
         }           /* for (fi= 0; fi< box_array_size; fi++) */

         /*-----------------------------------------------------------------
          *  Step 2:
          *  Determine and group the fine-to-coarse connections in a box.
          *  Grouped according to the coarsened fine grid interface nodes.
          *
          *  box_ranks              = ranks of coarsened fine grid interface
          *                           nodes.
          *  box_connections        = counter for the distinct coarsened fine
          *                           grid interface nodes. This can be
          *                           used to group all the Uventries of a
          *                           coarsened fine grid node.
          *  cindex[l]              = the hypre_Index of coarsen node l.
          *  parents_cnodes[l]      = parent box that contains the coarsened
          *                           fine grid interface node l.
          *  fine_interface_ranks[l]= rank of coarsened fine grid interface
          *                           node l.
          *  box_ranks_cnt[l]       = counter for no. of Uventries for
          *                           coarsened node l.
          *  coarse_contrib_Uv[l]   = Uventry indices for Uventries that
          *                           contain fine-to-coarse connections of
          *                           coarse node l.
          *-----------------------------------------------------------------*/
         for (fi = 0; fi < box_array_size; fi++)
         {
            /*-------------------------------------------------------------
             * Determine the coarse data ptrs corresponding to fine box fi.
             * These are needed in assigning the averaged unstructured
             * coefficients.
             *
             * Determine how many distinct coarse grid nodes are in the
             * unstructured connection for a given box. Each node has a
             * structures.
             *
             * temp1 & temp2 are linked lists vectors used for grouping the
             * Uventries for a given coarse node.
             *-------------------------------------------------------------*/
            box_ranks       = hypre_TAlloc(HYPRE_Int,  box_graph_cnts[fi], HYPRE_MEMORY_HOST);
            box_connections = hypre_TAlloc(HYPRE_Int,  box_graph_cnts[fi], HYPRE_MEMORY_HOST);
            parents         = hypre_TAlloc(HYPRE_Int,  box_graph_cnts[fi], HYPRE_MEMORY_HOST);
            temp1           = hypre_CTAlloc(HYPRE_Int,  box_graph_cnts[fi] + 1, HYPRE_MEMORY_HOST);
            temp2           = hypre_CTAlloc(HYPRE_Int,  box_graph_cnts[fi], HYPRE_MEMORY_HOST);
            Uv_cindex       = hypre_TAlloc(hypre_Index,  box_graph_cnts[fi], HYPRE_MEMORY_HOST);

            /*-------------------------------------------------------------
             * determine the parent box of this fgrid_box.
             *-------------------------------------------------------------*/
            hypre_ClearIndex(index_temp);
            for (i = 0; i < box_graph_cnts[fi]; i++)
            {
               Uventry = Uventries[box_graph_indices[fi][i]];

               /*-------------------------------------------------------------
                * Coarsen the fine grid interface nodes and then get their
                * ranks. The correct coarse grid is needed to determine the
                * correct data_box.
                * Save the rank of the coarsened index & the parent box id.
                *-------------------------------------------------------------*/
               hypre_CopyIndex(hypre_SStructUVEntryIndex(Uventry), index);
               hypre_StructMapFineToCoarse(index, index_temp, stridef, Uv_cindex[i]);
               hypre_BoxSetExtents(&fine_box, Uv_cindex[i], Uv_cindex[i]);

               ci = 0;
               for (j = 0; j < cboxi_fcnt[var1][fi]; j++)
               {
                  ci = cboxi_fboxes[var1][fi][j];
                  cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);
                  hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);
                  if (hypre_BoxVolume(&intersect_box) > 0)
                  {
                     break;
                  }
               }

               parents[i]  = ci;
               box_ranks[i] = cdata_space_ranks[ci] +
                              hypre_BoxIndexRank(hypre_BoxArrayBox(data_space, ci),
                                                 Uv_cindex[i]);
            }

            /*---------------------------------------------------------------
             * Determine and "group" the Uventries using the box_ranks.
             * temp2 stores the Uventries indices for a coarsen node.
             *---------------------------------------------------------------*/
            cnt1 = 0;
            j   = 0;
            temp1[cnt1] = j;

            for (i = 0; i < box_graph_cnts[fi]; i++)
            {
               if (box_ranks[i] != -1)
               {
                  k                 = box_ranks[i];
                  box_connections[i] = cnt1;
                  temp2[j++]        = box_graph_indices[fi][i];

                  for (l = i + 1; l < box_graph_cnts[fi]; l++)
                  {
                     if (box_ranks[l] == k)
                     {
                        box_connections[l] = cnt1;
                        temp2[j++]        = box_graph_indices[fi][l];
                        box_ranks[l]      = -1;
                     }
                  }
                  cnt1++;
                  temp1[cnt1] = j;
               }
            }

            /*-----------------------------------------------------------------
             *  Store the graph entry info and other index info for each coarse
             *  grid node.
             *-----------------------------------------------------------------*/
            parents_cnodes      = hypre_TAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
            fine_interface_ranks = hypre_TAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
            box_ranks_cnt       = hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
            coarse_contrib_Uv   = hypre_TAlloc(HYPRE_Int *,  cnt1, HYPRE_MEMORY_HOST);
            cindex              = hypre_TAlloc(hypre_Index,  cnt1, HYPRE_MEMORY_HOST);

            for (i = 0; i < box_graph_cnts[fi]; i++)
            {
               if (box_ranks[i] != -1)
               {
                  j                      = box_connections[i];
                  parents_cnodes[j]      = parents[i];
                  fine_interface_ranks[j] =
                     hypre_BoxIndexRank(hypre_BoxArrayBox(data_space, parents[i]),
                                        Uv_cindex[i]);
                  hypre_CopyIndex(Uv_cindex[i], cindex[j]);

                  box_ranks_cnt[j]       = temp1[j + 1] - temp1[j];
                  coarse_contrib_Uv[j]   = hypre_TAlloc(HYPRE_Int,  box_ranks_cnt[j], HYPRE_MEMORY_HOST);

                  l                      = temp1[j];
                  for (k = 0; k < box_ranks_cnt[j]; k++)
                  {
                     coarse_contrib_Uv[j][k] = temp2[l + k];
                  }
               }
            }

            if (box_ranks)
            {
               hypre_TFree(box_ranks, HYPRE_MEMORY_HOST);
            }
            if (box_connections)
            {
               hypre_TFree(box_connections, HYPRE_MEMORY_HOST);
            }
            if (parents)
            {
               hypre_TFree(parents, HYPRE_MEMORY_HOST);
            }
            if (temp1)
            {
               hypre_TFree(temp1, HYPRE_MEMORY_HOST);
            }
            if (temp2)
            {
               hypre_TFree(temp2, HYPRE_MEMORY_HOST);
            }
            if (Uv_cindex)
            {
               hypre_TFree(Uv_cindex, HYPRE_MEMORY_HOST);
            }

            /*------------------------------------------------------------------------
             *  Step 3:
             *  Create the interface stencils.
             *
             *   interface_max_stencil_ranks[i] =  stencil_shape rank for each coarse
             *                                     Uentry connection of coarsened node
             *                                     i (i.e., the stencil_shape ranks of
             *                                     the interface stencils at node i).
             *   interface_max_stencil_cnt[i][m]=  counter for number of Uentries
             *                                     that describes a connection which
             *                                     coarsens into stencil_shape rank m.
             *   coarse_stencil_cnts[i]         =  counter for the no. of distinct
             *                                     interface stencil_shapes (i.e., the
             *                                     no. entries of the interface stencil).
             *   interface_stencil_ranks[i][l]  =  stencil_shape rank for interface
             *                                     stencil entry l, for coarse node i.
             *   interface_rank_stencils[i][j]  =  interface stencil entry for
             *                                     stencil_shape rank j, for node i.
             *------------------------------------------------------------------------*/

            /*-----------------------------------------------------------------
             *  Extract rows & cols info for extracting data from IJ matrix.
             *  Extract for all connections for a box.
             *-----------------------------------------------------------------*/
            hypre_ClearIndex(index_temp);

            nrows = 0;
            box_to_ranks_cnt =  hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);
            for (i = 0; i < cnt1; i++)
            {
               for (j = 0; j < box_ranks_cnt[i]; j++)
               {
                  Uventry  = Uventries[ coarse_contrib_Uv[i][j] ];
                  for (k = 0; k < hypre_SStructUVEntryNUEntries(Uventry); k++)
                  {
                     if (hypre_SStructUVEntryToPart(Uventry, k) == part_crse)
                     {
                        box_to_ranks_cnt[i]++;
                     }
                  }
               }
               nrows += box_to_ranks_cnt[i];
            }

            ncols = hypre_TAlloc(HYPRE_Int,  nrows, HYPRE_MEMORY_HOST);
            for (i = 0; i < nrows; i++)
            {
               ncols[i] = 1;
            }

            rows =  hypre_TAlloc(HYPRE_BigInt,  nrows, HYPRE_MEMORY_HOST);
            cols =  hypre_TAlloc(HYPRE_BigInt,  nrows, HYPRE_MEMORY_HOST);
            vals =  hypre_CTAlloc(HYPRE_Real,  nrows, HYPRE_MEMORY_HOST);

            interface_max_stencil_ranks =  hypre_TAlloc(HYPRE_Int *,  cnt1, HYPRE_MEMORY_HOST);
            interface_max_stencil_cnt  =  hypre_TAlloc(HYPRE_Int *,  cnt1, HYPRE_MEMORY_HOST);
            interface_rank_stencils    =  hypre_TAlloc(HYPRE_Int *,  cnt1, HYPRE_MEMORY_HOST);
            interface_stencil_ranks    =  hypre_TAlloc(HYPRE_Int *,  cnt1, HYPRE_MEMORY_HOST);
            coarse_stencil_cnt         =  hypre_CTAlloc(HYPRE_Int,  cnt1, HYPRE_MEMORY_HOST);

            k = 0;
            for (i = 0; i < cnt1; i++)
            {
               /*-----------------------------------------------------------------
                * for each coarse interface node, we get a stencil. We compute only
                * the ranks assuming a maximum size stencil of 27.
                *-----------------------------------------------------------------*/
               interface_max_stencil_ranks[i] = hypre_TAlloc(HYPRE_Int,  box_to_ranks_cnt[i], HYPRE_MEMORY_HOST);
               interface_max_stencil_cnt[i]  = hypre_CTAlloc(HYPRE_Int,  max_stencil_size, HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * conjugate the coarse node index for determining the stencil
                * shapes for the Uentry connections.
                *-----------------------------------------------------------------*/
               hypre_CopyIndex(cindex[i], index1);
               hypre_SetIndex3(index1, -index1[0], -index1[1], -index1[2]);

               n = 0;
               for (j = 0; j < box_ranks_cnt[i]; j++)
               {
                  /*--------------------------------------------------------------
                   * extract the row rank for a given Uventry. Note that these
                   * are the ranks in the grid of A. Therefore, we grab the index
                   * from the nested_graph Uventry to determine the global rank.
                   * With the rank, find the corresponding Uventry of the graph
                   * of A. The to_ranks now can be extracted out.
                   *--------------------------------------------------------------*/
                  Uventry = Uventries[ coarse_contrib_Uv[i][j] ];
                  hypre_CopyIndex(hypre_SStructUVEntryIndex(Uventry), index);

                  hypre_SStructGridFindBoxManEntry(grid, part_fine, index, var1, &boxman_entry);
                  hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &rank, matrix_type);

                  Uventry = hypre_SStructGraphUVEntry(graph, rank - startrank);
                  nUentries = hypre_SStructUVEntryNUEntries(Uventry);

                  for (l = 0; l < nUentries; l++)
                  {
                     if (hypre_SStructUVEntryToPart(Uventry, l) == part_crse)
                     {
                        to_rank  = hypre_SStructUVEntryToRank(Uventry, l);
                        rows[k]  = rank;
                        cols[k++] = to_rank;

                        /*---------------------------------------------------------
                         * compute stencil shape for this Uentry.
                         *---------------------------------------------------------*/
                        hypre_CopyIndex( hypre_SStructUVEntryToIndex(Uventry, l),
                                         index );
                        hypre_AddIndexes(index, index1, 3, index2);

                        MapStencilRank(index2, m);
                        interface_max_stencil_ranks[i][n++] = m;
                        interface_max_stencil_cnt[i][m]++;
                     }
                  }
               }
               hypre_TFree(coarse_contrib_Uv[i], HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * Determine only the distinct stencil ranks for coarse node i.
                *-----------------------------------------------------------------*/
               l = 0;
               for (j = 0; j < max_stencil_size; j++)
               {
                  if (interface_max_stencil_cnt[i][j])
                  {
                     l++;
                  }
               }

               coarse_stencil_cnt[i] = l;
               interface_stencil_ranks[i] = hypre_TAlloc(HYPRE_Int,  l, HYPRE_MEMORY_HOST);
               interface_rank_stencils[i] = hypre_TAlloc(HYPRE_Int,  max_stencil_size, HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * For each stencil rank, assign one of the stencil_shape_i index.
                *-----------------------------------------------------------------*/
               l = 0;
               for (j = 0; j < max_stencil_size; j++)
               {
                  if (interface_max_stencil_cnt[i][j])
                  {
                     interface_rank_stencils[i][j] = l;
                     interface_stencil_ranks[i][l] = j;
                     l++;
                  }
               }
            }   /* for (i= 0; i< cnt1; i++) */

            hypre_TFree(coarse_contrib_Uv, HYPRE_MEMORY_HOST);
            hypre_TFree(box_ranks_cnt, HYPRE_MEMORY_HOST);
            hypre_TFree(cindex, HYPRE_MEMORY_HOST);

            /*-----------------------------------------------------------------
             * Extract data from IJ matrix
             *-----------------------------------------------------------------*/
            HYPRE_IJMatrixGetValues(ij_A, nrows, ncols, rows, cols, vals);

            hypre_TFree(ncols, HYPRE_MEMORY_HOST);
            hypre_TFree(rows, HYPRE_MEMORY_HOST);
            hypre_TFree(cols, HYPRE_MEMORY_HOST);

            /*-----------------------------------------------------------------
             *  Steps 4 & 5:
             *  Compute the arithmetically averaged interface stencils,
             *  and determine the interface stencil weights.
             *
             *    stencil_vals[l]       = averaged stencil coeff for interface
             *                            stencil entry l.
             *    common_rank_stencils  = final structured coarse stencil entries
             *                            for the stencil_shapes that the
             *                            interface stencils must collapse to.
             *    common_stencil_ranks  = final structured coarse stencil_shape
             *                            ranks for the stencil_shapes that the
             *                            interface stencils must collapse to.
             *    common_stencil_i      = stencil entry of the interface stencil
             *                            corresponding to the common
             *                            stencil_shape.
             *-----------------------------------------------------------------*/
            k = 0;
            for (i = 0; i < cnt1; i++)
            {
               stencil_vals = hypre_CTAlloc(HYPRE_Real,  coarse_stencil_cnt[i], HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * Compute the arithmetic stencil averages for coarse node i.
                *-----------------------------------------------------------------*/
               for (j = 0; j < box_to_ranks_cnt[i]; j++)
               {
                  m = interface_max_stencil_ranks[i][j];
                  l = interface_rank_stencils[i][m];
                  stencil_vals[l] += vals[k] / interface_max_stencil_cnt[i][m];
                  k++;
               }
               hypre_TFree(interface_max_stencil_ranks[i], HYPRE_MEMORY_HOST);
               hypre_TFree(interface_max_stencil_cnt[i], HYPRE_MEMORY_HOST);
               hypre_TFree(interface_rank_stencils[i], HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * Determine which stencil has to be formed. This is accomplished
                * by comparing the coarse grid stencil ranks with the computed
                * interface stencil ranks. We qsort (if there are more than one
                * rank) the ranks to give quick comparisons. Note that we need
                * to swap the elements of stencil_vals & fine_interface_ranks[i]'s
                * accordingly.
                *-----------------------------------------------------------------*/

               sort = falseV;
               for (j = 0; j < (coarse_stencil_cnt[i] - 1); j++)
               {
                  if (interface_stencil_ranks[i][j] > interface_stencil_ranks[i][j + 1])
                  {
                     sort = trueV;
                     break;
                  }
               }

               if ( (coarse_stencil_cnt[i] > 1) && (sort == trueV) )
               {
                  temp1 = hypre_TAlloc(HYPRE_Int,  coarse_stencil_cnt[i], HYPRE_MEMORY_HOST);
                  for (j = 0; j < coarse_stencil_cnt[i]; j++)
                  {
                     temp1[j] = j;
                  }

                  hypre_qsort1(interface_stencil_ranks[i], (HYPRE_Real *) temp1, 0,
                               coarse_stencil_cnt[i] - 1);

                  /*---------------------------------------------------------------
                   * swap the stencil_vals to agree with the rank swapping.
                   *---------------------------------------------------------------*/
                  temp3  = hypre_TAlloc(HYPRE_Real,  coarse_stencil_cnt[i], HYPRE_MEMORY_HOST);
                  for (j = 0; j < coarse_stencil_cnt[i]; j++)
                  {
                     m         = temp1[j];
                     temp3[j]  = stencil_vals[m];
                  }
                  for (j = 0; j < coarse_stencil_cnt[i]; j++)
                  {
                     stencil_vals[j] = temp3[j];
                  }

                  hypre_TFree(temp1, HYPRE_MEMORY_HOST);
                  hypre_TFree(temp3, HYPRE_MEMORY_HOST);
               }

               /*-----------------------------------------------------------------
                * Compute the weights for the averaged stencil contributions.
                * We need to convert the ranks back to stencil_shapes and then
                * find the abs of the stencil shape.
                *-----------------------------------------------------------------*/
               temp3 = hypre_TAlloc(HYPRE_Real,  coarse_stencil_cnt[i], HYPRE_MEMORY_HOST);
               for (j = 0; j < coarse_stencil_cnt[i]; j++)
               {
                  InverseMapStencilRank(interface_stencil_ranks[i][j], index_temp);
                  AbsStencilShape(index_temp, abs_stencil_shape);
                  temp3[j] = weights[abs_stencil_shape];
               }

               /*-----------------------------------------------------------------
                * Compare the coarse stencil and the interface stencil and
                * extract the common stencil shapes.
                * WE ARE ASSUMING THAT THE COARSE INTERFACE STENCIL HAS SOME
                * COMMON STENCIL SHAPE WITH THE COARSE STENCIL.
                *-----------------------------------------------------------------*/
               common_rank_stencils = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
               common_stencil_ranks = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
               common_stencil_i    = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);

               l = 0;
               m = 0;
               for (j = 0; j < stencil_size; j++)
               {
                  while (  (l < coarse_stencil_cnt[i])
                           && (stencil_ranks[j] > interface_stencil_ranks[i][l]) )
                  {
                     l++;
                  }

                  if (l >= coarse_stencil_cnt[i])
                  {
                     break;
                  }
                  /*--------------------------------------------------------------
                   * Check if a common stencil shape rank has been found.
                   *--------------------------------------------------------------*/
                  if (   (stencil_ranks[j] == interface_stencil_ranks[i][l])
                         && (l < coarse_stencil_cnt[i]) )
                  {
                     common_rank_stencils[m] = rank_stencils[ stencil_ranks[j] ];
                     common_stencil_ranks[m] = stencil_ranks[j];
                     common_stencil_i[m++]  = l;
                     l++;
                  }
               }
               /*-----------------------------------------------------------------
                * Find the contribution and weights for the averaged stencils.
                *-----------------------------------------------------------------*/
               for (j = 0; j < m; j++)
               {
                  hypre_CopyIndex(hypre_StructStencilElement(
                                     stencils, common_rank_stencils[j]),
                                  stencil_shape_i);
                  AbsStencilShape(stencil_shape_i, abs_stencil_shape);

                  crse_ptr = hypre_StructMatrixExtractPointerByIndex(crse_smatrix,
                                                                     parents_cnodes[i],
                                                                     stencil_shape_i);

                  /*-----------------------------------------------------------------
                   *  For a compact stencil (e.g., -1 <= hypre_Index[i] <= 1, i= 0-2),
                   *  the value of abs_stencil_shape can be used to determine the
                   *  stencil:
                   *     abs_stencil_shape=   3   only corners in 3-d
                   *                          2   corners in 2-d; or the centre plane
                   *                              in 3-d, or e,w,n,s of the bottom
                   *                              or top plane in 3-d
                   *                          1   e,w in 1-d; or e,w,n,s in 2-d;
                   *                              or the centre plane in 3-d,
                   *                              or c of the bottom or top plane
                   *                              in 3-d
                   *                          0   c in 1-d, 2-d, or 3-d.
                   *-----------------------------------------------------------------*/

                  switch (abs_stencil_shape)
                  {
                     case 3:    /* corners of 3-d stencil */

                        l = common_stencil_i[j];
                        crse_ptr[fine_interface_ranks[i]] = stencil_vals[l];

                        break;


                     case 2:    /* corners in 2-d or edges in 3-d */

                        if (ndim == 2)
                        {
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = stencil_vals[l];
                        }

                        else if (ndim == 3)
                        {
                           /*----------------------------------------------------------
                            * The edge values are weighted sums of the averaged
                            * coefficients. The weights and averaged coefficients must
                            * be found. The contributions are found using the stencil
                            * ranks and the stencil ordering
                            * top: 14  12  13  centre:  5  3  4  bottom 23   21   22
                            *      11   9  10           2  0  1         20   18   19
                            *      17  15  16           8  6  7         26   24   25
                            *----------------------------------------------------------*/
                           l    =  common_stencil_ranks[j];
                           temp1 =  hypre_TAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);

                           switch (l)
                           {
                              case 4:   /* centre plane ne */

                                 temp1[0] = 13;
                                 temp1[1] = 22;
                                 break;

                              case 5:   /* centre plane nw */

                                 temp1[0] = 14;
                                 temp1[1] = 23;
                                 break;

                              case 7:   /* centre plane se */

                                 temp1[0] = 16;
                                 temp1[1] = 25;
                                 break;

                              case 8:   /* centre plane sw */

                                 temp1[0] = 17;
                                 temp1[1] = 26;
                                 break;

                              case 10:   /* top plane e */

                                 temp1[0] = 13;
                                 temp1[1] = 16;
                                 break;

                              case 11:   /* top plane w */

                                 temp1[0] = 14;
                                 temp1[1] = 17;
                                 break;

                              case 12:   /* top plane n */

                                 temp1[0] = 13;
                                 temp1[1] = 14;
                                 break;

                              case 15:   /* top plane s */

                                 temp1[0] = 16;
                                 temp1[1] = 17;
                                 break;

                              case 19:   /* bottom plane e */

                                 temp1[0] = 22;
                                 temp1[1] = 25;
                                 break;

                              case 20:   /* bottom plane w */

                                 temp1[0] = 23;
                                 temp1[1] = 26;
                                 break;

                              case 21:   /* bottom plane n */

                                 temp1[0] = 22;
                                 temp1[1] = 23;
                                 break;

                              case 24:   /* bottom plane s */

                                 temp1[0] = 25;
                                 temp1[1] = 26;
                                 break;
                           }


                           /*-------------------------------------------------------
                            *  Add up the weighted contributions of the interface
                            *  stencils. This involves searching the ranks of
                            *  interface_stencil_ranks. The weights must be averaged.
                            *-------------------------------------------------------*/

                           l = common_stencil_i[j];
                           sum = temp3[l];
                           sum_contrib = sum * stencil_vals[l];

                           n = 1;
                           for (l = 0; l < 2; l++)
                           {
                              while (  (n < coarse_stencil_cnt[i])
                                       && (interface_stencil_ranks[i][n] < temp1[l]) )
                              {
                                 n++;
                              }

                              if (n >= coarse_stencil_cnt[i])
                              {
                                 break;
                              }

                              if (interface_stencil_ranks[i][n] == temp1[l])
                              {
                                 sum += temp3[n];
                                 sum_contrib += temp3[n] * stencil_vals[n];
                                 n++;
                              }
                           }

                           sum_contrib /= sum;   /* average out the weights */
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = sum_contrib;

                           hypre_TFree(temp1, HYPRE_MEMORY_HOST);

                        }    /* else if (ndim == 3) */

                        break;

                     case 1:     /* e,w in 1-d, or edges in 2-d, or faces in 3-d */

                        if (ndim == 1)
                        {
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = stencil_vals[l];
                        }

                        else if (ndim == 2)
                        {
                           l    =  common_stencil_ranks[j];
                           temp1 =  hypre_TAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);

                           switch (l)
                           {
                              case 1:   /* e */

                                 temp1[0] = 4;
                                 temp1[1] = 7;
                                 break;

                              case 2:   /* w */

                                 temp1[0] = 5;
                                 temp1[1] = 8;
                                 break;

                              case 3:   /* n */

                                 temp1[0] = 4;
                                 temp1[1] = 5;
                                 break;

                              case 6:   /* s */

                                 temp1[0] = 7;
                                 temp1[1] = 8;
                                 break;
                           }

                           /*-------------------------------------------------------
                            *  Add up the weighted contributions of the interface
                            *  stencils.
                            *-------------------------------------------------------*/

                           l = common_stencil_i[j];
                           sum = temp3[l];
                           sum_contrib = sum * stencil_vals[l];

                           n = 1;
                           for (l = 0; l < 2; l++)
                           {
                              while (  (n < coarse_stencil_cnt[i])
                                       && (interface_stencil_ranks[i][n] < temp1[l]) )
                              {
                                 n++;
                              }

                              if (n >= coarse_stencil_cnt[i])
                              {
                                 break;
                              }

                              if (interface_stencil_ranks[i][n] == temp1[l])
                              {
                                 sum += temp3[n];
                                 sum_contrib += temp3[n] * stencil_vals[n];
                                 n++;
                              }
                           }

                           sum_contrib /= sum;   /* average out the weights */
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = sum_contrib;

                           hypre_TFree(temp1, HYPRE_MEMORY_HOST);

                        }   /* else if (ndim == 2) */

                        else /* 3-d */
                        {
                           l    =  common_stencil_ranks[j];
                           temp1 =  hypre_TAlloc(HYPRE_Int,  8, HYPRE_MEMORY_HOST);

                           switch (l)
                           {
                              case 1:   /* centre plane e */

                                 temp1[0] = 4;
                                 temp1[1] = 7;
                                 temp1[2] = 10;
                                 temp1[3] = 13;
                                 temp1[4] = 16;
                                 temp1[5] = 19;
                                 temp1[6] = 22;
                                 temp1[7] = 25;
                                 break;

                              case 2:   /* centre plane w */

                                 temp1[0] = 5;
                                 temp1[1] = 8;
                                 temp1[2] = 11;
                                 temp1[3] = 14;
                                 temp1[4] = 17;
                                 temp1[5] = 20;
                                 temp1[6] = 23;
                                 temp1[7] = 26;
                                 break;

                              case 3:   /* centre plane n */

                                 temp1[0] = 4;
                                 temp1[1] = 5;
                                 temp1[2] = 12;
                                 temp1[3] = 13;
                                 temp1[4] = 14;
                                 temp1[5] = 21;
                                 temp1[6] = 22;
                                 temp1[7] = 23;
                                 break;

                              case 6:   /* centre plane s */

                                 temp1[0] = 7;
                                 temp1[1] = 8;
                                 temp1[2] = 15;
                                 temp1[3] = 16;
                                 temp1[4] = 17;
                                 temp1[5] = 24;
                                 temp1[6] = 25;
                                 temp1[7] = 26;
                                 break;

                              case 9:   /* top plane c */

                                 for (n = 0; n < 8; n++)
                                 {
                                    temp1[n] = 10 + n;
                                 }
                                 break;

                              case 18:   /* bottom plane c */

                                 for (n = 0; n < 8; n++)
                                 {
                                    temp1[n] = 19 + n;
                                 }
                                 break;

                           }

                           /*-------------------------------------------------------
                            *  Add up the weighted contributions of the interface
                            *  stencils.
                            *-------------------------------------------------------*/

                           l = common_stencil_i[j];
                           sum = temp3[l];
                           sum_contrib = sum * stencil_vals[l];

                           n = 1;
                           for (l = 0; l < 8; l++)
                           {
                              while (   (n < coarse_stencil_cnt[i])
                                        && (interface_stencil_ranks[i][n] < temp1[l]) )
                              {
                                 n++;
                              }

                              if (n >= coarse_stencil_cnt[i])
                              {
                                 break;
                              }

                              if (interface_stencil_ranks[i][n] == temp1[l])
                              {
                                 sum += temp3[n];
                                 sum_contrib += temp3[n] * stencil_vals[n];
                                 n++;
                              }
                           }

                           sum_contrib /= sum;   /* average out the weights */
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = sum_contrib;

                           hypre_TFree(temp1, HYPRE_MEMORY_HOST);

                        }    /* else */

                        break;

                  }   /* switch(abs_stencil_shape) */
               }       /* for (j= 0; j< m; j++) */

               hypre_TFree(interface_stencil_ranks[i], HYPRE_MEMORY_HOST);

               hypre_TFree(stencil_vals, HYPRE_MEMORY_HOST);
               hypre_TFree(temp3, HYPRE_MEMORY_HOST);
               hypre_TFree(common_rank_stencils, HYPRE_MEMORY_HOST);
               hypre_TFree(common_stencil_ranks, HYPRE_MEMORY_HOST);
               hypre_TFree(common_stencil_ranks, HYPRE_MEMORY_HOST);
               hypre_TFree(common_stencil_i, HYPRE_MEMORY_HOST);

            }          /* for (i= 0; i< cnt1; i++) */

            hypre_TFree(box_to_ranks_cnt, HYPRE_MEMORY_HOST);
            hypre_TFree(interface_max_stencil_ranks, HYPRE_MEMORY_HOST);
            hypre_TFree(interface_max_stencil_cnt, HYPRE_MEMORY_HOST);
            hypre_TFree(interface_rank_stencils, HYPRE_MEMORY_HOST);
            hypre_TFree(interface_stencil_ranks, HYPRE_MEMORY_HOST);
            hypre_TFree(coarse_stencil_cnt, HYPRE_MEMORY_HOST);
            hypre_TFree(fine_interface_ranks, HYPRE_MEMORY_HOST);
            hypre_TFree(parents_cnodes, HYPRE_MEMORY_HOST);
            hypre_TFree(vals, HYPRE_MEMORY_HOST);

            /*-----------------------------------------------------------
             *  Box fi is completed.
             *-----------------------------------------------------------*/
         }     /* for (fi= 0; fi< box_array_size; fi++) */

         hypre_TFree(stencil_ranks, HYPRE_MEMORY_HOST);
         hypre_TFree(rank_stencils, HYPRE_MEMORY_HOST);
         hypre_TFree(cdata_space_ranks, HYPRE_MEMORY_HOST);
         hypre_TFree(box_graph_cnts, HYPRE_MEMORY_HOST);
         for (i = 0; i < box_array_size; i++)
         {
            if (box_graph_indices[i])
            {
               hypre_TFree(box_graph_indices[i], HYPRE_MEMORY_HOST);
            }
         }
         hypre_TFree(box_graph_indices, HYPRE_MEMORY_HOST);

         hypre_TFree(box_starts, HYPRE_MEMORY_HOST);
         hypre_TFree(box_ends, HYPRE_MEMORY_HOST);
      }  /* for (var1= 0; var1< nvars; var1++) */
   }    /* if (nUventries > 0) */


   /*--------------------------------------------------------------------------
    *  STEP 3:
    *        Coarsened f/c interface coefficients can be used to create the
    *        centre components along the coarsened f/c nodes now. Loop over
    *        the coarsened fbox_bdy's and set the centre stencils.
    *--------------------------------------------------------------------------*/
   hypre_ClearIndex(index_temp);
   for (var1 = 0; var1 < nvars; var1++)
   {
      /* only like variables couple. */
      smatrix_var  = hypre_SStructPMatrixSMatrix(A_crse, var1, var1);
      stencils     = hypre_SStructPMatrixSStencil(A_crse, var1, var1);
      stencil_size = hypre_StructStencilSize(stencils);
      a_ptrs       = hypre_TAlloc(HYPRE_Real *,  stencil_size, HYPRE_MEMORY_HOST);

      rank_stencils = hypre_TAlloc(HYPRE_Int,  max_stencil_size, HYPRE_MEMORY_HOST);
      for (i = 0; i < stencil_size; i++)
      {
         hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                         stencil_shape_i);
         MapStencilRank(stencil_shape_i, rank);
         rank_stencils[rank] = i;
      }
      centre = rank_stencils[0];

      cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);

      hypre_ForBoxI(ci, cgrid_boxes)
      {
         A_dbox     = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(smatrix_var), ci);
         fbox_bdy_ci = fbox_bdy[var1][ci];

         for (i = 0; i < stencil_size; i++)
         {
            hypre_CopyIndex(hypre_StructStencilElement(stencils, i),
                            stencil_shape_i);
            a_ptrs[i] = hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                ci,
                                                                stencil_shape_i);
         }

         /*------------------------------------------------------------------
          * Loop over the boundaries of each patch inside cgrid_box ci.
          * These patch boxes must be coarsened to get the correct extents.
          *------------------------------------------------------------------*/
         hypre_ForBoxArrayI(arrayi, fbox_bdy_ci)
         {
            fbox_bdy_ci_fi = hypre_BoxArrayArrayBoxArray(fbox_bdy_ci, arrayi);
            hypre_ForBoxI(fi, fbox_bdy_ci_fi)
            {
               fgrid_box = hypre_BoxArrayBox(fbox_bdy_ci_fi, fi);
               hypre_StructMapFineToCoarse(hypre_BoxIMin(fgrid_box), index_temp,
                                           stridef, hypre_BoxIMin(&fine_box));
               hypre_StructMapFineToCoarse(hypre_BoxIMax(fgrid_box), index_temp,
                                           stridef, hypre_BoxIMax(&fine_box));

               hypre_CopyIndex(hypre_BoxIMin(&fine_box), cstart);
               hypre_BoxGetSize(&fine_box, loop_size);

#define DEVICE_VAR is_device_ptr(a_ptrs)
               hypre_BoxLoop1Begin(ndim, loop_size,
                                   A_dbox, cstart, stridec, iA);
               {
                  HYPRE_Int i;
                  for (i = 0; i < stencil_size; i++)
                  {
                     if (i != centre)
                     {
                        a_ptrs[centre][iA] -= a_ptrs[i][iA];
                     }
                  }
               }
               hypre_BoxLoop1End(iA);
#undef DEVICE_VAR

            }  /* hypre_ForBoxI(fi, fbox_bdy_ci_fi) */
         }      /* hypre_ForBoxArrayI(arrayi, fbox_bdy_ci) */
      }          /* hypre_ForBoxI(ci, cgrid_boxes) */

      hypre_TFree(a_ptrs, HYPRE_MEMORY_HOST);
      hypre_TFree(rank_stencils, HYPRE_MEMORY_HOST);

   }  /* for (var1= 0; var1< nvars; var1++) */

   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = hypre_StructGridBoxes(cgrid);

      fgrid = hypre_SStructPGridSGrid(hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      fgrid_boxes = hypre_StructGridBoxes(fgrid);

      hypre_ForBoxI(ci, cgrid_boxes)
      {
         hypre_BoxArrayDestroy(fgrid_crse_extents[var1][ci]);
         hypre_BoxArrayDestroy(fbox_interior[var1][ci]);
         hypre_BoxArrayArrayDestroy(fbox_bdy[var1][ci]);
         hypre_TFree(interior_fboxi[var1][ci], HYPRE_MEMORY_HOST);
         hypre_TFree(bdy_fboxi[var1][ci], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(fgrid_crse_extents[var1], HYPRE_MEMORY_HOST);
      hypre_TFree(fbox_interior[var1], HYPRE_MEMORY_HOST);
      hypre_TFree(fbox_bdy[var1], HYPRE_MEMORY_HOST);
      hypre_TFree(interior_fboxi[var1], HYPRE_MEMORY_HOST);
      hypre_TFree(bdy_fboxi[var1], HYPRE_MEMORY_HOST);

      hypre_ForBoxI(fi, fgrid_boxes)
      {
         hypre_TFree(cboxi_fboxes[var1][fi], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(cboxi_fboxes[var1], HYPRE_MEMORY_HOST);
      hypre_TFree(cboxi_fcnt[var1], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(fgrid_crse_extents, HYPRE_MEMORY_HOST);
   hypre_TFree(fbox_interior, HYPRE_MEMORY_HOST);
   hypre_TFree(fbox_bdy, HYPRE_MEMORY_HOST);
   hypre_TFree(interior_fboxi, HYPRE_MEMORY_HOST);
   hypre_TFree(bdy_fboxi, HYPRE_MEMORY_HOST);
   hypre_TFree(cboxi_fboxes, HYPRE_MEMORY_HOST);
   hypre_TFree(cboxi_fcnt, HYPRE_MEMORY_HOST);

   return 0;
}
